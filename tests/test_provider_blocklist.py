"""Coverage for BrevConfig.exclude_providers (3.15.0).

Brev support confirmed (issue #62 thread) that the OCI launchpad path
fails server-side on most orgs. We default to filtering it out of the
auto-pick selector so users don't burn time on a guaranteed failure.
"""

import json
from unittest import mock

import pytest

from runplz import App, BrevConfig, Image
from runplz.backends import brev


def _function(min_gpu_memory=24):
    app = App("demo")

    @app.function(
        image=Image.from_registry("ubuntu:22.04"),
        min_gpu_memory=min_gpu_memory,
    )
    def f():  # pragma: no cover
        pass

    return app.functions["f"]


def _search_rows(*type_strings):
    return json.dumps(
        [{"type": t, "hourly_price": 1.0 + i * 0.01} for i, t in enumerate(type_strings)]
    )


# ---------------------------------------------------------------------------
# Default config blocks OCI


def test_default_brev_config_excludes_oci():
    cfg = BrevConfig()
    assert "oci" in cfg.exclude_providers


def test_default_excludes_oci_from_picked_types():
    fn = _function()
    rows = _search_rows(
        "oci.a100x8.sxm.brev-dgxc",
        "verda_A100_sxm4_80Gx8",
        "massedcompute_A100_sxm4_80Gx8",
    )
    fake = mock.Mock(returncode=0, stdout=rows, stderr="")
    with mock.patch("runplz.backends.brev._brev_capture", return_value=fake):
        picks = brev._pick_instance_types(fn, n=3, exclude_providers=("oci",))
    assert all(not p.startswith("oci") for p in picks)
    # The non-OCI rows still come through.
    assert "verda_A100_sxm4_80Gx8" in picks
    assert "massedcompute_A100_sxm4_80Gx8" in picks


def test_excluding_with_no_matching_rows_returns_empty():
    """If `brev search` returns ONLY excluded providers, the picker should
    return empty so the caller can raise the standard 'no matches' error."""
    fn = _function()
    rows = _search_rows("oci.a100x8.sxm.brev-dgxc", "oci.h100x8.brev-dgxc")
    fake = mock.Mock(returncode=0, stdout=rows, stderr="")
    with mock.patch("runplz.backends.brev._brev_capture", return_value=fake):
        picks = brev._pick_instance_types(fn, n=3, exclude_providers=("oci",))
    assert picks == []


def test_empty_exclude_keeps_oci():
    """A user can opt back in to OCI by passing exclude_providers=()."""
    fn = _function()
    rows = _search_rows("oci.a100x8.sxm.brev-dgxc", "verda_A100_sxm4_80Gx8")
    fake = mock.Mock(returncode=0, stdout=rows, stderr="")
    with mock.patch("runplz.backends.brev._brev_capture", return_value=fake):
        picks = brev._pick_instance_types(fn, n=3, exclude_providers=())
    assert "oci.a100x8.sxm.brev-dgxc" in picks


# ---------------------------------------------------------------------------
# Match logic


def test_matches_excluded_provider_is_segment_aware():
    """Prefix `oci` matches `oci.a100...` but not `ocifoo` — boundary
    must be `.`, `_`, `-`, or end-of-string."""
    assert brev._matches_excluded_provider("oci.a100x8.sxm.brev-dgxc", ("oci",))
    assert brev._matches_excluded_provider("OCI.A100X8", ("oci",))  # case-insensitive
    assert brev._matches_excluded_provider("oci_a100", ("oci",))
    assert brev._matches_excluded_provider("oci-thing", ("oci",))
    assert brev._matches_excluded_provider("oci", ("oci",))
    # Not a provider-segment match — the `oci` is part of a longer word.
    assert not brev._matches_excluded_provider("ocifoo", ("oci",))
    assert not brev._matches_excluded_provider("massedcompute_oci_x8", ("oci",))
    assert not brev._matches_excluded_provider(None, ("oci",))
    assert not brev._matches_excluded_provider("anything", ())


def test_user_pinned_instance_type_bypasses_blocklist():
    """When the user explicitly pins instance_type=, runplz must respect
    that even if it's on the blocklist — the user is overriding our default."""
    cfg = BrevConfig(
        auto_create_instances=True,
        instance_type="oci.a100x8.sxm.brev-dgxc",
        mode="vm",
    )
    captured = {}

    def fake_run(cmd, *a, **kw):
        captured.setdefault("cmds", []).append(list(cmd))
        if cmd[:2] == ["brev", "create"]:
            return mock.Mock(returncode=0, stdout="", stderr="")
        return mock.Mock(returncode=0, stdout="", stderr="")

    fn = mock.Mock(
        name="t",
        gpu=None,
        min_cpu=None,
        min_memory=None,
        min_gpu_memory=None,
        min_disk=None,
        num_gpus=1,
    )
    img = Image.from_registry("ubuntu:22.04")
    with mock.patch("runplz.backends.brev.subprocess.run", fake_run):
        with mock.patch("runplz.backends.brev._verify_post_action_state"):
            with mock.patch("time.sleep", lambda _s: None):
                brev._create_instance("pinned-oci", cfg=cfg, image=img, function=fn)

    create_cmds = [c for c in captured["cmds"] if c[:2] == ["brev", "create"]]
    assert any("oci.a100x8.sxm.brev-dgxc" in c for c in create_cmds), (
        "user-pinned OCI type should still be passed to brev create"
    )


# ---------------------------------------------------------------------------
# Validation


def test_brev_config_rejects_non_tuple_exclude_providers():
    with pytest.raises(ValueError, match="must be a tuple"):
        BrevConfig(exclude_providers=["oci"])


def test_brev_config_rejects_empty_string_exclude_provider():
    with pytest.raises(ValueError, match="non-empty strings"):
        BrevConfig(exclude_providers=("",))
