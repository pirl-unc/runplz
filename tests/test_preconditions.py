"""Coverage for the optional remote precondition probe (issue #56)."""

from unittest import mock

import pytest

from runplz import App, Image
from runplz.app import _normalize_preconditions
from runplz.backends import _ssh_common

# ---------------------------------------------------------------------------
# Function API: validation


def test_function_accepts_known_precondition_keys():
    app = App("demo")

    @app.function(
        image=Image.from_registry("ubuntu:22.04"),
        preconditions={"shm_gb": 4.0, "disk_free_gb": 50},
    )
    def f():  # pragma: no cover
        pass

    assert app.functions["f"].preconditions == {"shm_gb": 4.0, "disk_free_gb": 50.0}


def test_function_rejects_unknown_precondition_key():
    app = App("demo")
    with pytest.raises(ValueError, match="unknown precondition key"):

        @app.function(
            image=Image.from_registry("ubuntu:22.04"),
            preconditions={"shm_gib": 4.0},  # typo
        )
        def f():  # pragma: no cover
            pass


def test_function_rejects_non_positive_value():
    app = App("demo")
    with pytest.raises(ValueError, match="must be a positive number"):

        @app.function(
            image=Image.from_registry("ubuntu:22.04"),
            preconditions={"shm_gb": 0},
        )
        def f():  # pragma: no cover
            pass


def test_normalize_preconditions_none_returns_empty():
    assert _normalize_preconditions("f", None) == {}


def test_normalize_preconditions_rejects_non_dict():
    with pytest.raises(ValueError, match="must be a dict"):
        _normalize_preconditions("f", [("shm_gb", 4)])


# ---------------------------------------------------------------------------
# Probe runner: warn vs fail


def _probe_stdout(*, shm_bytes=None, home_bytes=None, gpus=None, vram_mib=None):
    parts = ["---SHM_BYTES---"]
    if shm_bytes is not None:
        parts.append(str(shm_bytes))
    parts.append("---HOME_FREE_BYTES---")
    if home_bytes is not None:
        parts.append(str(home_bytes))
    parts.append("---GPU_COUNT---")
    parts.append(str(gpus if gpus is not None else 0))
    parts.append("---GPU_MIN_VRAM_MIB---")
    parts.append(str(vram_mib if vram_mib is not None else 0))
    parts.append("---END---")
    return "\n".join(parts) + "\n"


def test_check_preconditions_no_op_when_empty():
    # Empty dict must not even ssh.
    with mock.patch("runplz.backends._ssh_common._ssh_capture") as ssh_mock:
        _ssh_common._check_preconditions("box", {})
    ssh_mock.assert_not_called()


def test_check_preconditions_passes_when_observed_meets_minimum(capsys):
    # 4 GiB shm, 100 GiB disk free → exceeds the demands.
    stdout = _probe_stdout(
        shm_bytes=4 * 1024**3, home_bytes=100 * 1024**3, gpus=8, vram_mib=80 * 1024
    )
    with mock.patch("runplz.backends._ssh_common._ssh_capture", return_value=stdout):
        _ssh_common._check_preconditions(
            "box",
            {"shm_gb": 4.0, "disk_free_gb": 50.0, "gpu_count": 8, "gpu_memory_gb": 40},
        )
    assert "precondition warning" not in capsys.readouterr().out


def test_check_preconditions_warns_when_below_min_but_above_half(capsys):
    # 3 GiB shm vs declared 4 GB → below min, but above 50% → warn only.
    stdout = _probe_stdout(shm_bytes=3 * 1024**3, home_bytes=100 * 1024**3)
    with mock.patch("runplz.backends._ssh_common._ssh_capture", return_value=stdout):
        _ssh_common._check_preconditions("box", {"shm_gb": 4.0})
    out = capsys.readouterr().out
    assert "precondition warning" in out
    assert "shm_gb" in out


def test_check_preconditions_fails_when_below_half(capsys):
    # 1 GiB shm vs declared 4 GB → below 50% → hard fail.
    stdout = _probe_stdout(shm_bytes=1 * 1024**3, home_bytes=100 * 1024**3)
    with mock.patch("runplz.backends._ssh_common._ssh_capture", return_value=stdout):
        with pytest.raises(_ssh_common.PreconditionFailed, match="shm_gb"):
            _ssh_common._check_preconditions("box", {"shm_gb": 4.0})


def test_check_preconditions_warns_when_observed_unparseable(capsys):
    # Probe section empty (e.g. /dev/shm not mounted) → warn, don't fail.
    stdout = _probe_stdout(shm_bytes=None, home_bytes=100 * 1024**3)
    with mock.patch("runplz.backends._ssh_common._ssh_capture", return_value=stdout):
        _ssh_common._check_preconditions("box", {"shm_gb": 4.0})
    assert "could not probe shm_gb" in capsys.readouterr().out


def test_check_preconditions_gpu_count_below_half_fails():
    stdout = _probe_stdout(gpus=1)  # vs 4 declared → below 50% → fail
    with mock.patch("runplz.backends._ssh_common._ssh_capture", return_value=stdout):
        with pytest.raises(_ssh_common.PreconditionFailed, match="gpu_count"):
            _ssh_common._check_preconditions("box", {"gpu_count": 4})


def test_check_preconditions_gpu_memory_translates_mib_to_gb(capsys):
    # 24 GiB → 24576 MiB; declared min 80 GB; 24 < 40 (50% threshold) → fail.
    stdout = _probe_stdout(vram_mib=24 * 1024)
    with mock.patch("runplz.backends._ssh_common._ssh_capture", return_value=stdout):
        with pytest.raises(_ssh_common.PreconditionFailed, match="gpu_memory_gb"):
            _ssh_common._check_preconditions("box", {"gpu_memory_gb": 80})


def test_check_preconditions_aggregates_multiple_failures():
    stdout = _probe_stdout(shm_bytes=512 * 1024**2, home_bytes=1 * 1024**3, gpus=0)
    with mock.patch("runplz.backends._ssh_common._ssh_capture", return_value=stdout):
        with pytest.raises(_ssh_common.PreconditionFailed) as ei:
            _ssh_common._check_preconditions(
                "box",
                {"shm_gb": 4.0, "disk_free_gb": 50.0, "gpu_count": 8},
            )
    msg = str(ei.value)
    assert "shm_gb" in msg
    assert "disk_free_gb" in msg
    assert "gpu_count" in msg


# ---------------------------------------------------------------------------
# Helpers


def test_first_int_pulls_leading_number():
    assert _ssh_common._first_int("  4096\n") == 4096
    assert _ssh_common._first_int("") is None
    assert _ssh_common._first_int(None) is None
    assert _ssh_common._first_int("noise") is None


def test_bytes_to_gb_handles_zero_and_none():
    assert _ssh_common._bytes_to_gb(None) is None
    assert _ssh_common._bytes_to_gb(0) is None
    assert _ssh_common._bytes_to_gb(2 * 1024**3) == pytest.approx(2.0)
