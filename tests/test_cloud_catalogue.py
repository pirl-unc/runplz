"""Are the machine shapes runplz emits real?

Every other cloud test asserts that runplz passes the argv its author
intended. None of them can tell you whether the *values* in that argv
exist, because a mock accepts anything and so does a stub. That is not
hypothetical: `p3.xlarge` shipped in this repo's shape table and does not
exist (the smallest p3 is `p3.2xlarge`), and it would have failed at
`run-instances` time — after the retry budget, on a real account.

`botocore` bundles the EC2 instance-type catalogue as a static enum in its
service model, so this check needs no credentials, no account, no network
and no server. It is a dev-only dependency; runplz stays stdlib-only.

Note what moto does *not* give you here: driving `moto_server` with the
real `aws` CLI, `--instance-type not-a-real-type` is accepted. An API
emulator models the protocol, not the catalogue.
"""

import re

import pytest

from runplz.backends.aws import _cpu_size_name
from runplz.backends.provisioning import AWS_GPUS, GCP_GPUS

botocore = pytest.importorskip("botocore.session", reason="botocore is a dev dependency")


def _ec2_enum(shape_name: str) -> set:
    model = botocore.get_session().get_service_model("ec2")
    return set(model.shape_for(shape_name).enum)


@pytest.fixture(scope="module")
def real_instance_types() -> set:
    types = _ec2_enum("InstanceType")
    # Guard the guard: if botocore ever stops shipping the enum, this test
    # must fail loudly rather than pass against an empty set.
    assert len(types) > 500, f"implausibly small instance-type catalogue: {len(types)}"
    return types


@pytest.mark.parametrize(
    "label, count, shape",
    [
        (label, count, shape)
        for label, entry in AWS_GPUS.items()
        for count, shape in (entry.shapes or {}).items()
    ],
)
def test_every_aws_gpu_shape_is_a_real_instance_type(label, count, shape, real_instance_types):
    assert shape in real_instance_types, (
        f"AWS_GPUS[{label!r}] maps {count} GPU(s) to {shape!r}, which is not an EC2 "
        f"instance type. This fails at run-instances on a real account."
    )


@pytest.mark.parametrize("label", sorted(AWS_GPUS))
def test_every_aws_family_prefix_is_real(label, real_instance_types):
    """The family is used to answer 'does this instance type have a GPU?'."""
    family = AWS_GPUS[label].family
    assert any(t.split(".", 1)[0] == family for t in real_instance_types), (
        f"AWS_GPUS[{label!r}].family = {family!r} matches no real instance type"
    )


class _Fn:
    """Minimal stand-in for a Function, which is all `_cpu_size_name` reads."""

    def __init__(self, min_cpu=None, min_memory=None):
        self.min_cpu = min_cpu
        self.min_memory = min_memory


@pytest.mark.parametrize(
    "fn",
    [_Fn()]
    + [_Fn(min_cpu=c) for c in (1, 2, 3, 4, 8, 16, 32, 48, 64, 96, 200)]
    + [_Fn(min_memory=m) for m in (1, 8, 16, 64, 128, 256, 512, 2048)],
    ids=lambda f: f"cpu={f.min_cpu},mem={f.min_memory}",
)
def test_every_cpu_only_instance_type_runplz_can_pick_is_real(fn, real_instance_types):
    """The CPU path builds `m6i.<size>` by string concatenation.

    A size name that does not exist in the m6i family produces a plausible
    string that EC2 rejects, so drive the sizer across its whole range
    rather than trusting the table it reads.
    """
    picked = f"m6i.{_cpu_size_name(fn)}"
    assert picked in real_instance_types, (
        f"min_cpu={fn.min_cpu} min_memory={fn.min_memory} picks {picked!r}, "
        f"which is not an EC2 instance type"
    )


def test_aws_gpu_counts_are_plausible_for_the_family():
    """Structural check on the count -> shape mapping.

    botocore's model carries the instance-type *names* but not their GPU
    counts (that needs `describe-instance-types`, which needs an account),
    so this cannot verify that `g5.12xlarge` really has 4 GPUs. It can
    verify the table is internally consistent: counts ascending, shapes
    distinct, and every shape inside its declared family.
    """
    for label, entry in AWS_GPUS.items():
        shapes = entry.shapes or {}
        counts = list(shapes)
        assert counts == sorted(counts), f"{label}: GPU counts are not ascending"
        assert len(set(shapes.values())) == len(shapes), f"{label}: duplicate shapes"
        for count, shape in shapes.items():
            assert shape.split(".", 1)[0] == entry.family, (
                f"{label}: {shape!r} is not in the declared family {entry.family!r}"
            )


# ---------------------------------------------------------------------------
# GCP has no offline equivalent, so this is what can honestly be checked.

# `family-series-size`, e.g. n2-standard-8, a2-highgpu-1g, g2-standard-24.
GCE_MACHINE_TYPE_RE = re.compile(r"[a-z]\d*[a-z]*-[a-z]+-\d+[a-z]?")
# GCE accelerator names, e.g. nvidia-tesla-t4, nvidia-l4, nvidia-h100-80gb.
GCE_ACCELERATOR_RE = re.compile(r"nvidia-[a-z0-9-]+")


def test_gcp_shapes_are_well_formed():
    """GCE machine types cannot be validated against a catalogue.

    `gcloud emulators` ships only firestore and spanner, and
    `google-cloud-compute` models MachineType as a *resource message*, not
    an enum -- machine types are per-zone API resources with no bundled
    list. So unlike AWS, a wrong-but-plausible GCE machine type cannot be
    caught offline. This checks the format only, and that is the honest
    limit of it.
    """
    for label, entry in GCP_GPUS.items():
        assert GCE_ACCELERATOR_RE.fullmatch(entry.accelerator), (
            f"GCP_GPUS[{label!r}].accelerator = {entry.accelerator!r} is not a GCE accelerator name"
        )
        for count, shape in (entry.shapes or {}).items():
            assert GCE_MACHINE_TYPE_RE.fullmatch(shape), (
                f"GCP_GPUS[{label!r}][{count}] = {shape!r} is not a well-formed GCE machine type"
            )


def test_gcp_bundled_gpu_shapes_encode_their_count():
    """a2/a3/g2 machine types carry the GPU count in the name.

    `a2-highgpu-4g` means four GPUs, so the table key and the name must
    agree -- a mismatch here silently provisions the wrong-sized box.
    """
    for label, entry in GCP_GPUS.items():
        for count, shape in (entry.shapes or {}).items():
            suffix = shape.rsplit("-", 1)[-1]
            if suffix.endswith("g") and suffix[:-1].isdigit():
                assert int(suffix[:-1]) == count, (
                    f"GCP_GPUS[{label!r}] maps {count} GPU(s) to {shape!r}, "
                    f"whose name says {suffix[:-1]}"
                )
