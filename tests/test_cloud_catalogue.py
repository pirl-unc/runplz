"""Provider catalogues and the production resolver must agree.

These tests are generated from the same capability records the drivers use.
That keeps the useful invariant central: every selected machine exists in the
catalogue and satisfies all requested resources. Adding a shape expands the
test surface automatically rather than requiring another hand-written case.
"""

import re
from types import SimpleNamespace

import botocore.session
import pytest

from runplz.backends import aws, gcp
from runplz.backends.provisioning import (
    AWS_CPU_SHAPES,
    AWS_GPUS,
    GCP_CPU_SHAPES,
    GCP_GPUS,
    CloudCliError,
)


def _ec2_enum(shape_name: str) -> set:
    model = botocore.session.get_session().get_service_model("ec2")
    return set(model.shape_for(shape_name).enum)


@pytest.fixture(scope="module")
def real_instance_types() -> set:
    types = _ec2_enum("InstanceType")
    # Guard the guard: a missing botocore enum must fail, not vacuously pass.
    assert len(types) > 500, f"implausibly small instance-type catalogue: {len(types)}"
    return types


def _fn(*, gpu=None, min_gpus=None, min_cpu=None, min_memory=None):
    return SimpleNamespace(
        gpu=gpu,
        min_gpus=min_gpus,
        min_cpu=min_cpu,
        min_memory=min_memory,
        min_gpu_memory=None,
    )


AWS_CFG = SimpleNamespace(instance_type=None)
GCP_CFG = SimpleNamespace(machine_type=None, accelerator=None)


def _offering_for_name(offerings, name, gpus=0):
    matches = [item for item in offerings if item.name == name and item.gpus == gpus]
    assert len(matches) == 1, f"catalogue does not uniquely describe {name} with {gpus} GPUs"
    return matches[0]


def test_every_aws_instance_type_is_real(real_instance_types):
    expected = {item.name for item in AWS_CPU_SHAPES}
    expected.update(item.name for entry in AWS_GPUS.values() for item in entry.offerings)
    missing = expected - real_instance_types
    assert not missing, f"runplz would emit nonexistent EC2 types: {sorted(missing)}"


def test_every_aws_family_prefix_is_real(real_instance_types):
    """The family is also used to recognize explicitly pinned GPU machines."""
    families = {item.split(".", 1)[0] for item in real_instance_types}
    missing = {entry.family for entry in AWS_GPUS.values()} - families
    assert not missing


@pytest.mark.parametrize("offering", AWS_CPU_SHAPES, ids=lambda item: item.name)
def test_aws_cpu_resolver_never_underprovisions(offering):
    picked = aws.resolve_instance_type(
        AWS_CFG,
        _fn(min_cpu=offering.vcpus, min_memory=offering.memory_gb),
    )
    selected = _offering_for_name(AWS_CPU_SHAPES, picked)
    assert selected.satisfies(
        min_cpu=offering.vcpus,
        min_memory=offering.memory_gb,
        gpus=0,
    )


AWS_GPU_OFFERINGS = tuple(
    (label, offering) for label, entry in AWS_GPUS.items() for offering in entry.offerings
)


@pytest.mark.parametrize(
    "label,offering",
    AWS_GPU_OFFERINGS,
    ids=lambda value: value.name if hasattr(value, "name") else value,
)
def test_aws_gpu_resolver_never_underprovisions(label, offering):
    picked = aws.resolve_instance_type(
        AWS_CFG,
        _fn(
            gpu=label,
            min_gpus=offering.gpus,
            min_cpu=offering.vcpus,
            min_memory=offering.memory_gb,
        ),
    )
    selected = _offering_for_name(AWS_GPUS[label].offerings, picked, offering.gpus)
    assert selected.satisfies(
        min_cpu=offering.vcpus,
        min_memory=offering.memory_gb,
        gpus=offering.gpus,
    )


def test_aws_cpu_resolver_rejects_more_than_the_catalogue_can_satisfy():
    largest = max(AWS_CPU_SHAPES, key=lambda item: item.vcpus)
    with pytest.raises(CloudCliError, match="no known CPU-only machine"):
        aws.resolve_instance_type(AWS_CFG, _fn(min_cpu=largest.vcpus + 1))


@pytest.mark.parametrize("label", sorted(AWS_GPUS))
def test_aws_gpu_resolver_rejects_unknown_counts_and_oversized_minima(label):
    entry = AWS_GPUS[label]
    missing_count = next(
        count for count in range(1, max(entry.gpu_counts) + 2) if count not in entry.gpu_counts
    )
    with pytest.raises(CloudCliError, match="GPU counts"):
        aws.resolve_instance_type(AWS_CFG, _fn(gpu=label, min_gpus=missing_count))

    count = entry.gpu_counts[0]
    largest = max(
        (item for item in entry.offerings if item.gpus == count),
        key=lambda item: item.vcpus,
    )
    with pytest.raises(CloudCliError, match="no known"):
        aws.resolve_instance_type(
            AWS_CFG,
            _fn(gpu=label, min_gpus=count, min_cpu=largest.vcpus + 1),
        )


# GCE has no bundled offline machine-type enum. Validate the registry's
# structure, then drive every record through the real resolver.
GCE_MACHINE_TYPE_RE = re.compile(r"[a-z]\d*[a-z]*-[a-z]+-\d+[a-z]?")
GCE_ACCELERATOR_RE = re.compile(r"nvidia-[a-z0-9-]+")


def test_gcp_catalogue_is_well_formed():
    for label, entry in GCP_GPUS.items():
        assert GCE_ACCELERATOR_RE.fullmatch(entry.accelerator), label
        assert entry.offerings, label
        for offering in entry.offerings:
            assert GCE_MACHINE_TYPE_RE.fullmatch(offering.name), (label, offering)
            assert offering.vcpus > 0 and offering.memory_gb > 0 and offering.gpus > 0
            suffix = offering.name.rsplit("-", 1)[-1]
            if suffix.endswith("g") and suffix[:-1].isdigit():
                assert int(suffix[:-1]) == offering.gpus, (label, offering)


@pytest.mark.parametrize("offering", GCP_CPU_SHAPES, ids=lambda item: item.name)
def test_gcp_cpu_resolver_never_underprovisions(offering):
    picked, accelerator = gcp.resolve_shape(
        GCP_CFG,
        _fn(min_cpu=offering.vcpus, min_memory=offering.memory_gb),
    )
    selected = _offering_for_name(GCP_CPU_SHAPES, picked)
    assert accelerator is None
    assert selected.satisfies(
        min_cpu=offering.vcpus,
        min_memory=offering.memory_gb,
        gpus=0,
    )


GCP_GPU_OFFERINGS = tuple(
    (label, offering) for label, entry in GCP_GPUS.items() for offering in entry.offerings
)


@pytest.mark.parametrize(
    "label,offering",
    GCP_GPU_OFFERINGS,
    ids=lambda value: value.name if hasattr(value, "name") else value,
)
def test_gcp_gpu_resolver_never_underprovisions(label, offering):
    picked, accelerator = gcp.resolve_shape(
        GCP_CFG,
        _fn(
            gpu=label,
            min_gpus=offering.gpus,
            min_cpu=offering.vcpus,
            min_memory=offering.memory_gb,
        ),
    )
    selected = _offering_for_name(GCP_GPUS[label].offerings, picked, offering.gpus)
    assert selected.satisfies(
        min_cpu=offering.vcpus,
        min_memory=offering.memory_gb,
        gpus=offering.gpus,
    )
    if GCP_GPUS[label].attached:
        assert accelerator == f"type={GCP_GPUS[label].accelerator},count={offering.gpus}"
    else:
        assert accelerator is None


def test_gcp_cpu_resolver_rejects_more_than_the_catalogue_can_satisfy():
    largest = max(GCP_CPU_SHAPES, key=lambda item: item.memory_gb)
    with pytest.raises(CloudCliError, match="no known CPU-only machine"):
        gcp.resolve_shape(GCP_CFG, _fn(min_memory=largest.memory_gb + 1))


@pytest.mark.parametrize("label", sorted(GCP_GPUS))
def test_gcp_gpu_resolver_rejects_unknown_counts_and_oversized_minima(label):
    entry = GCP_GPUS[label]
    missing_count = next(
        count for count in range(1, max(entry.gpu_counts) + 2) if count not in entry.gpu_counts
    )
    with pytest.raises(CloudCliError, match="GPU counts"):
        gcp.resolve_shape(GCP_CFG, _fn(gpu=label, min_gpus=missing_count))

    count = entry.gpu_counts[0]
    largest = max(
        (item for item in entry.offerings if item.gpus == count),
        key=lambda item: item.memory_gb,
    )
    with pytest.raises(CloudCliError, match="no known"):
        gcp.resolve_shape(
            GCP_CFG,
            _fn(gpu=label, min_gpus=count, min_memory=largest.memory_gb + 1),
        )
