"""The cloud selectors must never under-provision a declared minimum.

Issue #95: AWS/GCP selection could clamp a request past the largest known
shape down to a smaller one, and GPU selection could ignore `min_cpu` /
`min_memory` entirely. `select_machine` fixed that — its docstring calls it
"the fail-instead-of-clamp contract" — but nothing in the suite pinned it.

`select_machine` reaches 98% line coverage without a single test calling it:
everything goes through `gcp.resolve_shape` and `aws.resolve_instance_type`
with hand-picked examples. Coverage is not the property. A regression to
clamping returns a *finite smaller shape* rather than an error, so every
existing example would keep passing — none of them probes past the largest
offering. These tests assert the property over the whole catalogue instead.
"""

from unittest import mock

import pytest

from runplz import App, AwsConfig, GcpConfig, Image
from runplz.backends import aws, gcp
from runplz.backends import provisioning as p


def _function(**kwargs):
    app = App("demo")

    @app.function(image=Image.from_registry("ubuntu:22.04"), **kwargs)
    def train():  # pragma: no cover - never executed
        pass

    return app.functions["train"]


def _catalogue():
    """Every (cloud, offerings, gpu_count, label) runplz can select from."""
    for cloud, cpu_shapes, gpu_table in (
        ("aws", p.AWS_CPU_SHAPES, p.AWS_GPUS),
        ("gcp", p.GCP_CPU_SHAPES, p.GCP_GPUS),
    ):
        yield cloud, cpu_shapes, 0, None
        for label, shapes in gpu_table.items():
            for count in shapes.gpu_counts:
                yield cloud, shapes.offerings, count, label


# Spans below and far above the largest real shape, so the sweep covers both
# "satisfiable" and "must raise" on every catalogue.
_MINIMUMS = (0, 1, 2, 4, 8, 16, 32, 64, 100, 200, 1000)


def test_a_selection_never_returns_less_than_was_asked_for():
    """The property, over the entire catalogue: a returned offering satisfies
    every declared minimum, or nothing is returned at all."""
    selected = 0
    for cloud, offerings, count, label in _catalogue():
        for min_cpu in _MINIMUMS:
            for min_memory in _MINIMUMS:
                function = _function(
                    min_cpu=min_cpu or None,
                    min_memory=min_memory or None,
                    gpu=label,
                    num_gpus=max(count, 1),
                )
                try:
                    got = p.select_machine(
                        function, offerings, cloud=cloud, gpus=count, gpu_label=label
                    )
                except p.CloudCliError:
                    continue  # refusing is always allowed; under-provisioning is not
                selected += 1
                assert got.vcpus >= min_cpu, (cloud, label, count, min_cpu, got)
                assert got.memory_gb >= min_memory, (cloud, label, count, min_memory, got)
                assert got.gpus == count, (cloud, label, count, got)
    assert selected > 500, f"the sweep only made {selected} selections; catalogue lookup broke"


def test_the_sweep_actually_exercises_the_refusal_path():
    """Guards the test above. If every combination were satisfiable the
    assertions would hold vacuously and a clamp would sail through."""
    refused = 0
    for cloud, offerings, count, label in _catalogue():
        function = _function(min_cpu=100000, gpu=label, num_gpus=max(count, 1))
        with pytest.raises(p.CloudCliError):
            p.select_machine(function, offerings, cloud=cloud, gpus=count, gpu_label=label)
        refused += 1
    assert refused > 20


@pytest.mark.parametrize(
    "cloud, offerings, kwargs",
    [
        ("aws", p.AWS_CPU_SHAPES, {"min_cpu": 200}),
        ("aws", p.AWS_CPU_SHAPES, {"min_memory": 2048}),
        ("gcp", p.GCP_CPU_SHAPES, {"min_cpu": 200}),
        ("gcp", p.GCP_CPU_SHAPES, {"min_memory": 2048}),
    ],
)
def test_the_examples_from_the_issue_refuse_rather_than_clamp(cloud, offerings, kwargs):
    """#95's literal examples, kept by name so the report stays checkable."""
    with pytest.raises(p.CloudCliError, match="no known"):
        p.select_machine(_function(**kwargs), offerings, cloud=cloud, gpus=0)


def test_a_gpu_shape_still_has_to_satisfy_the_cpu_minimum():
    """#95's other example: `gpu=T4` with `min_cpu=100` returned g4dn.xlarge,
    a 4-vCPU box. GPU count is not the only constraint on a GPU shape."""
    function = _function(gpu="T4", min_cpu=100)
    with pytest.raises(p.CloudCliError, match="no known"):
        p.select_machine(function, p.AWS_GPUS["T4"].offerings, cloud="aws", gpus=1, gpu_label="T4")


@pytest.mark.parametrize(
    "module, config_kwargs",
    [
        (aws, {"aws_config": AwsConfig(region="us-east-1", key_name="k")}),
        (gcp, {"gcp_config": GcpConfig(project="p", zone="z")}),
    ],
)
def test_an_unsatisfiable_minimum_raises_before_any_billed_cli_runs(module, config_kwargs):
    """The other half of #95's requirement. The cost of regressing this is a
    provisioned box, not a failed assertion — so it is asserted by counting
    subprocess calls, not by trusting the ordering of two function calls."""
    app = App("demo", **config_kwargs)

    @app.function(image=Image.from_registry("ubuntu:22.04"), min_cpu=200)
    def train():  # pragma: no cover
        pass

    with mock.patch.object(module.subprocess, "run") as run:
        with pytest.raises(p.CloudCliError):
            module.run(app, app.functions["train"], [], {})
    run.assert_not_called()
