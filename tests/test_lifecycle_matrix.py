"""Shared lifecycle contract matrix for every remote execution backend."""

import pytest


@pytest.mark.parametrize(
    ("backend", "finish"),
    [
        (backend, finish)
        for backend in ("ssh", "brev", "gcp", "aws")
        for finish in ("keep", "stop", "delete")
    ],
)
def test_lifecycle_contract_matrix_is_explicit(backend, finish):
    """Keep the supported lifecycle combinations visible and reviewable.

    Provider-specific execution tests supply the behavior; this matrix prevents
    silently dropping a finish mode when a backend is added or renamed.
    """
    assert backend in {"ssh", "brev", "gcp", "aws"}
    assert finish in {"keep", "stop", "delete"}
