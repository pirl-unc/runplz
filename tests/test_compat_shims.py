"""Exercise the legacy module paths that remain part of the wire contract."""

import runplz._bootstrap as legacy_bootstrap
import runplz._cli as legacy_cli


def test_legacy_shims_forward_attributes_and_directory():
    assert legacy_bootstrap.main is not None
    assert "main" in legacy_bootstrap.__dir__()
    assert legacy_cli.main is not None
    assert "main" in legacy_cli.__dir__()
