"""Targeted tests for small production branches absent from the main matrix."""

import warnings

from runplz import App, Image


def test_legacy_repo_root_alias_getter_and_setter():
    app = App("coverage")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        assert app._repo_root is None
        app._repo_root = "."
        assert app.repo_root is not None


def test_none_environment_normalizes_to_empty_mapping():
    app = App("coverage")

    @app.function(image=Image.from_registry("python:3.12-slim"), env=None)
    def job():
        return None

    assert job.env == {}
