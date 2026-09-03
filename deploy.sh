#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# Build/upload tooling lives in a throwaway venv owned by this script.
#
# Issue #75: `python3 -m pip install --upgrade build twine` against a
# Homebrew (or any PEP 668 "externally managed") interpreter dies with
# `externally-managed-environment` — and it used to do so *after* the
# full test suite had run, so you paid several minutes of lint+tests to
# learn that you'd forgotten to activate a virtualenv. Provisioning the
# venv up front both isolates the install and fails fast.
BUILD_VENV="$ROOT/.deploy-venv"
python3 -m venv --clear "$BUILD_VENV"
BUILD_PY="$BUILD_VENV/bin/python"
"$BUILD_PY" -m pip install --quiet --upgrade pip build twine

./lint.sh
./test.sh

# Issue #136: `build/` is cleared for the same reason `dist/` always was — a
# release must not depend on what a previous one left behind. setuptools
# copies sources into `build/lib` without pruning, so a module deleted from
# the tree lingers there indefinitely; this repo's had carried three modules
# renamed away in 3.20.0 for five minor versions.
#
# That staleness is inert today: `python -m build` with no arguments makes the
# sdist from the source tree, extracts it to a temp dir, and builds the wheel
# from *that*, so `build/lib` is never read. It stops being inert the moment
# the build path changes — a `--wheel` flag skipping the sdist round trip, or
# a setuptools that reuses the directory — and by then the evidence is in a
# published artifact.
#
# `runplz.egg-info/` is deliberately left alone: a legacy `pip install -e`
# (what develop.sh does) resolves through it, so clearing it here would break
# the working tree of whoever is cutting the release.
rm -rf build dist
"$BUILD_PY" -m build
"$BUILD_PY" -m twine upload dist/*

VERSION="$(python3 -c 'from runplz.version import __version__; print(__version__)')"
TAG="v${VERSION}"
git tag -a "$TAG" -m "Release ${TAG}"
git push origin "refs/tags/${TAG}"
