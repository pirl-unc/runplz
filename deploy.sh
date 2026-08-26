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

rm -rf dist
"$BUILD_PY" -m build
"$BUILD_PY" -m twine upload dist/*

VERSION="$(python3 -c 'from runplz.version import __version__; print(__version__)')"
TAG="v${VERSION}"
git tag -a "$TAG" -m "Release ${TAG}"
git push origin "refs/tags/${TAG}"
