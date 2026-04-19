#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

./lint.sh
./test.sh

python3 -m pip install --upgrade build twine
rm -rf dist
python3 -m build
python3 -m twine upload dist/*

VERSION="$(python3 -c 'from runplz.version import __version__; print(__version__)')"
TAG="v${VERSION}"
git tag -a "$TAG" -m "Release ${TAG}"
git push origin "refs/tags/${TAG}"
