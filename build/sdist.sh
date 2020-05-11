#!/bin/bash -e
set -u -o xtrace

# Build source distribution. Run from build directory.

pushd .. # need to run in same directory as setup.py
echo "Python version installed (>= 3.5 required):"
python --version
pip install --user cython # not present in GitHub macos-10.15 environment
python setup.py sdist --dist-dir=build/sdist
rm -rf esig.egg-info
popd