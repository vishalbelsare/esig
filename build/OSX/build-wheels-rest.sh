#!/bin/bash
# Top-level script for building esig for MacOS.
# Python wheels will be created for Python versions specified in python_versions.txt.

rm -rf ~/.pyenv/versions # for reproducibility

# esig needs `boost`. Packages `pyenv` and `pyenv-virtualenv` are needed for the script below
# to build for multiple Python versions. Python 3.7 requires `openssl`.
# TODO: avoid Homebrew checking for updates
brew install boost
brew install pyenv
brew install pyenv-virtualenv
brew install openssl

# Python versions.
source install_all_python_versions.sh

# Build the esig wheels.
for p in $(cat python_versions.txt); do
   . mac_wheel_builder.sh $p
   if [ $? -eq 0 ]
   then
      echo "Successfully created wheel"
   else
      echo "Failed to create wheel"
      exit 1
   fi
done