#!/bin/bash -e
# Top-level script for building esig for Linux on MacOS or Linux.
# $1: one of {i686, x86_64}
# Python wheels will be created for specified architecture, for each Python version in python_versions.txt.
if [[ $1 == "i686" ]] || [[ $1 == "x86_64" ]]
then
   arch=$1
else
   echo "Invalid architecture"
   exit 1
fi

docker build -t esig_builder_linux_${arch} -f Dockerfile_${arch}.dockerfile .

# Get esig sources.
source ../sdist.sh

# Wheels that pass their tests will be placed here.
rm -rf wheelhouse

# Build the esig wheels inside the docker container.
docker run --rm -v ${PWD}:/data esig_builder_linux_${arch} \
   "source ~/.bashrc; cd /data; source linux_wheel_builder.sh $arch" || exit 1
