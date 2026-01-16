#!/bin/bash
# Script to prepare the build environment for PyCUDA.
#
# Example usage:
#   ./prepare_for_build.sh v2026.1

set -euxo pipefail

export ROOT=`pwd`

if [ $# -ne 1 ]; then
    echo "Usage: $0 <pycuda_version>"
    echo "Example: $0 v2026.1"
    exit 1
fi

PYCUDA_VERSION=$1

# Configure PyCUDA with CUDA paths.
python configure.py --cuda-root="${CUDA_ROOT}"

# Apply patches if they exist for this version.
if [ -d "${ROOT}/build_scripts/patches/${PYCUDA_VERSION}" ]; then
    for patch in "${ROOT}/build_scripts/patches/${PYCUDA_VERSION}"/*.patch; do
        if [ -f "$patch" ]; then
            patch -p1 -d "${ROOT}" -i "${patch}"
        fi
    done
fi
