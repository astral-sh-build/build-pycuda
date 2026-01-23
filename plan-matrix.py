# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "packaging",
# ]
# ///

import json
import os

from packaging.version import Version

# Supported Python versions for pycuda.
PYTHON_VERSIONS = ["3.9", "3.10", "3.11", "3.12", "3.13", "3.14"]

# CUDA versions to build against.
CUDA_VERSIONS = ["12.4", "12.6", "12.8", "12.9", "13.0"]

# Architecture to CUDA version mapping.
ARCH_CUDA_PAIRS = {
    "x86_64": ["12.4", "12.6", "12.8", "12.9", "13.0"],
    "aarch64": ["12.6", "12.8", "12.9", "13.0"],
}

# The glibc version to use for manylinux builds.
GLIBC_VERSION = "2_28"

AUDITWHEEL_BLANKET_EXCLUDES = [
    "libcuda.so",
    "libcuda.so.1",
]

AUDITWHEEL_CUDA_VERSION_EXCLUDES = {
    "12": [
        "libcudart.so.12",
        "libcudart.so.12.0",
    ],
    "13": [
        "libcudart.so.13",
        "libcudart.so.13.0",
    ],
}


def main() -> None:
    rows = []
    for target_arch, cuda_versions in ARCH_CUDA_PAIRS.items():
        for cuda_version in cuda_versions:
            cuda_version_parsed = Version(cuda_version)
            for python_version in PYTHON_VERSIONS:
                row = {
                    "target-arch": target_arch,
                    "python-version": python_version,
                    "cuda-version": cuda_version,
                }
                rows.append(row)

    # Transform each row to add various nice-to-have representations of fields.
    for row in rows:
        # `CI_*` variables: same as the original ones.
        row["CI_CUDA_VERSION"] = row["cuda-version"]
        row["CI_PYTHON_VERSION"] = row["python-version"]

        # `MATRIX_CUDA_VERSION`: XY instead of X.Y
        cuda_version = Version(row["cuda-version"])
        row["MATRIX_CUDA_VERSION"] = f"{cuda_version.major}{cuda_version.minor}"

        # `MATRIX_PYTHON_VERSION`: same as `python-version`, but with the dot removed
        row["MATRIX_PYTHON_VERSION"] = row["python-version"].replace(".", "")

        # `MANYLINUX_CUDA_VERSION`: X.Y instead of X.Y.Z
        row["MANYLINUX_CUDA_VERSION"] = f"{cuda_version.major}.{cuda_version.minor}"

        # `MANYLINUX_CUDA_COMPAT_VERSION`: X-Y instead of X.Y.Z
        row["MANYLINUX_CUDA_COMPAT_VERSION"] = (
            f"{cuda_version.major}-{cuda_version.minor}"
        )

        # MANYLINUX_GLIBC_VERSION: the glibc version to use for manylinux builds.
        row["MANYLINUX_GLIBC_VERSION"] = GLIBC_VERSION

        # `CI_AUDITWHEEL_EXCLUDES`: `--exclude {lib}` for each lib that should
        # be excluded when running `auditwheel repair`.
        cuda_major = str(cuda_version.major)
        auditwheel_excludes = (
            AUDITWHEEL_BLANKET_EXCLUDES
            + AUDITWHEEL_CUDA_VERSION_EXCLUDES.get(cuda_major, [])
        )
        row["CI_AUDITWHEEL_EXCLUDES"] = " ".join(
            f"--exclude {lib}" for lib in auditwheel_excludes
        )

        # RUNNER: the GitHub Actions runner to use.
        if row["target-arch"] == "x86_64":
            row["RUNNER"] = "depot-ubuntu-24.04-64"
        elif row["target-arch"] == "aarch64":
            row["RUNNER"] = "depot-ubuntu-24.04-arm-64"
        else:
            raise ValueError(f"Unknown target arch: {row['target-arch']}")

    # For PR builds, limit matrix to a single entry for faster CI.
    if os.environ.get("LIMIT_MATRIX") == "1":
        rows = rows[:1]
    print(json.dumps(rows))


if __name__ == "__main__":
    main()
