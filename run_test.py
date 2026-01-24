"""
Modal test script for PyCUDA wheel builds.

This script tests PyCUDA wheels from a GitHub release on GPU-enabled machines.
It downloads the wheel from the release assets and runs a basic GPU test.

Run with:
    GITHUB_TOKEN=<token> uv run --with modal modal run run_test.py

Or to test a specific Python/CUDA version:
    GITHUB_TOKEN=<token> PYTHON_VERSION=3.12 CUDA_VERSION=12.6 uv run --with modal modal run run_test.py
"""

import os
import sys

import modal

# GitHub release information
GITHUB_OWNER = "astral-sh"
GITHUB_REPO = "build-pycuda"
RELEASE_TAG = os.environ.get("RELEASE_TAG", "v2026.1")
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")

if not GITHUB_TOKEN:
    raise ValueError("GITHUB_TOKEN environment variable must be set")

# Test configuration
# For Modal, we can only test x86_64 wheels (no ARM GPU support)
# CUDA versions available in Modal: 12.4, 12.6, 12.8 (13.0 not yet available in drivers)
CUDA_VERSIONS_TO_TEST = os.environ.get("CUDA_VERSION", "12.4,12.6,12.8").split(",")
PYTHON_VERSION = os.environ.get("PYTHON_VERSION", "3.12")

print(f"Setting up tests for PyCUDA {RELEASE_TAG}")
print(f"  Python {PYTHON_VERSION}, testing with CUDA versions: {CUDA_VERSIONS_TO_TEST}")

# Create a single app
app = modal.App("test-pycuda")


def download_wheel_func():
    """Download the PyCUDA wheel from GitHub release."""
    import os
    import requests

    GITHUB_TOKEN = os.environ["GITHUB_TOKEN"]
    GITHUB_OWNER = os.environ["GITHUB_OWNER"]
    GITHUB_REPO = os.environ["GITHUB_REPO"]
    RELEASE_TAG = os.environ["RELEASE_TAG"]
    WHEEL_NAME = os.environ["WHEEL_NAME"]

    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/octet-stream",
    }

    # Get release assets
    release_url = f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}/releases/tags/{RELEASE_TAG}"
    response = requests.get(
        release_url,
        headers={"Authorization": f"token {GITHUB_TOKEN}", "Accept": "application/vnd.github.v3+json"},
    )
    response.raise_for_status()

    release_data = response.json()
    assets = release_data.get("assets", [])

    # Find matching asset
    asset = None
    for a in assets:
        if a["name"] == WHEEL_NAME:
            asset = a
            break

    if not asset:
        raise ValueError(f"Could not find asset {WHEEL_NAME} in release {RELEASE_TAG}")

    # Download the wheel
    download_url = asset["url"]
    response = requests.get(download_url, headers=headers, stream=True)
    response.raise_for_status()

    wheel_path = f"/{WHEEL_NAME}"
    with open(wheel_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    print(f"Downloaded: {WHEEL_NAME}")


def get_wheel_name(cuda_version: str, python_version: str) -> str:
    """Generate the wheel filename for the given CUDA and Python version."""
    py_ver = python_version.replace(".", "")
    # Use the correct manylinux tag based on CUDA version
    if cuda_version in ("12.4", "12.6"):
        manylinux_tag = "manylinux_2_24_x86_64.manylinux_2_28_x86_64"
    else:
        manylinux_tag = "manylinux_2_27_x86_64.manylinux_2_28_x86_64"
    return f"pycuda-2026.1+cu{cuda_version}-cp{py_ver}-cp{py_ver}-{manylinux_tag}.whl"


def get_cuda_image_tag(cuda_version: str) -> str:
    """Get the NVIDIA CUDA image tag for a given CUDA version."""
    # Map CUDA versions to available NVIDIA images
    cuda_image_map = {
        "12.4": "12.4.1-devel-ubuntu22.04",
        "12.6": "12.6.3-devel-ubuntu22.04",
        "12.8": "12.8.0-devel-ubuntu22.04",
    }
    return cuda_image_map.get(cuda_version, f"{cuda_version}.0-devel-ubuntu22.04")


# Create images for each CUDA version
images = {}
for cuda_version in CUDA_VERSIONS_TO_TEST:
    wheel_name = get_wheel_name(cuda_version, PYTHON_VERSION)
    cuda_image_tag = get_cuda_image_tag(cuda_version)
    images[cuda_version] = (
        # Use NVIDIA CUDA image which includes nvcc
        modal.Image.from_registry(f"nvidia/cuda:{cuda_image_tag}")
        # Install uv for fast Python management
        .run_commands(
            "apt-get update && apt-get install -y curl ca-certificates",
            "curl -LsSf https://astral.sh/uv/install.sh | sh",
            # Create venv with uv and install packages
            f"/root/.local/bin/uv venv /venv --python {PYTHON_VERSION}",
            "/root/.local/bin/uv pip install --python /venv/bin/python requests numpy",
        )
        .env({
            "PATH": "/venv/bin:/root/.local/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            "VIRTUAL_ENV": "/venv",
            "GITHUB_TOKEN": GITHUB_TOKEN,
            "GITHUB_OWNER": GITHUB_OWNER,
            "GITHUB_REPO": GITHUB_REPO,
            "RELEASE_TAG": RELEASE_TAG,
            "WHEEL_NAME": wheel_name,
        })
        .run_function(download_wheel_func)
        .run_commands(f"/root/.local/bin/uv pip install --python /venv/bin/python /{wheel_name}")
    )


def _run_pycuda_test(cuda_version: str):
    """Core test logic for PyCUDA - parameterized by CUDA version."""
    import subprocess

    import numpy as np

    print("=" * 80)
    print(f"Testing PyCUDA with CUDA {cuda_version}")
    print("=" * 80)
    print()

    print("=" * 80)
    print("GPU Information")
    print("=" * 80)
    subprocess.run(["nvidia-smi"], check=True)
    print()

    print("=" * 80)
    print("Python and Package Versions")
    print("=" * 80)
    print(f"Python version: {sys.version}")

    import pycuda
    import pycuda.autoinit
    import pycuda.driver as cuda

    print(f"PyCUDA version: {pycuda.VERSION_TEXT}")
    print(f"CUDA driver version: {cuda.get_driver_version()}")
    print(f"Device count: {cuda.Device.count()}")

    device = cuda.Device(0)
    print(f"Device name: {device.name()}")
    print(f"Compute capability: {device.compute_capability()}")
    print(f"Total memory: {device.total_memory() // (1024**2)} MB")
    print()

    print("=" * 80)
    print("Testing PyCUDA Functionality")
    print("=" * 80)

    # Test 1: Basic memory allocation and transfer
    print("\nTest 1: Memory allocation and transfer")
    from pycuda import gpuarray

    a_cpu = np.random.randn(1000).astype(np.float32)
    a_gpu = gpuarray.to_gpu(a_cpu)
    a_back = a_gpu.get()
    assert np.allclose(a_cpu, a_back), "Memory transfer failed"
    print("✓ Memory allocation and transfer works")

    # Test 2: GPU array operations
    print("\nTest 2: GPU array operations")
    b_gpu = gpuarray.to_gpu(np.random.randn(1000).astype(np.float32))
    c_gpu = a_gpu + b_gpu
    c_cpu = c_gpu.get()
    expected = a_cpu + b_gpu.get()
    assert np.allclose(c_cpu, expected), "GPU addition failed"
    print("✓ GPU array operations work")

    # Test 3: Custom CUDA kernel
    print("\nTest 3: Custom CUDA kernel compilation and execution")
    from pycuda.compiler import SourceModule

    mod = SourceModule("""
    __global__ void multiply_by_two(float *a, int n)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx < n)
            a[idx] *= 2.0f;
    }
    """)

    multiply_by_two = mod.get_function("multiply_by_two")

    test_array = np.ones(256, dtype=np.float32)
    test_gpu = gpuarray.to_gpu(test_array)
    multiply_by_two(test_gpu, np.int32(256), block=(256, 1, 1), grid=(1, 1))
    result = test_gpu.get()
    assert np.allclose(result, 2.0), f"Kernel execution failed: expected 2.0, got {result[0]}"
    print("✓ Custom CUDA kernel compilation and execution works")

    # Test 4: cuBLAS operations via gpuarray
    print("\nTest 4: Matrix operations")
    from pycuda import cumath

    x = gpuarray.to_gpu(np.random.randn(100).astype(np.float32))
    y = cumath.exp(x)
    y_expected = np.exp(x.get())
    assert np.allclose(y.get(), y_expected, rtol=1e-5), "cumath.exp failed"
    print("✓ Math operations work")

    print()
    print("=" * 80)
    print(f"All tests passed for CUDA {cuda_version}!")
    print("=" * 80)

    return {
        "status": "success",
        "cuda_test_version": cuda_version,
        "pycuda_version": pycuda.VERSION_TEXT,
        "driver_version": cuda.get_driver_version(),
        "gpu_name": device.name(),
    }


# Static test functions for each CUDA version
@app.function(image=images.get("12.4"), gpu="a10g", timeout=600)
def test_cuda124():
    return _run_pycuda_test("12.4")


@app.function(image=images.get("12.6"), gpu="a10g", timeout=600)
def test_cuda126():
    return _run_pycuda_test("12.6")


@app.function(image=images.get("12.8"), gpu="a10g", timeout=600)
def test_cuda128():
    return _run_pycuda_test("12.8")


# Map CUDA versions to their test functions
test_functions = {
    "12.4": test_cuda124,
    "12.6": test_cuda126,
    "12.8": test_cuda128,
}


@app.local_entrypoint()
def main():
    """Main entry point - runs all tests in parallel."""
    print("=" * 80)
    print("PyCUDA Multi-CUDA Version Test Suite")
    print("=" * 80)
    print(f"Release: {RELEASE_TAG}")
    print(f"Python version: {PYTHON_VERSION}")
    print(f"Testing against CUDA versions: {CUDA_VERSIONS_TO_TEST}")
    print("=" * 80)
    print()
    print("Running all tests in parallel...")
    print()

    # Filter to only test versions we have functions for
    versions_to_run = [v for v in CUDA_VERSIONS_TO_TEST if v in test_functions]
    if not versions_to_run:
        print(f"Error: No test functions available for CUDA versions: {CUDA_VERSIONS_TO_TEST}")
        print(f"Available: {list(test_functions.keys())}")
        sys.exit(1)

    # Run all tests in parallel
    tasks = []
    for cuda_version in versions_to_run:
        tasks.append(test_functions[cuda_version].spawn())

    # Wait for all tasks to complete
    results = {}
    for i, cuda_version in enumerate(versions_to_run):
        try:
            results[cuda_version] = tasks[i].get()
        except Exception as e:
            results[cuda_version] = {"status": "failed", "error": str(e)}

    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Release: {RELEASE_TAG}")
    print(f"Python: {PYTHON_VERSION}")
    print()

    failed = []
    for cuda_version in versions_to_run:
        result = results[cuda_version]
        if isinstance(result, Exception):
            status = "✗ FAIL"
            failed.append(cuda_version)
            print(f"CUDA {cuda_version}: {status}")
            print(f"  Error: {result}")
        elif result.get("status") == "success":
            status = "✓ PASS"
            print(f"CUDA {cuda_version}: {status}")
            print(f"  GPU: {result.get('gpu_name', 'unknown')}")
            print(f"  PyCUDA: {result.get('pycuda_version', 'unknown')}")
            print(f"  Driver: {result.get('driver_version', 'unknown')}")
        else:
            status = "✗ FAIL"
            failed.append(cuda_version)
            print(f"CUDA {cuda_version}: {status}")
            print(f"  Error: {result.get('error', 'unknown error')}")

    print("=" * 80)

    if failed:
        print(f"\n❌ {len(failed)} test(s) failed: {', '.join(failed)}")
        sys.exit(1)
    else:
        print(f"\n✅ All {len(versions_to_run)} tests passed!")
