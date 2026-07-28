import importlib
from importlib.metadata import version

import pytest
import torch


@pytest.fixture(scope="module")
def device() -> torch.device:
    assert torch.cuda.is_available(), "The tests must run on a CUDA GPU"
    device = torch.device("cuda")
    return device


def test_published_cuda_wheel(device: torch.device) -> None:
    assert version("pycuda") == "2026.1+cu.12.8"
    assert torch.__version__ == "2.10.0+cu128"
    assert torch.version.cuda == "12.8"
    assert torch.cuda.get_device_name(device)


@pytest.mark.parametrize("module_name", ["pycuda", "pycuda.driver"])
def test_native_module(device: torch.device, module_name: str) -> None:
    assert importlib.import_module(module_name) is not None


def test_cuda_driver_device(device: torch.device) -> None:
    import pycuda.driver as cuda

    cuda.init()
    assert cuda.Device.count() > 0
    assert cuda.Device(0).compute_capability()[0] >= 8


def test_cuda_device_memory_round_trip(device: torch.device) -> None:
    import numpy as np
    import pycuda.driver as cuda

    cuda.init()
    context = cuda.Device(0).make_context()
    try:
        expected = np.arange(32, dtype=np.float32)
        actual = np.empty_like(expected)
        allocation = cuda.mem_alloc(expected.nbytes)
        cuda.memcpy_htod(allocation, expected)
        cuda.memcpy_dtoh(actual, allocation)
        np.testing.assert_array_equal(actual, expected)
    finally:
        context.pop()
