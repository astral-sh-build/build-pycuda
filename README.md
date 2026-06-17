# build-pycuda

Pre-built Linux wheels for [PyCUDA](https://github.com/inducer/pycuda), across Python, CUDA, and
CPU architectures.

## Installation

Artifacts are published to a separate index for each CUDA version. Each wheel has a local version
suffix that identifies the CUDA version it was built against, such as `pycuda==2026.1+cu12.8`.

Pre-built wheels are available on [Astral's GPU indexes](https://wheels.astralhosted.com/index.html).
For example, to install a CUDA 12.8 build:

```console
$ uv add pycuda --index astral-cu128=https://wheels.astralhosted.com/simple/cu128/
```

This configures the index and uses it as the source for `pycuda`:

```toml
[tool.uv.sources]
pycuda = { index = "astral-cu128" }

[[tool.uv.index]]
name = "astral-cu128"
url = "https://wheels.astralhosted.com/simple/cu128/"
```

Or, with `uv pip`:

```console
$ uv pip install --index https://wheels.astralhosted.com/simple/cu128/ pycuda
```

## Supported versions

Wheels are available for the following `pycuda` versions:

- [`2026.1`](https://github.com/astral-sh-build/build-pycuda/releases/tag/v2026.1)
- [`2025.1.3`](https://github.com/astral-sh-build/build-pycuda/releases/tag/v2025.1.3-r1)
- [`2025.1.2`](https://github.com/astral-sh-build/build-pycuda/releases/tag/v2025.1.2-r1)

The latest upstream release, PyCUDA 2026.1, supports the following combinations:

| Python   | `x86_64` CUDA                | `aarch64` CUDA         |
| -------- | ---------------------------- | ---------------------- |
| 3.9-3.14 | 12.4, 12.6, 12.8, 12.9, 13.0 | 12.6, 12.8, 12.9, 13.0 |

## License

build-pycuda is licensed under the [Apache License, Version 2.0](LICENSE).

<div align="center">
  <a target="_blank" href="https://astral.sh" style="background:none">
    <img src="https://raw.githubusercontent.com/astral-sh/ruff/main/assets/svg/Astral.svg" alt="Made by Astral">
  </a>
</div>
