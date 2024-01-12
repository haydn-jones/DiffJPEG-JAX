# DiffJPEG: A Jax Implementation

This is a Jax implementation of the differentiable JPEG compression algorithm, based on the [PyTorch implementation](https://github.com/mlomnitz/DiffJPEG).


## Requirements

- JAX

## Installation

Can be installed with pip:

```bash
pip install diffjpeg_jax
```

## Usage

Unlike the PyTorch version, this is ML library agnostic, so it simply is implemented as a function.

```python

from diffjpeg_jax import diff_jpeg

img = ... # (H, W, C)
jpeg = diff_jpeg(img, quality=75)
```

Note: The implementation is not wrapped in JIT, so make sure to do that if you want to. For batch processing just use vmap.