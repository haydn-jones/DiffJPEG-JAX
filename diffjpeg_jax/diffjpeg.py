"""
DiffJPEG in JAX

Based on
- https://github.com/necla-ml/Diff-JPEG
- https://github.com/mlomnitz/DiffJPEG
- https://machine-learning-and-security.github.io/papers/mlsec17_paper_54.pdf
"""
# ruff:noqa: F722

from typing import Tuple
import jax.numpy as jnp
import jax
from jaxtyping import Float, Array


Y_TABLE = jnp.array(
    [
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99],
    ],
    dtype=jnp.float32,
)

C_TABLE = jnp.array(
    [
        [17, 18, 24, 47, 99, 99, 99, 99],
        [18, 21, 26, 66, 99, 99, 99, 99],
        [24, 26, 56, 99, 99, 99, 99, 99],
        [47, 66, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
    ],
)


def quality_to_scale(quality: Float[Array, "1"]) -> Float[Array, "1"]:
    return jnp.floor(
        jnp.where(
            quality < 50,
            5000.0 / quality,
            200.0 - quality * 2,
        )
    )


def rgb_to_ycbcr(image: Float[Array, "H W 3"]) -> Float[Array, "H W 3"]:
    """Converts RGB image to YCbCr. Channels last"""
    matrix = jnp.array(
        [
            [0.299, 0.587, 0.114],
            [-0.168736, -0.331264, 0.5],
            [0.5, -0.418688, -0.081312],
        ],
        dtype=jnp.float32,
    ).T
    shift = jnp.array([0.0, 128.0, 128.0])

    image = jnp.swapaxes(image, 0, 1)
    result = jnp.tensordot(image, matrix, axes=1) + shift
    return result.reshape(image.shape)


def ycbcr_to_rgb(image: Float[Array, "H W 3"]) -> Float[Array, "H W 3"]:
    """Converts YCbCr image to RGB. Channels last"""
    matrix = jnp.array(
        [
            [1.0, 0.0, 1.402],
            [1, -0.344136, -0.714136],
            [1, 1.772, 0],
        ],
        dtype=jnp.float32,
    ).T
    shift = jnp.array([0.0, -128.0, -128.0])
    result = jnp.tensordot(image + shift, matrix, axes=1)
    return result


def chroma_subsampling(image: Float[Array, "H W 3"]) -> Float[Array, "H W 3"]:
    """Chroma subsampling on CbCr channels. Channels last"""
    H, W, C = image.shape

    y = image[..., 0]
    cb = jax.image.resize(
        image[..., 1], (H // 2, W // 2), method="bilinear", antialias=True
    )
    cr = jax.image.resize(
        image[..., 2], (H // 2, W // 2), method="bilinear", antialias=True
    )

    return y, cb, cr


def chroma_upsampling(
    y: Float[Array, "H W"], cb: Float[Array, "H W"], cr: Float[Array, "H W"]
) -> Float[Array, "H W 3"]:
    def repeat(x: jnp.array, k: int = 2):
        H, W = x.shape
        x = x.reshape(1, H, W, 1)
        x = jnp.tile(x, (1, 1, k, k))
        x = x.reshape(H * k, W * k)
        return x

    cb = repeat(cb)
    cr = repeat(cr)
    return jnp.stack([y, cb, cr], axis=-1)


def block_splitting(image: Float[Array, "H W"], k: int = 8) -> Float[Array, "H W k k"]:
    H, W = image.shape
    image = image.reshape(H // k, k, W // k, k)
    image = jnp.transpose(image, (2, 0, 3, 1))
    image = image.reshape(-1, k, k)
    return image


def block_merging(
    patches: Float[Array, "H W k k"], H: int, W: int, k: int = 8
) -> Float[Array, "H W"]:
    patches = patches.reshape(H // k, W // k, k, k)
    patches = patches.transpose((0, 2, 1, 3))
    return patches.reshape(H, W)


def quantize(
    image: Float[Array, "H W k k"],
    table: Float[Array, "k k"],
    quality: Float[Array, "1"],
) -> Float[Array, "H W k k"]:
    table = table * quality_to_scale(quality)
    table = jnp.clip(jnp.floor((table + 50.0) / 100.0), 1, 255)
    output = image / table
    output = jnp.round(output)
    return output


def dequantize(
    image: Float[Array, "H W k k"],
    table: Float[Array, "k k"],
    quality: Float[Array, "1"],
) -> Float[Array, "H W k k"]:
    table = table * quality_to_scale(quality)
    # Perform scaling
    output = image * jnp.clip(jnp.floor((table + 50.0) / 100.0), 1, 255)
    return output


def dct8x8(image: Float[Array, "H W 8 8"]) -> Float[Array, "H W 8 8"]:
    """Applies DCT on 8x8 blocks. Channels last"""

    indices = jnp.arange(8)
    x, y, u, v = jnp.meshgrid(indices, indices, indices, indices, indexing="ij")
    tensor = jnp.cos((2 * x + 1) * u * jnp.pi / 16) * jnp.cos(
        (2 * y + 1) * v * jnp.pi / 16
    )

    alpha = jnp.array([1.0 / jnp.sqrt(2)] + [1] * 7)
    scale = jnp.outer(alpha, alpha) * 0.25

    image = image - 128
    result = scale * jnp.tensordot(image, tensor, axes=2)
    result = result.reshape(image.shape)
    return result


def idct8x8(image: Float[Array, "H W 8 8"]) -> Float[Array, "H W 8 8"]:
    """Applies IDCT on 8x8 blocks. Channels last"""

    indices = jnp.arange(8)
    x, y, u, v = jnp.meshgrid(indices, indices, indices, indices, indexing="ij")
    tensor = jnp.cos((2 * u + 1) * x * jnp.pi / 16) * jnp.cos(
        (2 * v + 1) * y * jnp.pi / 16
    )

    alpha = jnp.array([1.0 / jnp.sqrt(2)] + [1] * 7)
    alpha = jnp.outer(alpha, alpha)

    image = image * alpha
    result = 0.25 * jnp.tensordot(image, tensor, axes=2) + 128
    result.reshape(image.shape)
    return result


def compress_jpeg(
    image: Float[Array, "H W 3"], quality: Float[Array, "1"]
) -> Tuple[Float[Array, "H W 8 8"], Float[Array, "H W 8 8"], Float[Array, "H W 8 8"]]:
    y, cb, cr = chroma_subsampling(rgb_to_ycbcr(image))

    y = quantize(dct8x8(block_splitting(y)), Y_TABLE, quality=quality)
    cb = quantize(dct8x8(block_splitting(cb)), C_TABLE, quality=quality)
    cr = quantize(dct8x8(block_splitting(cr)), C_TABLE, quality=quality)

    return y, cb, cr


def decompress_jpeg(
    y: Float[Array, "H W 8 8"],
    cb: Float[Array, "H W 8 8"],
    cr: Float[Array, "H W 8 8"],
    H: int,
    W: int,
    quality: Float[Array, "1"],
) -> Float[Array, "H W 3"]:
    y = block_merging(idct8x8(dequantize(y, Y_TABLE, quality=quality)), H, W)
    cb = block_merging(
        idct8x8(dequantize(cb, C_TABLE, quality=quality)), H // 2, W // 2
    )
    cr = block_merging(
        idct8x8(dequantize(cr, C_TABLE, quality=quality)), H // 2, W // 2
    )

    image = chroma_upsampling(y, cb, cr)
    image = ycbcr_to_rgb(image)
    image = jnp.clip(image, 0, 255)
    return image


def diff_jpeg(
    image: Float[Array, "H W 3"], quality: Float[Array, "1"]
) -> Float[Array, "H W 3"]:
    """
    Applies DiffJPEG compression on an image. The input should be in [0, 255] range, not [0, 1].

    Args:
        image: Image to compress (H, W, C)
        quality: Quality in (0, 100)
    """
    H, W, _ = image.shape
    quality = jnp.asarray(quality, dtype=jnp.float32)

    # Pad image to multiple of 16 as we use 8x8 blocks on cb and cr channels which are 2x subsampled
    hpad = 16 - (H % 16)
    wpad = 16 - (W % 16)

    H = H + hpad
    W = W + wpad

    image = jnp.pad(
        image, ((0, hpad), (0, wpad), (0, 0)), mode="constant", constant_values=0
    )

    y, cb, cr = compress_jpeg(image, quality=quality)

    image = decompress_jpeg(y, cb, cr, H=H, W=W, quality=quality)
    image = image[: H - hpad, : W - wpad]
    return image
