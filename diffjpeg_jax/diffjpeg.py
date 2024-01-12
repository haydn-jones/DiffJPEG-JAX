"""
    DiffJPEG in JAX

    Based on
    https://github.com/mlomnitz/DiffJPEG
    and
    https://machine-learning-and-security.github.io/papers/mlsec17_paper_54.pdf
"""

import jax.numpy as jnp
import jax

y_table = jnp.array(
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
).T

c_table = jnp.full((8, 8), 99, dtype=jnp.float32)
c_table = c_table.at[:4, :4].set(
    jnp.array(
        [
            [17, 18, 24, 47],
            [18, 21, 26, 66],
            [24, 26, 56, 99],
            [47, 66, 99, 99],
        ]
    ).T
)


def quality_to_factor(quality: float) -> float:
    return (
        jnp.where(
            quality < 50,
            5000.0 / quality,
            200.0 - quality * 2,
        )
        / 100.0
    )


def rgb_to_ycbcr(image: jnp.ndarray) -> jnp.ndarray:
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


def ycbcr_to_rgb(image: jnp.ndarray) -> jnp.ndarray:
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


def chroma_subsampling(image: jnp.ndarray) -> jnp.ndarray:
    """Chroma subsampling on CbCr channels. Channels last"""
    H, W, C = image.shape

    y = image[..., 0]
    cb = jax.image.resize(
        image[..., 1], (H // 2, W // 2), method="linear", antialias=False
    )
    cr = jax.image.resize(
        image[..., 2], (H // 2, W // 2), method="linear", antialias=False
    )

    return y, cb, cr


def chroma_upsampling(y: jnp.ndarray, cb: jnp.ndarray, cr: jnp.ndarray) -> jnp.ndarray:
    def repeat(x: jnp.array, k: int = 2):
        H, W = x.shape
        x = x.reshape(1, H, W, 1)
        x = jnp.tile(x, (1, 1, k, k))
        x = x.reshape(H * k, W * k)
        return x

    cb = repeat(cb)
    cr = repeat(cr)
    return jnp.stack([y, cb, cr], axis=-1)


def block_splitting(image: jnp.ndarray, k: int = 8) -> jnp.ndarray:
    H, W = image.shape
    image = image.reshape(H // k, k, W // k, k)
    image = jnp.transpose(image, (2, 0, 3, 1))
    image = image.reshape(-1, k, k)
    return image


def block_merging(patches: jnp.ndarray, H: int, W: int, k: int = 8) -> jnp.ndarray:
    patches = patches.reshape(H // k, W // k, k, k)
    patches = patches.transpose((0, 2, 1, 3))
    return patches.reshape(H, W)


def y_quantize(image: jnp.ndarray, factor: float = 1.0) -> jnp.ndarray:
    image = image / (y_table * factor)
    image = jnp.round(image)
    return image


def y_dequantize(image: jnp.ndarray, factor: float = 1.0) -> jnp.ndarray:
    image = image * (y_table * factor)
    return image


def c_quantize(image: jnp.ndarray, factor: float = 1.0) -> jnp.ndarray:
    image = image / (c_table * factor)
    image = jnp.round(image)
    return image


def c_dequantize(image: jnp.ndarray, factor: float = 1.0) -> jnp.ndarray:
    image = image * (c_table * factor)
    return image


def dct8x8(image: jnp.ndarray) -> jnp.ndarray:
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


def idct8x8(image: jnp.ndarray) -> jnp.ndarray:
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


def compress_jpeg(image: jnp.ndarray, factor: float = 1.0) -> jnp.ndarray:
    y, cb, cr = chroma_subsampling(rgb_to_ycbcr(image * 255.0))

    y = y_quantize(dct8x8(block_splitting(y)), factor=factor)
    cb = c_quantize(dct8x8(block_splitting(cb)), factor=factor)
    cr = c_quantize(dct8x8(block_splitting(cr)), factor=factor)

    return y, cb, cr


def decompress_jpeg(
    y: jnp.ndarray,
    cb: jnp.ndarray,
    cr: jnp.ndarray,
    H: int,
    W: int,
    factor: float = 1.0,
) -> jnp.ndarray:
    y = block_merging(idct8x8(y_dequantize(y, factor=factor)), H, W)
    cb = block_merging(idct8x8(c_dequantize(cb, factor=factor)), H // 2, W // 2)
    cr = block_merging(idct8x8(c_dequantize(cr, factor=factor)), H // 2, W // 2)

    image = chroma_upsampling(y, cb, cr)
    image = ycbcr_to_rgb(image)
    image = jnp.clip(image, 0, 255)
    return image / 255.0


def diff_jpeg(image: jnp.ndarray, quality: float = 75.0) -> jnp.ndarray:
    """
    Applies DiffJPEG compression on an image.

    Args:
        image: Image to compress (H, W, C)
        quality: Quality factor in (0, 100)
    """
    H, W, _ = image.shape

    # Pad image to multiple of 16
    hpad = 16 - (H % 16)
    wpad = 16 - (W % 16)

    H = H + hpad
    W = W + wpad

    image = jnp.pad(
        image, ((0, hpad), (0, wpad), (0, 0)), mode="constant", constant_values=0
    )

    factor = quality_to_factor(quality)
    y, cb, cr = compress_jpeg(image, factor=factor)

    image = decompress_jpeg(y, cb, cr, H=H, W=W, factor=factor)
    image = image[: H - hpad, : W - wpad]
    return image
