import cv2
import numpy as np
import click


@click.command()
@click.argument("foreground", type=click.Path())
@click.argument("mask", type=click.Path())
@click.argument("background", type=click.Path())
@click.option("--save-as", type=click.Path(), default="output.jpg")
@click.option("--scale", type=float, default=1.0,
              help="foreground image scaling factor")
@click.option("--xloc", type=float, default=0.5,
              help="fg center location on bg image (x axis)")
@click.option("--yloc", type=float, default=0.66,
              help="fg center location on bg image (y axis)")
@click.option("--pyramid-size", type=int, default=6,
              help="size of Laplacian pyramids used for blending")
def main(foreground: str, mask: str, background: str, save_as: str,
         scale: float, xloc: float, yloc: float, pyramid_size: int):
    # load images
    fg_img = load_rgb(foreground)
    bg_img = load_rgb(background)
    mask_img = load_grey(mask)
    mask_img = mask_img.astype(np.float64) / 255.

    # absolute coordinates of foreground image center on background
    h, w, _ = bg_img.shape
    yc = int(round(h * yloc))
    xc = int(round(w * xloc))

    # make foreground and background aligned
    fg_img, mask_img = scale_and_shift(fg_img, mask_img, scale, h, w, yc, xc)
    # perform blending
    new_img = blend_images(fg_img, bg_img, mask_img, pyramid_size)
    new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(save_as, new_img)


def blend_images(fg_img: np.ndarray, bg_img: np.ndarray, mask: np.ndarray,
                 pyr_size: int = 6) -> np.ndarray:
    """
    Blend images with a Laplacian pyramid of size `pyr_size`.
    """
    # build image pyramids
    assert fg_img.shape == bg_img.shape and mask.shape == fg_img.shape[:2]
    lp_fg = laplacian_pyramid(fg_img, pyr_size)
    lp_bg = laplacian_pyramid(bg_img, pyr_size)
    gp_mask = gaussian_pyramid(mask, pyr_size)

    # merge foreground and background image pyramids
    pyr = []
    for i in range(pyr_size + 1):
        fg_i = lp_fg[i].astype(np.float64)
        bg_i = lp_bg[i].astype(np.float64)
        mask_i = gp_mask[i][..., np.newaxis]
        lvl = fg_i * mask_i + bg_i * (1 - mask_i)
        if i != pyr_size:
            pyr.append(lvl.round().astype(np.int16))
        else:
            pyr.append(lvl.round().clip(0, 255).astype(np.uint8))

    # use newly created pyramid to reconstuct image
    return reconstruct_from_laplacian_pyramid(pyr)


def gaussian_pyramid(img: np.ndarray, n: int = 6) -> list[np.ndarray]:
    lvl = img.copy()
    pyramid = [lvl]
    for _ in range(n):
        lvl = cv2.pyrDown(lvl)
        pyramid.append(lvl)
    return pyramid


def laplacian_pyramid(img: np.ndarray, n: int = 6) -> list[np.ndarray]:
    "All but last level use `np.int16` type"
    g_pyr = gaussian_pyramid(img, n)
    pyramid = []
    for i in range(n):
        diff = g_pyr[i].astype(np.int16)
        upscaled = cv2.pyrUp(g_pyr[i + 1])[:diff.shape[0], :diff.shape[1]]
        lvl = diff - upscaled
        pyramid.append(lvl)
    pyramid.append(g_pyr[-1])
    return pyramid


def reconstruct_from_laplacian_pyramid(pyr: list[np.ndarray]) -> np.ndarray:
    i = len(pyr) - 1
    img = pyr[i]
    while i != 0:
        diff = pyr[i - 1]
        img = cv2.pyrUp(img)[:diff.shape[0], :diff.shape[1]]
        img = (img + diff).clip(0, 255).astype(np.uint8)
        i -= 1
    return img


def load_rgb(path):
    img = cv2.imread(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_grey(path):
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)


def scale_and_shift(img: np.ndarray, mask: np.ndarray, scale: float,
                    height: int, width: int, yc: int, xc: int
                    ) -> tuple[np.ndarray, np.ndarray]:
    """
    Scale image + its mask and place it at location `(yc, xc)`. This method
    uses constant value (0) padding. Image center is computed from mask.

    Parameters
    ----------
    img : np.ndarray
        Image to scale.
    mask : np.ndarray
        Image mask, must have same height and width as `img`.
    scale : float
        Scaling coefficient, should be <= 1 if performing downscaling.
    height : int
        Output image height.
    width : int
        Output image width.
    yc : int
        Location on y-axis (first coordinate) to place image center.
    xc : int
        Location on x-axis (second coordinate) to place image center.

    """
    # check input shapes
    assert img.shape[:2] == mask.shape

    # scaling
    if scale == 1.0:
        img_rescaled = img.copy()
        mask_rescaled = mask.copy()
    else:
        inter_flag = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
        new_h, new_w = [int(round(sz * scale)) for sz in img.shape[:2]]
        img_rescaled = cv2.resize(img, (new_w, new_h),
                                  interpolation=inter_flag)
        mask_rescaled = cv2.resize(mask, (new_w, new_h),
                                   interpolation=inter_flag)

    yc0, xc0 = get_mask_center(mask_rescaled)

    dy = yc - yc0
    y1, y2 = -dy, height - dy
    padding_y = (max(-y1, 0), max(y2 - img_rescaled.shape[0], 0))
    inds_y = slice(max(y1, 0), min(y2, img_rescaled.shape[0]))

    dx = xc - xc0
    x1, x2 = -dx, width - dx
    padding_x = (max(-x1, 0), max(x2 - img_rescaled.shape[1], 0))
    inds_x = slice(max(x1, 0), min(x2, img_rescaled.shape[1]))

    img_out = np.pad(img_rescaled[inds_y, inds_x],
                     (padding_y, padding_x, (0, 0)))
    mask_out = np.pad(mask_rescaled[inds_y, inds_x], (padding_y, padding_x))
    return img_out, mask_out


def get_mask_center(mask: np.ndarray) -> tuple[int, int]:
    xx, yy = np.meshgrid(np.arange(mask.shape[1]), np.arange(mask.shape[0]))
    s = np.sum(mask)
    xc = int(round(np.sum(xx * mask) / s))
    yc = int(round(np.sum(yy * mask) / s))
    return yc, xc


if __name__ == "__main__":
    main()
