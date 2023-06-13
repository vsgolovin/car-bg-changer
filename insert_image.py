import cv2
import numpy as np
import click
from src.blending import scale_and_shift, blend_images


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


def load_rgb(path):
    img = cv2.imread(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_grey(path):
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)


if __name__ == "__main__":
    main()
