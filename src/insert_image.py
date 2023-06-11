import cv2
import numpy as np


def blend_images(fg_img: np.ndarray, bg_img: np.ndarray, mask: np.ndarray,
                 pyr_size: int = 6) -> np.ndarray:
    """
    Blend images with a Laplacian pyramid of size `pyr_size`.
    """
    # build image pyramids
    assert fg_img.shape == bg_img.shape and mask.shape == fg_img.shape[:2]
    lp_fg = laplacian_pyramid(fg_img, pyr_size)
    lp_bg = laplacian_pyramid(bg_img, pyr_size)
    gp_mask = gaussian_pyramid(mask.astype(np.uint8), pyr_size)

    # merge foreground and background image pyramids
    pyr = []
    for fg_i, bg_i, mask_i in zip(lp_fg, lp_bg, gp_mask):
        lvl = bg_i.copy()
        mask_i = mask_i.astype(bool)
        lvl[mask_i] = fg_i[mask_i]
        pyr.append(lvl)

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


def main():
    img_name = "photo_2023-06-06_08-59-59"
    fg_img = load_rgb(f"data/fg_images/{img_name}.jpg")
    bg_img = load_rgb("data/bg_images/background.jpg")
    mask = load_grey(f"data/masks_processed/{img_name}.png")

    h, w, c = bg_img.shape
    scale = 0.5
    yc = int(h * 0.7)
    xc = w // 2

    fg_resized = cv2.resize(fg_img, (int(w * scale), int(h * scale)),
                            interpolation=cv2.INTER_AREA)
    mask_resized = cv2.resize(mask, (int(w * scale), int(h * scale)),
                              interpolation=cv2.INTER_AREA)
    y0 = yc - fg_resized.shape[0] // 2
    x0 = xc - fg_resized.shape[1] // 2

    fg = np.zeros((h, w, c), dtype=np.uint8)
    inds = (slice(y0, y0 + fg_resized.shape[0]),
            slice(x0, x0 + fg_resized.shape[1]))
    fg[inds] = fg_resized
    msk = np.zeros((h, w), dtype=bool)
    msk[inds] = mask_resized.astype(bool)

    new_img = blend_images(fg, bg_img, msk, 6)
    new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB)
    cv2.imwrite('temp.jpg', new_img)


if __name__ == "__main__":
    main()
