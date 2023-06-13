from typing import Callable
import cv2
import numpy as np


class HSVTransform:
    """
    Applies several transforms only to image value component.
    """
    def __init__(self, value_transforms: list[Callable]):
        """
        Parameters
        ----------
        value_transforms : list[Callable]
            List of transformation functions. All functions should accept a
            2d numpy array of floats between 0 and 1.

        """
        self.vtransforms = value_transforms

    def __call__(self, img: np.ndarray
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert input image to HSV, apply transforms to V component, convert
        back to RGB.

        Parameters
        ----------
        img : np.ndarray
            RGB image in np.uint8 format.

        Returns
        -------
        rgb_noisy : np.ndarray
            Noisy RGB image created by modifying value channel in HSV
            representation.
            dtype: np.uint8
        hsv_original : np.ndarray
            Original image in HSV representation.
            dtype: np.uint8
        v_target : np.ndarray
            Value channel of original image in HSV representation scaled
            between -1 and 1.
            dtype: np.float32

        """
        # transform original RGB image to HSV
        assert img.ndim == 3, f"{img.ndim}"
        assert img.dtype == np.uint8, f"{img.dtype}"
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        hsv_original = hsv.copy()  # original image in HSV

        # transform value channel
        v = hsv[:, :, 2].astype(np.float64) / 255.  # in [0, 1] range
        v_target = v.astype(np.float32, copy=True)  # target neural net output
        for vtransform in self.vtransforms:
            v = vtransform(v)

        # use transformed value channel to create noisy rgb image
        v = (v * 255).clip(0, 255).round().astype(np.uint8)
        hsv[:, :, 2] = v
        rgb_noisy = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        # types: (np.uint8, np.uint8, np.float32)
        return rgb_noisy, hsv_original, v_target


class WhiteNoise:
    "Add white noise from range [-d, d]"
    def __init__(self, d: float = 0.1):
        assert 0 < d < 1
        self.d = d

    def __call__(self, x: np.ndarray) -> np.ndarray:
        noise = (np.random.random(x.shape) - 0.5) * self.d
        return x + noise


class RandomCutOut:
    "Cuts out a part of 2d array"
    def __init__(self, p: float = 0.5, height_min: float = 0.0,
                 height_max: float = 1.0, width_min: float = 0.0,
                 width_max: float = 1.0, fill_value: float = 0.0):
        self.p = p
        self.hmin = height_min
        self.hmax = height_max
        self.wmin = width_min
        self.wmax = width_max
        self.fill = fill_value

    def __call__(self, x: np.ndarray) -> np.ndarray:
        "Modifies input array"
        if np.random.random() > self.p:
            return x
        h, w = x.shape
        cut_height = int(round(
            (np.random.random() * (self.hmax - self.hmin) + self.hmin)
            * h))
        cut_width = int(round(
            (np.random.random() * (self.wmax - self.wmin) + self.wmin)
            * w))
        x1 = np.random.randint(low=0, high=w - cut_width)
        x[-cut_height:, x1:x1 + cut_width] = self.fill
        return x


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    img = cv2.imread("data/cars196/cars_train/00001.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    transform = HSVTransform([
        RandomCutOut(1.0, 0.5, 0.5, 0.1, 0.6, 0.5),
        WhiteNoise(0.2)
    ])
    noisy, _, _ = transform(img)
    plt.imshow(noisy)
    plt.show()
