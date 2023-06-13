from pathlib import Path
import cv2
import numpy as np
import click


@click.command()
@click.argument("src", type=click.Path())
@click.argument("dst", type=click.Path())
def main(src: str, dst: str):
    # check input and output directories
    input_dir = Path(src)
    assert input_dir.exists(), "src does not exist"
    output_dir = Path(dst)
    assert output_dir.exists(), "dst does not exist"

    # process every mask from src
    for file in input_dir.iterdir():
        if file.suffix.lower() != ".png":
            continue
        raw = cv2.imread(str(file), cv2.IMREAD_GRAYSCALE)
        # thresholding
        _, thresh = cv2.threshold(raw, 100, 255, cv2.THRESH_BINARY)
        # keep only largest connected component
        retval, labels, stats, _ = cv2.connectedComponentsWithStats(thresh)
        assert retval >= 2
        label = np.argmax(stats[1:, -1]) + 1
        mask = np.zeros_like(raw)
        mask[labels == label] = 255
        # export new mask
        cv2.imwrite(str(output_dir / file.name), mask)


if __name__ == "__main__":
    main()
