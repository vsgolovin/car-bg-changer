from pathlib import Path
import cv2
import torch
from torchvision.models import segmentation as sgm_models
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

    # load segmentation model and its transform
    model = sgm_models.deeplabv3_mobilenet_v3_large(
        weights="DEFAULT")
    model.eval()
    transform = sgm_models.DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT \
        .transforms()

    # find car mask for every jpg image in src
    for file in input_dir.iterdir():
        if file.suffix.lower() != ".jpg":
            continue
        img = cv2.cvtColor(cv2.imread(str(file)), cv2.COLOR_BGR2RGB)
        height, width = img.shape[:2]
        x = torch.tensor(img.transpose(2, 0, 1))
        with torch.no_grad():
            output = model(transform(x).unsqueeze(0))
        # "car" is the 7th class in PASCAL VOC dataset
        car_mask = torch.softmax(output["out"], 1).squeeze(0).numpy()[7]
        car_mask = cv2.resize(car_mask, (width, height),
                              interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(
            str(output_dir / (file.stem + ".png")),
            (car_mask * 255.).round().astype("uint8")
        )


if __name__ == "__main__":
    main()
