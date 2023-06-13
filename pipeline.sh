#!/usr/bin/env bash

FG_DIR='data/fg_images'
RAW_MASKS='data/masks_raw'
PROCESSED_MASKS='data/masks_processed'

# create masks for foreground images
mkdir $RAW_MASKS
python3 get_masks.py $FG_DIR $RAW_MASKS
mkdir $PROCESSED_MASKS
python3 process_masks.py $RAW_MASKS $PROCESSED_MASKS

# use Laplacian pyramids to blend images
OUT_DIR="output/blended"
for file in data/fg_images/*.jpg; do
    img=$(basename "$file" .jpg)
    python3 insert_image.py \
        "${FG_DIR}/${img}.jpg" \
        "${PROCESSED_MASKS}/${img}.png" \
        data/bg_images/background.jpg \
        --save-as "output/blended/${img}.jpg" \
        --scale 0.25
done;
