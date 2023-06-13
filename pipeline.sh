#!/usr/bin/env bash

FG_DIR='data/fg_images'
RAW_MASKS='data/masks_raw'
PROCESSED_MASKS='data/masks_processed'

# create masks for foreground images
mkdir $RAW_MASKS
python3 get_masks.py $FG_DIR $RAW_MASKS
mkdir $PROCESSED_MASKS
python3 process_masks.py $RAW_MASKS $PROCESSED_MASKS

