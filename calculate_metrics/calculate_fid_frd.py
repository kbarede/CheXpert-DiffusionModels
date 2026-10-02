#!/usr/bin/env python3
"""Main script for the DDPM/NCSN chest-X-ray evaluation.
The script will make six copmerasion: 
    "real_vs_real"
    "ddpm_vs_real"
    "ncsn_vs_real"
    "ncsnEma_vs_real"
    "extNcsn_vs_real"
    "extNcsnEma_vs_real"

if you want to use less than six sets, u need to adapt "run_evaluation" function in helper_function.py and its call.

Edit the paths and settings in the first section below. Keep this file and
``helper_functions.py`` in the same directory.

Every image file must contain one X-ray only, without axes, titles, colour
bars, or a grid of multiple generated images.
"""

from pathlib import Path
from calculate_metrics.helper_functions import run_evaluation

# ======================= Path to images ==================================
# Folder containing the test-set of 2000 real images
REAL_TEST_DIR = Path("/PATH/TO/TEST-SET") # Test set must be kept sperate from training set!!

# DDPM-generated images.
DDPM_GENERATED_DIR = Path("/PATH/TO/DDPM/GENERATED-IMAGES")

# NCSN-generated images.
NCSN_GENERATED_DIR = Path("/PATH/TO/NCSN/GENERATED-IMAGES")

# NCSN-EMA-generated images.
NCSN_EMA_GENERATED_DIR = Path("/PATH/TO/NCSN-EMA/GENERATED-IMAGES")

# EXT-NCSN-generated images.
EXT_NCSN_GENERATED_DIR = Path("/PATH/TO/EXT-NCSN/GENERATED-IMAGES")

# EXT-NCSN-EMA-generated images.
EXT_NCSN_EMA_GENERATED_DIR = Path("/PATH/TO/EXT-NCSN-EMA/GENERATED-IMAGES")

# result folder
OUTPUT_DIR = Path("/PATH/TO/SAVE-RESULTS-AT")

# ================================ SETTINGS ======================================
SAMPLE_SIZE = 1000 # Number of images used in each comparison.
IMAGE_SIZE = 128 # Images are converted to this common size.
RESIZE_MODE = "stretch" # "stretch" = resize directly to IMAGE_SIZE x IMAGE_SIZE, also "pad" and "crop"
DEVICE = "auto"  # FID settings: "auto", "cuda", or "cpu"
FID_BATCH_SIZE = 32
FID_LOADER_WORKERS = 2
FRD_WORKERS = None
SEED = 2026 # Fixed selection seed for ground truth images
CALCULATE_REAL_BASELINE = True
FRD_USE_PAPER_LOG = True # FRD_USE_PAPER_LOG: True follows the equation in the FRDv1 journal paper: log(sqrt(d_squared)).
VERBOSE_FRD = False


def main() -> None:
    run_evaluation(
        real_dir=REAL_TEST_DIR,
        ddpm_dir=DDPM_GENERATED_DIR,
        ncsn_dir=NCSN_GENERATED_DIR,
        ncsn_ema_dir=NCSN_EMA_GENERATED_DIR,
        ext_ncsn_dir=EXT_NCSN_GENERATED_DIR,
        ext_ncsn_ema_dir=EXT_NCSN_EMA_GENERATED_DIR,
        output_dir=OUTPUT_DIR,
        sample_size=SAMPLE_SIZE,
        image_size=IMAGE_SIZE,
        resize_mode=RESIZE_MODE,
        batch_size=FID_BATCH_SIZE,
        loader_workers=FID_LOADER_WORKERS,
        frd_workers=FRD_WORKERS,
        device=DEVICE,
        seed=SEED,
        calculate_real_baseline=CALCULATE_REAL_BASELINE,
        frd_paper_log=FRD_USE_PAPER_LOG,
        verbose_frd=VERBOSE_FRD,
    )


if __name__ == "__main__":
    main()
