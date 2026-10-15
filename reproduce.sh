#!/bin/bash
# The whole pipeline, from the Kaggle data in data/ to work/submission.csv (about 3 hours on one RTX 4070;
# step 4 only produces the validation score and can be skipped, saving 30 minutes).
set -euo pipefail
cd "$(dirname "$0")"

python run.py tiles                                            # 1. detector training tiles
for k in 0 1 2 3; do python run.py train-detector --fold $k; done   # 2. four detectors, one per held-out fold
for k in 0 1 2 3; do python run.py propose --fold $k; done          # 3. out-of-fold and test boxes

python run.py train-segmenter --name validation --exclude-fold 0    # 4. validation: masks never trained on fold 0
python run.py segment --name validation --split heldout --fold 0
python run.py evaluate --name validation

python run.py train-segmenter --name final                          # 5. final mask model on all training images
for k in 0 1 2 3; do python run.py segment --name final --split test --fold $k; done
python run.py submit --name final                                   # 6. work/submission.csv
