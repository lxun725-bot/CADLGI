# CADLGI

**Content-Aware Dynamic Local–Global Interaction** for fine-grained classification of imperfect barley kernels.

This repository provides the PyTorch implementations of CADLGI-Rep (RepViT-M1.0), CADLGI-Swin (Swin-T), CADPA, CCE, DPG, ALGF, and prototype cross-attention, together with experiment configurations and training and evaluation results.

## Repository layout

```text
CADLGI/
  configs/repvit/           # Backbone, framework and module ablations
  configs/swin/            # Corresponding Swin-T configurations
  src/models/modules/      # CADPA, CCE, DPG, ALGF and cross-attention
  src/data/                # Indexed dataset loading and preprocessing
  src/engine/              # Training, validation selection and final evaluation
  model/pretrained/        # Upstream weights; binaries excluded from Git
  result/                  # Training histories and submission data
  env/                     # Pinned candidate Linux dependencies
  third_party/             # Retained upstream notices and licenses
```

## Environment

Runtime dependencies are listed in `requirements.txt`. Candidate Linux/Python 3.10 dependency pins are provided in `env/requirements-linux.txt`; install a compatible PyTorch and torchvision build separately for the target CUDA environment. These files specify runtime dependencies rather than a fully pinned training environment.

```bash
python -m pip install -r requirements.txt
```

## Data and pretrained initialization

Set `DATA.DATAPATH` to the dataset root, containing `category2index.json`, `train.json`, `val.json`, `test.json` and the referenced image files. Keep image paths in the JSON indices relative to that root. The default is `../split_cropped_dataset`.

Class IDs are `CY_KW=0`, `CY_PP=1`, `CY_PS=2`, `CY_SM=3`, `CY_SY=4`, `CY_ZC=5`. The split contains 31,999 training, 4,001 validation, and 4,000 test images. Anonymized split CSVs and class statistics are in `result/reproducibility/`. Images and private path-based JSON indices are not included. See [pretrained weight instructions](model/pretrained/README.md) for the initialization files.

## Training and evaluation

Run commands from the `CADLGI` root:

```bash
python run.py --config-file configs/repvit/cadlgi.yaml DATA.DATAPATH /path/to/dataset
python run.py --config-file configs/swin/cadlgi.yaml DATA.DATAPATH /path/to/dataset
```

Configurations default to training/validation only (`NO_TEST=True`). Checkpoints are selected by validation top1, retaining the earliest tied maximum. After freezing experimental choices, evaluate the selected checkpoint:

```bash
python test_best.py --config-file configs/repvit/cadlgi.yaml --weight /path/to/best_model.pth DATA.DATAPATH /path/to/dataset
```

Final evaluation writes per-image predictions and probabilities, prototype diagnostics, classification metrics and confusion matrices. Checkpoint implementation, model configuration, class mapping and split indices must match. The supplied experiment configurations use seed 0.

## Result data

The CSV files contain results recorded during model training and evaluation. Training histories include the per-epoch metrics available for each experiment; prediction files contain per-image model outputs.

- `result/CADLGI-Rep/` and `result/CADLGI-Swin/`: the six main 100-epoch training histories; full-model histories are in `full/metrics.csv`.
- `result/CADLGI-Rep/full/test_predictions.csv` and `result/CADLGI-Swin/full/test_predictions.csv`: per-image test predictions, class probabilities, and logits for both complete models.
- `result/reproducibility/figures_12_13/`: confusion-matrix images and count/percentage CSVs for Figures 12 and 13.
- `result/reproducibility/`: anonymized split indices, class statistics, split provenance, and per-sample prototype counts with K/V-length and complexity fields.
- `result/CADLGI-Rep/ALGF-internal-ablation/`: histories and predictions for static, concatenation, sample, and spatial fusion controls.
- `result/comparison/`: 100-epoch histories for 14 comparison models.

The dataset loader consumes path-based JSON indices, not the anonymized CSVs directly. Running training or inference requires access to the images and compatible JSON indices.

## RepViT ALGF internal ablations

Five configurations in `configs/repvit/algf_internal/` compare static fusion, concatenation, sample gating, spatial gating, and joint gating. All disable CADPA and retain the same fixed 32-prototype memory. The existing ALGF implementation supports all five modes.

```bash
python run.py --config-file configs/repvit/algf_internal/joint.yaml DATA.DATAPATH /path/to/dataset
```

All five configurations use seed 0. The `joint` configuration is provided as a runnable control; the internal-ablation result directory contains the four other modes.

## License

The existing [project license](LICENSE) and [RepViT upstream license](third_party/repvit/LICENSE) are retained without changes. Upstream provenance is documented in `third_party/repvit/README.md`.
