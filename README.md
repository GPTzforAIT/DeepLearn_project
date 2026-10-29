# Palmer Penguins species classifier (MLP, PyTorch)

Capstone project for the Deep Learning course. A small multilayer perceptron, written and
trained from scratch, classifies penguins into **Adelie / Chinstrap / Gentoo** from four body
measurements plus island and sex. It is evaluated on a held-out test set against two baselines
(majority class and logistic regression), with an honest look at where it fails.

- Report (PDF): `[add link or file name]`
- Demo (runnable Colab): `[add Colab link]`

## Data

[Palmer Penguins](https://allisonhorst.github.io/palmerpenguins/), loaded at run time with
`seaborn.load_dataset("penguins")` (**the data is not stored in this repo**).
344 rows; 11 rows with missing values are dropped -> **333 rows**.

Features (9 inputs): `bill_length_mm`, `bill_depth_mm`, `flipper_length_mm`, `body_mass_g`
(standardized with **training-set** statistics only) + one-hot `island` (3) and `sex` (2).

Split: random 60 / 20 / 20 with a seeded generator -> **201 train / 66 validation / 66 test**.
The split is *not* stratified (see Limitations).

Citation: Gorman, Williams & Fraser (2014), *PLoS ONE* 9(3): e90081; Horst, Hill & Gorman (2020),
*palmerpenguins* R package.

## Model and training

`MLP`: Linear(9,32) -> ReLU -> Dropout(0.2) -> Linear(32,32) -> ReLU -> Linear(32,3); 1,475 parameters.

| Hyperparameter | Value |
|---|---|
| Loss | cross-entropy |
| Optimizer | Adam, lr = 1e-3 |
| Regularization | dropout 0.2 + weight decay 1e-4 |
| Batch size / epochs | 16 / 100 (the model after the last epoch is used; no early stopping) |
| Seed | 0 |

## Setup and reproduction

Requires Python 3.9+ (internet is needed once to download the dataset through seaborn).

```bash
pip install -r requirements.txt

python tests/test_data.py          # sanity checks: split sizes, no overlap, no scaler leakage
python -m src.train                # trains on train, monitors validation; saves models/ and results/
python -m src.failure_cases        # lists the validation errors (failure cases)
python -m src.evaluate             # ONE-SHOT test evaluation + baselines (run last, once)
python -m src.predict --bill-length 47.5 --bill-depth 15.0 --flipper-length 217 \
                      --body-mass 5000 --island Biscoe --sex Male
```

One-change-at-a-time ablations (validation only):

```bash
python -m src.train --dropout 0.5 --out models/abl_dropout05.pt --results-dir results/abl_dropout05
python -m src.train --hidden 64   --out models/abl_hidden64.pt  --results-dir results/abl_hidden64
```

Evaluation discipline: `src/train.py` never receives the test set. `src/evaluate.py` is the only
script that reads it and refuses to overwrite `results/test_metrics.json` without `--force`.

## Results

Numbers below come from the Colab run (seed 0); `python -m src.train` and `python -m src.evaluate`
regenerate them and write the plots and JSON files to `results/` (small differences are possible
between CPU and GPU).

**Test set (66 rows, evaluated once)**

| Model | Test accuracy |
|---|---|
| Majority class (Adelie) | 57.58% |
| Logistic regression (fitted on train) | `[fill from python -m src.evaluate]` |
| **MLP** | **100.00%** (66 / 66; exact 95% CI about 94.6% - 100%) |

A perfect score on 66 rows does not mean a perfect model. The test set contains roughly 10
Chinstrap penguins, the hardest class.

**Validation (66 rows, used for all choices)**

| Run | Val accuracy | Train-val gap |
|---|---|---|
| baseline (hidden 32, dropout 0.2) | 96.97% | +3.03% |
| dropout 0.5 | 96.97% | +3.03% |
| hidden 64 | 96.97% | +3.03% |

The three configurations are indistinguishable on validation (differences are within the noise
of a 66-row set), so the simplest one (baseline) was kept.

**Learning curves** (`results/learning_curves.png`): training accuracy reaches 100% by about
epoch 15 while validation accuracy plateaus at 96.97%; validation loss bottoms out near 0.02-0.03
and drifts up slightly later - mild overfitting that dropout and weight decay keep in check.

**Failure case** (`python -m src.failure_cases`, validation only): 2 of 17 Chinstrap penguins are
predicted as Adelie (Chinstrap recall 88.24%). The two species overlap in body mass and flipper
length, and Chinstrap has the fewest training examples (about 41 of 201). Confusion matrix:
`results/confusion_val.png`.

## Repository layout

```
src/data.py            load data, split, standardize (train statistics only)
src/model.py           the MLP
src/engine.py          training loop, evaluation, confusion matrix
src/train.py           train + validation diagnostics, saves checkpoint and plots
src/evaluate.py        one-shot test evaluation and baselines
src/failure_cases.py   validation errors with their raw features
src/predict.py         predict one new penguin from the checkpoint
tests/test_data.py     data-pipeline checks
notebooks/             the Colab notebook used for experiments and the demo
models/                trained checkpoint (small)
results/               curves, confusion matrix, metrics (generated)
```

## Limitations

- Very small data: validation and test have 66 rows each, so one row is about 1.5 percentage
  points and small differences are within noise.
- Single non-stratified split and single seed: class proportions differ between splits
  (Chinstrap, the hardest class, is rare in the test set).
- `island` may act as a shortcut for species (some species occur on only some islands); this was
  not ablated.
- Data come from one research site; results do not say anything about other penguin populations.
