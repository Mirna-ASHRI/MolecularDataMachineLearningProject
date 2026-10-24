# Predicting Molecular Photostability from Small Data

Solution notebook for the Kaggle competition **Molecular Data Machine Learning**
(NSF Molecule Maker Lab Institute). The task is to predict the photostability (T80)
of thiophene-based oligomers from a very small dataset, and to identify which
molecular features explain it.

T80 is the time for a material's performance to fall to 80% of its initial value
under light exposure. A larger T80 means better photostability.

## The problem

- 42 labeled molecules, 143 usable computed features each (mass, heteroatom
  count, electronic density-of-states values such as TDOS4.0, ...), plus SMILES
- 9 test molecules with hidden T80
- Metric: mean squared logarithmic error (MSLE)
- Core difficulty: more features than molecules, so models can memorize noise and
  some features correlate with T80 purely by chance

## Approach

1. **Log target.** Train on `log1p(T80)` so ordinary squared error equals MSLE.
   T80 is strongly right-skewed (median 10.5, mean 22.7, max 103.9). Predictions
   are converted back with `expm1` and clipped at 0.
2. **Honest validation.** Repeated 5-fold CV (3 repeats) and leave-one-out (LOO),
   always compared with a "predict the mean" baseline. Imputation and scaling are
   inside scikit-learn pipelines, so nothing leaks from held-out molecules.
3. **Models.** Ridge at several penalty strengths, Random Forest, Lasso, and
   symbolic regression (gplearn).
4. **Feature importance, three ways.** Random Forest importance, permutation
   importance on held-out folds, and Lasso selection frequency over 100 bootstrap
   resamples, combined by average rank.
5. **Small-subset tests.** All subsets of the three key features (`TDOS4.0`,
   `NumHeteroatoms`, `Mass`), plus `SelectKBest` run inside CV as a fair test of
   automatic feature selection.
6. **Final model.** 70/30 blend of Ridge and a shallow Random Forest on the three
   key features, trained on all 42 molecules.

## Results

Cross-validated MSLE (lower is better):

| Model | Features | 5-fold x3 | Leave-one-out |
|---|---|---|---|
| Mean baseline | none | 1.133 | 1.125 |
| Ridge a=1 | all 143 | 3.578 | |
| Ridge a=10 | all 143 | 1.767 | |
| Ridge a=100 | all 143 | 0.863 | |
| Random Forest | all 143 | 0.828 | |
| Random Forest (depth 4) | 3 key features | 0.746 | 0.828 |
| **Ridge a=1** | **TDOS4.0 + NumHeteroatoms + Mass** | **0.647** | **0.685** |
| Ridge a=1 | TDOS4.0 + NumHeteroatoms | 0.655 | |
| Blend 70% Ridge + 30% RF | 3 key features | | 0.680 |
| Ridge + SelectKBest inside CV | auto-selected, k=10 | 0.929 | |
| Symbolic regression | 3 key features | 1.252 | |

The all-feature models reached a training error of only 0.34 RMSLE but a
cross-validated error near 0.9, which is clear memorization.

### Feature importance

| Feature | RF rank | Permutation rank | Lasso freq. rank | Average |
|---|---|---|---|---|
| TDOS4.0 | 4 | 1 | 4 | 3.0 |
| NumHeteroatoms | 6 | 5 | 1 | 4.0 |
| Mass | 3 | 3 | 29 | 11.7 |

Ridge on subsets of the key features: `TDOS4.0 + NumHeteroatoms` (0.655) is almost
as good as all three (0.647) and clearly better than any single feature (0.76 to
0.85), so two features carry nearly all of the signal. Standardized Ridge
coefficients were -0.57 (TDOS4.0), -0.57 (NumHeteroatoms) and +0.28 (Mass).

## Findings

- Strongly regularized models on a few features beat models on all 143 features,
  which barely beat the baseline or were worse than it.
- `TDOS4.0` and `NumHeteroatoms` were the most consistently selected features
  across the three methods. `TDOS3.9` ranks right behind TDOS4.0 because the two
  are neighboring, highly correlated bins.
- In this dataset, higher TDOS4.0 and more heteroatoms go with shorter T80.
  A possible explanation (a hypothesis, not a conclusion) is that heteroatoms add
  reactive sites for photodegradation.
- Symbolic regression found a formula that reduced to roughly
  `log T80 ~ 2.65 - standardized NumHeteroatoms`, but it generalized worse than
  the baseline. A plain linear model is about as good as this data allows.
- The final blend (LOO 0.680) and plain Ridge (LOO 0.685) are a statistical tie.

## Limitations

- Only 42 training and 9 test molecules: CV estimates are noisy, 5-fold folds hold
  about 8 molecules, and any leaderboard score on 9 molecules is mostly luck.
- The three key features were identified by the competition organizers using the
  full dataset, so scores using them are optimistic. When features were selected
  automatically inside CV, MSLE was about 0.93, much closer to the baseline.
  A realistic expectation for finding features from scratch is nearer 0.9 than 0.65.
- The brute-force subset search (Section 6F) tested hundreds of combinations using
  rankings computed on all data. Its best result (0.603) is exploratory and
  affected by the winner's curse.
- Correlated features (for example TDOS3.9 and TDOS4.0) split importance, so a low
  rank does not prove a feature is unimportant.
- Importance shows association with T80, not proven causation.
- The Ridge alpha was chosen on the same CV used to report scores, which is
  slightly optimistic.

## Not pre-trained

No pre-trained molecular models were used, so this entry belongs to the
non-pre-trained category.

## Repository contents

- `moleculardatamachinelearningproject.ipynb`: full analysis with outputs
- `submission.csv`: predictions for the 9 test molecules
- `README.md`: this file

The competition data (`train.csv`, `test.csv`) is not included. Download it from the
[competition page](https://www.kaggle.com/competitions/molecular-machine-learning)
and place it where the notebook can find it.

## How to run

1. Open the notebook on Kaggle with the competition data attached, or locally with
   the CSV files and an updated path.
2. Install dependencies: `pip install numpy pandas matplotlib scikit-learn rdkit gplearn`
3. Run all cells top to bottom.

## Tools

Python, pandas, NumPy, scikit-learn, RDKit, matplotlib, gplearn
