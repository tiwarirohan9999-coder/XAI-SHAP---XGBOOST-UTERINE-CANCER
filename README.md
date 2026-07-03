# Uterine Cancer Report (XGBoost + SMOTEN)

A classification project on the **Uterine Corpus Endometrial Carcinoma (TCGA, PanCancer Atlas)** dataset, predicting patient survival status using XGBoost, with an Explainable AI (XAI) section showing *why* the model makes the predictions it does.

## What this script does

1. **Loads and explores** the dataset (`UterineCancer.csv`) — shape, info, and summary stats.
2. **Cleans the data** — drops ID columns and columns that would leak the answer (`Disease Free Status`, `Disease-specific Survival status` are other versions of the same outcome), then drops any row with missing values.
3. **Builds the target** — `Overall Survival Status` is converted into a simple binary label: `0 = Living`, `1 = Deceased`.
4. **Visualizes the data** — pie chart of class distribution, correlation heatmap of features vs. target.
5. **Encodes every feature as categorical**:
   - Numeric columns (`Mutation Count`, `Fraction Genome Altered`, `Diagnosis Age`, `MSI MANTIS Score`, `MSIsensor Score`) are binned into 4 quartile groups (`0–3`) using `pd.qcut`.
   - Text columns (`Race Category`, `Subtype`, `Tumor Type`) are label-encoded into integer codes.
   - This is required because **SMOTEN only works on categorical data** (unlike regular SMOTE, which needs continuous/scaled numbers).
6. **Balances the training data with SMOTEN** — the dataset is imbalanced (~83% Living / ~17% Deceased), so SMOTEN generates synthetic minority-class (Deceased) rows until the training set is 50/50. Only the training set is balanced — the test set stays real and imbalanced, which is correct practice.
7. **Trains a single model: XGBoost** (no Logistic Regression, no Decision Tree in this version).
8. **Evaluates the model** — accuracy, recall, F1 score, MSE, classification report, and a confusion matrix.
9. **Explains the model (XAI)** using two techniques:
   - **XGBoost Feature Importance** — how much each feature helped the model's trees split the data cleanly.
   - **SHAP Summary Plot** — how much each feature contributed to individual predictions, across the whole test set.

## Requirements

Install the required packages before running:

```bash
pip install pandas numpy scikit-learn imbalanced-learn xgboost matplotlib seaborn shap
```

## How to run

1. Place `UterineCancer.csv` in the same folder as the script.
2. Run the script top to bottom, in order (important if using a notebook — cells depend on earlier ones being run first):

```bash
python uterine_cancer_report.py
```

Or open it as a Jupyter notebook (`# %%` markers denote notebook cells) and run all cells sequentially.

## Output files generated

| File | Description |
|---|---|
| `pie_chart.png` | Class distribution of the target (Living vs. Deceased) |
| `heatmap.png` | Correlation heatmap of encoded features vs. target |
| `confusion_matrix.png` | Confusion matrix for the XGBoost model |
| `xai_xgb_importance.png` | XGBoost feature importance |
| `xai_shap_summary.png` | SHAP summary plot of feature contributions |

## Dataset

Source: [Kaggle – Uterine Corpus Endometrial Carcinoma (TCGA, PanCancer Atlas)](https://www.kaggle.com/datasets/yeganehbavafa/uterine-corpus-endometrial-carcinoma), originally from the TCGA PanCancer Atlas initiative. 529 patients, clinical/genomic features including mutation count, fraction of genome altered, diagnosis age, MSI scores, race, molecular subtype, and tumor type.

## Notes

- The target is imbalanced (~83% Living / ~17% Deceased) — this is why SMOTEN is used, and why accuracy alone can be misleading here; recall and F1 on the "Deceased" class matter more.
- Numeric features are quartile-binned specifically so SMOTEN (categorical-only) can be applied. This trades some precision in the raw values for compatibility with SMOTEN's distance metric.
- No feature scaling is used — XGBoost is tree-based and scale-invariant, so `StandardScaler` isn't needed here (unlike the Logistic Regression models in the Iris report).
- Given the small number of real Deceased cases (79 total), results — especially recall on the minority class — reflect a genuine data-size limitation, not a bug in the pipeline.
## AUTHOR
ROHAN TIWARI , RESEARCHER , COE-DSAI , TIET-UQ CENTER
