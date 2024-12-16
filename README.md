# Random Forest and Hyperparameter Optimization Workflow

This project involves using a Random Forest classifier for defect prediction in software datasets. It includes preprocessing, hyperparameter optimization using Hill Climbing, and evaluation of the model with the optimized parameters.

---

## Workflow Steps

### Step 1: Data Grouping

Script: `step1_data_grouping.py`

- Groups raw CSV files based on predefined criteria.
- Saves grouped files into the `Dataset` folder.

---

### Step 2: Join Files

Script: `step2_join_files.py`

- Combines grouped files into larger datasets for each group (e.g., `group_1.csv`, `group_2.csv`).
- Outputs these combined datasets into the `Dataset/CSV only` folder.

---

### Step 3: Apply SMOTE

Script: `step3_SMOTE.py`

- Balances the data in each group using the SMOTE technique.
- Outputs balanced datasets as `group_1_SMOTE.csv`, `group_2_SMOTE.csv`, etc., in the `Dataset/CSV only` folder.

---

### Step 4: Hill Climbing for Hyperparameter Optimization

Script: `step4_hill_climbing.py`

- Uses the Hill Climbing algorithm to find a single, unified set of optimized hyperparameters for the Random Forest model.
- Evaluates the same set of hyperparameters across all SMOTE-balanced datasets (`group_1_SMOTE.csv`, `group_2_SMOTE.csv`, etc.).
- Saves the best unified hyperparameters in `Result/unified_hill_climbing_results.txt`.
- Progress, including average accuracy for each iteration, is printed to the console and logged.
- An **accuracy progress plot** (`unified_accuracy_progress.png`) is saved to visualize the optimization process.

---

### Step 5: Random Forest with Unified Optimized Parameters

Script: `step5_random_forest.py`

- Reads the unified best hyperparameters from `unified_hill_climbing_results.txt` (generated in Step 4).
- Trains and evaluates the Random Forest model **using the same optimized parameters** across all groups.
- Outputs include:
  - **Accuracy, Classification Report, and Confusion Matrix**: Saved to `Result/result.txt`.
  - **Feature Importance Plots**: Saved as `group_1_SMOTE_importance.png`, `group_2_SMOTE_importance.png`, etc., in the `Result` folder.
  - **Accuracy Comparison Plot**: Saved as `accuracy_comparison.png` in the `Result` folder.

---

## Requirements

To run the scripts, ensure the following Python libraries are installed:

- `pandas`
- `scikit-learn`
- `matplotlib`
- `imbalanced-learn`

Install all dependencies with:

```bash
pip install pandas scikit-learn matplotlib imbalanced-learn
```
