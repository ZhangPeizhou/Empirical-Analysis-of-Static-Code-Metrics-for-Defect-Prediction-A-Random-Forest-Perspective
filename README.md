# Instruction for Running the Program

This project processes datasets, balances them using SMOTE, and evaluates results using a Random Forest classifier. The program is structured into four sequential Python scripts.

## Steps to Execute

1. Run the Python scripts **step by step**, starting from `step1_data_grouping.py` and ending with `step4_random_forest.py`.
2. The final results, including detailed metrics, feature importance, and comparison plots, will be stored in the **Result** folder.

---

## Description of Each Python Script

### `step1_data_grouping.py`

- **Purpose**: Groups and combines related CSV files based on the configuration provided in `grouped_file_columns.csv`.
- **Input**:
  - A folder containing the dataset CSV files (`Dataset/CSV only`).
  - A configuration file (`grouped_file_columns.csv`) listing file groups.
- **Process**:
  - Reads the grouping configuration.
  - Combines the files belonging to each group into a single CSV.
  - Saves the combined files with names derived from group names (e.g., `group_1.csv`, `group_2.csv`).
- **Output**:
  - Combined group files stored in the dataset folder.
  - Prints row count statistics for each file and group.

### `step2_join_files.py`

- **Purpose**: Combines grouped files into unified datasets for subsequent processing.
- **Input**: Grouped files from the previous step (`group_1.csv`, `group_2.csv`, etc.).
- **Process**:
  - Merges grouped files as specified in the grouping configuration.
  - Performs integrity checks, ensuring that all files exist.
- **Output**:
  - Merged datasets saved in the same folder.
  - Provides summary statistics for the merged data.

### `step3_SMOTE.py`

- **Purpose**: Applies SMOTE (Synthetic Minority Oversampling Technique) to balance the datasets.
- **Input**: Grouped files from the previous step.
- **Process**:
  - Loads the grouped datasets and checks for the `defects` target column.
  - Balances the dataset by oversampling the minority class.
  - Prints the cl
