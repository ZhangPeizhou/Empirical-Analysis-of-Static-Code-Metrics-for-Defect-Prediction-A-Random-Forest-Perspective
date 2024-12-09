import os
import pandas as pd
from imblearn.over_sampling import SMOTE

# Path to dataset directory
dataset_path = "Dataset/CSV only"
group_files = ["group_1.csv", "group_2.csv", "group_3.csv", "group_4.csv", "group_5.csv"]

# Function to print the distribution of the target column
def print_distribution(data, label_column, stage):
    true_count = data[label_column].sum()  # Count of True
    false_count = len(data) - true_count  # Count of False
    total = len(data)
    true_percentage = (true_count / total) * 100 if total > 0 else 0
    false_percentage = (false_count / total) * 100 if total > 0 else 0
    print(f"{stage} distribution:")
    print(f"  TRUE: {true_count} ({true_percentage:.2f}%)")
    print(f"  FALSE: {false_count} ({false_percentage:.2f}%)")
    print("-" * 40)

# Process each group file
for group_file in group_files:
    file_path = os.path.join(dataset_path, group_file)
    
    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"File {group_file} not found, skipping.")
        continue
    
    # Read the CSV file
    data = pd.read_csv(file_path)
    
    # Ensure 'defects' column exists
    if 'defects' not in data.columns:
        print(f"'defects' column not found in {group_file}, skipping.")
        continue

    # Separate features (X) and target (y), ignoring 'id'
    if 'id' in data.columns:
        X = data.drop(columns=["defects", "id"])
    else:
        X = data.drop(columns=["defects"])
    y = data["defects"]

    # Print distribution before SMOTE
    print(f"\nProcessing {group_file}...")
    print_distribution(data, 'defects', "Before SMOTE")
    
    # Apply SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # Combine resampled data
    resampled_data = pd.concat([X_resampled, pd.Series(y_resampled, name="defects")], axis=1)

    # Add back the 'id' column with new unique values
    resampled_data["id"] = range(1, len(resampled_data) + 1)

    # Print distribution after SMOTE
    print_distribution(resampled_data, 'defects', "After SMOTE")
    
    # Save the balanced dataset
    smote_file_name = group_file.replace(".csv", "_SMOTE.csv")
    smote_file_path = os.path.join(dataset_path, smote_file_name)
    resampled_data.to_csv(smote_file_path, index=False)

    print(f"SMOTE applied and saved to {smote_file_name}")
