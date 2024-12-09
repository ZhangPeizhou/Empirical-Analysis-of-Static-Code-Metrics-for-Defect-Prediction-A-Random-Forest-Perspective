import os
import pandas as pd
from collections import defaultdict

# Path to the directory containing the "CSV only" folder
dataset_directory = "./Dataset/CSV only"

# Dictionary to group files by their column content (ignoring order and 'id')
column_groups = defaultdict(list)

for file_name in os.listdir(dataset_directory):
    if file_name.endswith(".csv"):  # Process only .csv files
        file_path = os.path.join(dataset_directory, file_name)
        df = pd.read_csv(file_path)
        
        # Get sorted set of column names, excluding 'id'
        columns = frozenset([col for col in df.columns if col != 'id'])
        
        # Use the sorted set of column names as a key
        column_groups[columns].append(file_name)

# Prepare data for the CSV file
grouped_data = []
for idx, (columns, files) in enumerate(column_groups.items(), 1):
    grouped_data.append({
        "Group": f"Group {idx}",
        "Files": ", ".join(files),
        "Column Titles": ", ".join(sorted(columns))  # Sort columns for better readability
    })

# Create a DataFrame for grouped data
output_df = pd.DataFrame(grouped_data)

# Save the grouped information to a CSV file
output_path = "./Dataset/grouped_file_columns.csv"
output_df.to_csv(output_path, index=False)

# Display the grouped files and their columns
print("Grouped CSV Files by Column Structure (ignoring order):")
for idx, (columns, files) in enumerate(column_groups.items(), 1):
    print(f"\nGroup {idx}:")
    print(f"Columns: {sorted(columns)}")  # Sort for display purposes
    print(f"Files: {files}")

print(f"\nGrouped information saved to: {output_path}")
