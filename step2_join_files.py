import os
import pandas as pd

dataset_path = "Dataset/CSV only"
grouped_file_columns_path = "Dataset/grouped_file_columns.csv"

# Read grouped file columns CSV
group_df = pd.read_csv(grouped_file_columns_path)

# Dictionary to store row counts for each file and group
file_row_counts = {}
group_row_counts = {}

# Iterate over each group
for _, row in group_df.iterrows():
    group_name = row["Group"]
    file_list = row["Files"].split(", ")
    
    # Initialize an empty DataFrame for combined data
    combined_data = pd.DataFrame()
    
    for file_name in file_list:
        file_path = os.path.join(dataset_path, file_name)
        
        # Check if the file exists
        if not os.path.exists(file_path):
            print(f"File {file_name} not found, skipping.")
            continue
        
        # Read the CSV file
        data = pd.read_csv(file_path)
        file_row_counts[file_name] = len(data)
        
        # Append the data to the combined DataFrame
        combined_data = pd.concat([combined_data, data], ignore_index=True)
    
    # Save the combined data for the group
    group_file_name = f"{group_name.lower().replace(' ', '_')}.csv"
    group_file_path = os.path.join(dataset_path, group_file_name)
    combined_data.to_csv(group_file_path, index=False)
    
    # Store the row count for the group
    group_row_counts[group_file_name] = len(combined_data)

# Print the row counts for each group
for group_index, (group_name, row) in enumerate(group_df.iterrows(), start=1):
    print(f"Group {group_index}:")
    
    # Print row counts for each file in the group
    file_list = row["Files"].split(", ")
    for file_name in file_list:
        if file_name in file_row_counts:
            print(f"  {file_name}: {file_row_counts[file_name]} rows")
    
    # Print row count for the combined group file
    group_file_name = f"{row['Group'].lower().replace(' ', '_')}.csv"
    if group_file_name in group_row_counts:
        print(f"  Combined {group_file_name}: {group_row_counts[group_file_name]} rows")
    
    # Separator between groups
    print("-" * 40)
