import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from contextlib import redirect_stdout

# Path to dataset directory and result directory
dataset_path = "Dataset/CSV only"
result_path = "Result"
os.makedirs(result_path, exist_ok=True)  # Ensure the Result folder exists

# File containing the best hyperparameters (from Hill Climbing or another method)
best_params_file = os.path.join(result_path, "hill_climbing_results.txt")
group_files = ["group_1_SMOTE.csv", "group_2_SMOTE.csv", "group_3_SMOTE.csv", "group_4_SMOTE.csv", "group_5_SMOTE.csv"]

# Dictionary to store the best hyperparameters for each group
best_hyperparameters = {}

# Load the best hyperparameters from the file
if os.path.exists(best_params_file):
    print(f"Loading best hyperparameters from {best_params_file}...")
    with open(best_params_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            if "Best hyperparameters for" in line:
                group_name = line.split("for ")[1].strip().replace(":", "")
            elif line.startswith("{") and "}" in line:
                best_hyperparameters[group_name] = eval(line.strip())  # Convert string to dictionary
else:
    print(f"Best hyperparameters file '{best_params_file}' not found. Using default hyperparameters.")

# Dictionary to store results for comparison
results = {}

# File to save all results
output_file = os.path.join(result_path, "result.txt")

# Redirect print output to a file while also printing progress
with open(output_file, "w") as f:
    with redirect_stdout(f):
        for group_file in group_files:
            print(f"Processing {group_file}...")
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
            
            # Split the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Use the best hyperparameters for the current group if available
            group_name = group_file.replace(".csv", "")
            if group_name in best_hyperparameters:
                print(f"Using best hyperparameters for {group_file}: {best_hyperparameters[group_name]}")
                rf_model = RandomForestClassifier(random_state=42, **best_hyperparameters[group_name])
            else:
                print(f"No best hyperparameters found for {group_file}. Using default settings.")
                rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
            
            # Train the model
            print(f"Training model for {group_file}...")
            rf_model.fit(X_train, y_train)
            
            # Predict on the test set
            print(f"Predicting results for {group_file}...")
            y_pred = rf_model.predict(X_test)
            
            # Evaluate the model
            accuracy = accuracy_score(y_test, y_pred)
            print(f"\nResults for {group_file}:")
            print(f"Accuracy: {accuracy:.2f}")
            print("Classification Report:")
            print(classification_report(y_test, y_pred))
            print("Confusion Matrix:")
            print(confusion_matrix(y_test, y_pred))
            
            # Store results for comparison
            results[group_file] = {
                "accuracy": accuracy,
                "precision": classification_report(y_test, y_pred, output_dict=True)["weighted avg"]["precision"],
                "recall": classification_report(y_test, y_pred, output_dict=True)["weighted avg"]["recall"],
            }

            # Feature importance
            feature_importances = rf_model.feature_importances_
            feature_names = X.columns
            importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
            importance_df = importance_df.sort_values(by='Importance', ascending=False)
            print("\nFeature Importance:")
            print(importance_df)

            # Plot feature importance (save to Result folder)
            print(f"Saving feature importance plot for {group_file}...")
            plt.figure(figsize=(15, 8))  # Increase figure size for better readability
            plt.bar(importance_df['Feature'], importance_df['Importance'])
            plt.xticks(rotation=45, ha='right', fontsize=10)  # Rotate labels and align them to the right
            plt.title(f"Feature Importance for {group_file}")
            plt.tight_layout()  # Adjust layout to prevent clipping of labels
            plot_file = os.path.join(result_path, group_file.replace(".csv", "_importance.png"))
            plt.savefig(plot_file)
            print(f"Feature importance plot saved as {plot_file}")
            plt.close()  # Close the plot to free memory

        # Compare results across groups
        print("\nComparing results across groups...")
        results_df = pd.DataFrame(results).T
        print(results_df)

        # Plot accuracy comparison (save to Result folder)
        print("Saving accuracy comparison plot...")
        plt.figure(figsize=(15, 8))
        results_df['accuracy'].plot(kind='bar', title='Accuracy Comparison Across Groups')
        plt.ylabel('Accuracy')
        plt.xlabel('Groups')
        plt.xticks(rotation=45)
        plt.tight_layout()  # Ensure layout is adjusted
        accuracy_plot_file = os.path.join(result_path, "accuracy_comparison.png")
        plt.savefig(accuracy_plot_file)
        print(f"Accuracy comparison plot saved as {accuracy_plot_file}")
        plt.close()  # Close the plot
