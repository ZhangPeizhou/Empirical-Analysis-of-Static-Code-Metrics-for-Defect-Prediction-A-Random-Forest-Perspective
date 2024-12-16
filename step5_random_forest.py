import os
import pandas as pd
import ast  # Safe eval for parsing strings to Python objects
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from contextlib import redirect_stdout

# Paths
dataset_path = "Dataset/CSV only"
result_path = "Result"
os.makedirs(result_path, exist_ok=True)  # Ensure the Result folder exists

# File containing the unified best hyperparameters
best_params_file = os.path.join(result_path, "unified_hill_climbing_results.txt")
group_files = ["group_1_SMOTE.csv", "group_2_SMOTE.csv", "group_3_SMOTE.csv", "group_4_SMOTE.csv", "group_5_SMOTE.csv"]

# Load unified best hyperparameters
unified_hyperparameters = {"n_estimators": 100, "random_state": 42}  # Default fallback parameters

if os.path.exists(best_params_file):
    print(f"Loading unified best hyperparameters from {best_params_file}...")
    with open(best_params_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            if "Best Unified Hyperparameters" in line:
                try:
                    hyperparam_str = line.split(":", 1)[1].strip()  # Split and clean up the line
                    unified_hyperparameters = ast.literal_eval(hyperparam_str)  # Safely parse
                    print(f"Unified hyperparameters successfully loaded: {unified_hyperparameters}")
                    break
                except (ValueError, SyntaxError) as e:
                    print(f"Error parsing hyperparameters: {e}")
                    print(f"Invalid line content: {line}")
else:
    print(f"Best parameters file '{best_params_file}' not found. Using default parameters.")

# Dictionary to store results for comparison
results = {}

# Output file to save all results
output_file = os.path.join(result_path, "result.txt")

# Redirect print output to a file while also printing progress
with open(output_file, "w") as f:
    with redirect_stdout(f):
        print(f"Using unified hyperparameters: {unified_hyperparameters}")
        print("=" * 40)

        for group_file in group_files:
            print(f"Processing {group_file}...")
            file_path = os.path.join(dataset_path, group_file)

            # Check if the file exists
            if not os.path.exists(file_path):
                print(f"File {group_file} not found, skipping.")
                continue

            # Read the dataset
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

            # Train Random Forest with unified hyperparameters
            print(f"Training model for {group_file}...")
            rf_model = RandomForestClassifier(**unified_hyperparameters, random_state=42)
            rf_model.fit(X_train, y_train)

            # Predict on the test set
            print(f"Predicting results for {group_file}...")
            y_pred = rf_model.predict(X_test)

            # Evaluate the model
            accuracy = accuracy_score(y_test, y_pred)
            print(f"\nResults for {group_file}:")
            print(f"Accuracy: {accuracy:.4f}")
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

            # Plot feature importance
            print(f"Saving feature importance plot for {group_file}...")
            plt.figure(figsize=(15, 8))
            plt.bar(importance_df['Feature'], importance_df['Importance'])
            plt.xticks(rotation=45, ha='right', fontsize=10)
            plt.title(f"Feature Importance for {group_file}")
            plt.tight_layout()
            plot_file = os.path.join(result_path, group_file.replace(".csv", "_importance.png"))
            plt.savefig(plot_file)
            print(f"Feature importance plot saved as {plot_file}")
            plt.close()

        # Compare results across groups
        print("\nComparing results across groups...")
        results_df = pd.DataFrame(results).T
        print(results_df)

        # Plot accuracy comparison
        print("Saving accuracy comparison plot...")
        plt.figure(figsize=(15, 8))
        results_df['accuracy'].plot(kind='bar', title='Accuracy Comparison Across Groups')
        plt.ylabel('Accuracy')
        plt.xlabel('Groups')
        plt.xticks(rotation=45)
        plt.tight_layout()
        accuracy_plot_file = os.path.join(result_path, "accuracy_comparison.png")
        plt.savefig(accuracy_plot_file)
        print(f"Accuracy comparison plot saved as {accuracy_plot_file}")
        plt.close()
