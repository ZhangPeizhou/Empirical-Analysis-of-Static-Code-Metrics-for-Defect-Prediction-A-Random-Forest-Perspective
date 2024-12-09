import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from contextlib import redirect_stdout

# Path to dataset directory
dataset_path = "Dataset/CSV only"
group_files = ["group_1_SMOTE.csv", "group_2_SMOTE.csv", "group_3_SMOTE.csv", "group_4_SMOTE.csv", "group_5_SMOTE.csv"]

# Dictionary to store results for comparison
results = {}

# File to save all results
output_file = "result.txt"

# Redirect print output to a file
with open(output_file, "w") as f:
    with redirect_stdout(f):
        # Process each group separately
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
            
            # Split the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Initialize the Random Forest model
            rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
            
            # Train the model
            rf_model.fit(X_train, y_train)
            
            # Predict on the test set
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

            # Plot feature importance (save and close)
            plt.figure(figsize=(10, 6))
            plt.bar(importance_df['Feature'], importance_df['Importance'])
            plt.xticks(rotation=90)
            plt.title(f"Feature Importance for {group_file}")
            # Save plot to a file
            plot_file = group_file.replace(".csv", "_importance.png")
            plt.savefig(plot_file)
            print(f"Feature importance plot saved as {plot_file}")
            plt.close()  # Close the plot to free memory

        # Compare results across groups
        results_df = pd.DataFrame(results).T
        print("\nComparison of Results Across Groups:")
        print(results_df)

        # Plot accuracy comparison (save and close)
        plt.figure(figsize=(10, 6))
        results_df['accuracy'].plot(kind='bar', title='Accuracy Comparison Across Groups')
        plt.ylabel('Accuracy')
        plt.xlabel('Groups')
        plt.xticks(rotation=45)
        # Save accuracy plot to a file
        accuracy_plot_file = "accuracy_comparison.png"
        plt.savefig(accuracy_plot_file)
        print(f"Accuracy comparison plot saved as {accuracy_plot_file}")
        plt.close()  # Close the plot
