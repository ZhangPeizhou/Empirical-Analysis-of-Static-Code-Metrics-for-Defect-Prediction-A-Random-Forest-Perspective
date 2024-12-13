import os
import random
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Paths for dataset and results
dataset_path = "Dataset/CSV only"
result_path = "Result"
os.makedirs(result_path, exist_ok=True)  # Ensure the Result folder exists

group_files = ["group_1_SMOTE.csv", "group_2_SMOTE.csv", "group_3_SMOTE.csv", "group_4_SMOTE.csv", "group_5_SMOTE.csv"]

# Function to randomly generate hyperparameters
def random_hyperparameters():
    return {
        "n_estimators": random.randint(50, 300),  # Number of trees
        "max_depth": random.choice([None, 10, 20, 30, 50]),  # Maximum depth of trees
        "min_samples_split": random.choice([2, 5, 10]),  # Minimum samples to split a node
        "min_samples_leaf": random.choice([1, 2, 4]),  # Minimum samples in a leaf
        "bootstrap": random.choice([True, False])  # Whether to use bootstrap sampling
    }

# Function to evaluate a Random Forest model with given hyperparameters
def evaluate_model(X_train, X_test, y_train, y_test, hyperparameters):
    model = RandomForestClassifier(random_state=42, **hyperparameters)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy, model

# Hill Climbing optimization
def hill_climbing(X_train, X_test, y_train, y_test, iterations=100):
    # Start with a random set of hyperparameters
    best_hyperparameters = random_hyperparameters()
    best_accuracy, best_model = evaluate_model(X_train, X_test, y_train, y_test, best_hyperparameters)

    print(f"Initial hyperparameters: {best_hyperparameters}, Initial accuracy: {best_accuracy:.4f}")

    # Perform hill climbing
    for iteration in range(1, iterations + 1):
        # Slightly modify the current best hyperparameters
        new_hyperparameters = best_hyperparameters.copy()
        parameter_to_modify = random.choice(list(new_hyperparameters.keys()))  # Randomly choose a parameter to modify

        if parameter_to_modify == "n_estimators":
            new_hyperparameters["n_estimators"] = random.randint(50, 300)
        elif parameter_to_modify == "max_depth":
            new_hyperparameters["max_depth"] = random.choice([None, 10, 20, 30, 50])
        elif parameter_to_modify == "min_samples_split":
            new_hyperparameters["min_samples_split"] = random.choice([2, 5, 10])
        elif parameter_to_modify == "min_samples_leaf":
            new_hyperparameters["min_samples_leaf"] = random.choice([1, 2, 4])
        elif parameter_to_modify == "bootstrap":
            new_hyperparameters["bootstrap"] = not new_hyperparameters["bootstrap"]

        # Evaluate the new hyperparameters
        new_accuracy, new_model = evaluate_model(X_train, X_test, y_train, y_test, new_hyperparameters)

        # If the new hyperparameters perform better, update the best ones
        if new_accuracy > best_accuracy:
            best_hyperparameters = new_hyperparameters
            best_accuracy = new_accuracy
            best_model = new_model
            print(f"Iteration {iteration}: New best hyperparameters: {best_hyperparameters}, Accuracy: {best_accuracy:.4f}")
        else:
            print(f"Iteration {iteration}: No improvement, Current best accuracy: {best_accuracy:.4f}")

    return best_hyperparameters, best_accuracy, best_model

# File to store results
output_file = os.path.join(result_path, "hill_climbing_results.txt")

# Redirect print outputs to a file while keeping progress visible
with open(output_file, "w") as f:
    for group_file in group_files:
        file_path = os.path.join(dataset_path, group_file)
        
        # Check if the file exists
        if not os.path.exists(file_path):
            print(f"File {group_file} not found, skipping.", file=f)
            print(f"File {group_file} not found, skipping.")  # Print to console
            continue

        # Read the dataset
        data = pd.read_csv(file_path)
        
        # Ensure 'defects' column exists
        if 'defects' not in data.columns:
            print(f"'defects' column not found in {group_file}, skipping.", file=f)
            print(f"'defects' column not found in {group_file}, skipping.")  # Print to console
            continue

        # Separate features (X) and target (y), ignoring 'id'
        if 'id' in data.columns:
            X = data.drop(columns=["defects", "id"])
        else:
            X = data.drop(columns=["defects"])
        y = data["defects"]

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Perform hill climbing to optimize hyperparameters
        print(f"Optimizing hyperparameters for {group_file}...")
        print(f"Optimizing hyperparameters for {group_file}...", file=f)
        best_hyperparameters, best_accuracy, best_model = hill_climbing(X_train, X_test, y_train, y_test)

        # Output the results
        print(f"\nBest hyperparameters for {group_file}:")
        print(f"\nBest hyperparameters for {group_file}:", file=f)
        print(best_hyperparameters)
        print(best_hyperparameters, file=f)
        print(f"Best accuracy for {group_file}: {best_accuracy:.2f}")
        print(f"Best accuracy for {group_file}: {best_accuracy:.2f}", file=f)

        # Evaluate the best model
        y_pred = best_model.predict(X_test)
        print("Classification Report:")
        print("Classification Report:", file=f)
        print(classification_report(y_test, y_pred))
        print(classification_report(y_test, y_pred), file=f)
        print("Confusion Matrix:")
        print("Confusion Matrix:", file=f)
        print(confusion_matrix(y_test, y_pred))
        print(confusion_matrix(y_test, y_pred), file=f)
        print("=" * 40)
        print("=" * 40, file=f)

print(f"Hill climbing optimization results saved to {output_file}")
