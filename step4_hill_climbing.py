import os
import random
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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

# Function to evaluate the average accuracy across all datasets
def evaluate_average_accuracy(hyperparameters, datasets):
    accuracies = []
    for data in datasets:
        X_train, X_test, y_train, y_test = data
        model = RandomForestClassifier(random_state=42, **hyperparameters)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracies.append(accuracy_score(y_test, y_pred))
    return sum(accuracies) / len(accuracies)  # Return average accuracy

# Hill Climbing optimization to find unified hyperparameters
def hill_climbing(datasets, iterations=100):
    # Start with a random set of hyperparameters
    best_hyperparameters = random_hyperparameters()
    best_accuracy = evaluate_average_accuracy(best_hyperparameters, datasets)

    print(f"Initial hyperparameters: {best_hyperparameters}, Initial average accuracy: {best_accuracy:.4f}")

    # Track progress for plotting
    accuracy_progress = [best_accuracy]
    iteration_progress = [0]

    # Perform hill climbing
    for iteration in range(1, iterations + 1):
        # Slightly modify the current best hyperparameters
        new_hyperparameters = best_hyperparameters.copy()
        parameter_to_modify = random.choice(list(new_hyperparameters.keys()))

        if parameter_to_modify == "n_estimators":
            new_hyperparameters["n_estimators"] = random.randint(50, 100)
        elif parameter_to_modify == "max_depth":
            new_hyperparameters["max_depth"] = random.choice([None, 10, 20, 30])
        elif parameter_to_modify == "min_samples_split":
            new_hyperparameters["min_samples_split"] = random.choice([2, 5, 10])
        elif parameter_to_modify == "min_samples_leaf":
            new_hyperparameters["min_samples_leaf"] = random.choice([1, 2, 4])
        elif parameter_to_modify == "bootstrap":
            new_hyperparameters["bootstrap"] = not new_hyperparameters["bootstrap"]

        # Evaluate the new hyperparameters
        new_accuracy = evaluate_average_accuracy(new_hyperparameters, datasets)

        # Print the iteration progress
        print(f"Iteration {iteration}:")
        print(f"  Hyperparameters: {new_hyperparameters}")
        print(f"  Average accuracy: {new_accuracy:.4f}")

        # If the new hyperparameters improve the average accuracy, update the best ones
        if new_accuracy > best_accuracy:
            best_hyperparameters = new_hyperparameters
            best_accuracy = new_accuracy
            print(f"  Improved! New best accuracy: {best_accuracy:.4f}")
        else:
            print("  No improvement.")

        # Track progress for plotting
        iteration_progress.append(iteration)
        accuracy_progress.append(best_accuracy)

    return best_hyperparameters, best_accuracy, iteration_progress, accuracy_progress

# Load and prepare all datasets
print("Loading and preparing datasets...")
datasets = []
for group_file in group_files:
    file_path = os.path.join(dataset_path, group_file)
    if os.path.exists(file_path):
        data = pd.read_csv(file_path)
        if 'id' in data.columns:
            X = data.drop(columns=["defects", "id"])
        else:
            X = data.drop(columns=["defects"])
        y = data["defects"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        datasets.append((X_train, X_test, y_train, y_test))
    else:
        print(f"File {group_file} not found, skipping.")

# Perform hill climbing to find unified hyperparameters
print("Starting Hill Climbing to find unified hyperparameters...")
best_params, best_avg_accuracy, iterations, accuracies = hill_climbing(datasets)

# Save the results
output_file = os.path.join(result_path, "unified_hill_climbing_results.txt")
with open(output_file, "w") as f:
    f.write(f"Best Unified Hyperparameters: {best_params}\n")
    f.write(f"Best Average Accuracy: {best_avg_accuracy:.4f}\n")

# Plot accuracy progress
plt.figure(figsize=(10, 6))
plt.plot(iterations, accuracies, marker='o', linestyle='-', color='b')
plt.title("Hill Climbing Accuracy Progress (Unified Parameters)")
plt.xlabel("Iteration")
plt.ylabel("Average Accuracy")
plt.grid()
plt.tight_layout()
plot_file = os.path.join(result_path, "unified_accuracy_progress.png")
plt.savefig(plot_file)
plt.close()

print(f"Best Unified Hyperparameters: {best_params}")
print(f"Best Average Accuracy: {best_avg_accuracy:.4f}")
print(f"Results saved to {output_file}")
print(f"Accuracy progress plot saved to {plot_file}")
