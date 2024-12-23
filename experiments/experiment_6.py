import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# Load the dataset
file_path = "trec-rf10-data.txt"
data = pd.read_csv(file_path, sep="\t", header=0)

# Step 1: Feature Engineering
def compute_worker_accuracy(gold_data):
    """Compute worker accuracy based on gold-labeled tasks."""
    worker_accuracy = {}
    grouped = gold_data.groupby('workerID')
    for worker, group in grouped:
        correct = (group['label'] == group['gold']).sum()
        total = len(group)
        accuracy = correct / total if total > 0 else 0
        worker_accuracy[worker] = accuracy
    return worker_accuracy

# Filter only tasks with gold labels
gold_data = data[data['gold'] != -1]

# Compute worker accuracy
worker_accuracy = compute_worker_accuracy(gold_data)

# Add features to the dataset
data['worker_accuracy'] = data['workerID'].map(worker_accuracy)
data['worker_accuracy'].fillna(0.0, inplace=True)  # Default accuracy for unseen workers
data['worker_experience'] = data['workerID'].map(data['workerID'].value_counts())

# Group data by task to create aggregated features
grouped_tasks = data.groupby(['topicID', 'docID'])
task_features = []

for (topicID, docID), group in grouped_tasks:
    # Extract features for each task
    feature_row = {
        'topicID': topicID,
        'docID': docID,
        'num_workers': len(group),
        'mean_worker_accuracy': group['worker_accuracy'].mean(),
        'max_worker_accuracy': group['worker_accuracy'].max(),
        'worker_experience_mean': group['worker_experience'].mean(),
        'worker_experience_max': group['worker_experience'].max(),
        'label_0_count': (group['label'] == 0).sum(),
        'label_1_count': (group['label'] == 1).sum(),
        'label_2_count': (group['label'] == 2).sum(),
        'label_-2_count': (group['label'] == -2).sum(),
        'label_entropy': -sum((group['label'].value_counts(normalize=True) * 
                               np.log2(group['label'].value_counts(normalize=True) + 1e-6))),
        'gold_label': group['gold'].iloc[0]  # True label (gold)
    }
    task_features.append(feature_row)

# Convert to DataFrame
task_features_df = pd.DataFrame(task_features)
# Step 2: Map labels to positive integers
label_mapping = {-2: 0, -1: 1, 0: 2, 1: 3, 2: 4}  # Map labels
inverse_label_mapping = {v: k for k, v in label_mapping.items()}  # Reverse mapping for evaluation

# Apply label mapping
task_features_df['gold_label'] = task_features_df['gold_label'].map(label_mapping)

# Prepare Train-Test Data
X = task_features_df.drop(columns=['topicID', 'docID', 'gold_label'])  # Features
y = task_features_df['gold_label']  # Mapped gold labels

# Train-test split (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Train XGBoost Classifier with Hyperparameter Tuning
print("Training XGBoost Classifier with Hyperparameter Tuning...")

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'colsample_bytree': [0.7, 1.0]
}

xgb = XGBClassifier(random_state=42)
grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=3, scoring='accuracy', verbose=1)
grid_search.fit(X_train, y_train)

# Best model from GridSearch
best_xgb = grid_search.best_estimator_
print(f"Best Parameters: {grid_search.best_params_}")

# Step 4: Make Predictions and Convert Back to Original Labels
y_pred = best_xgb.predict(X_test)
y_test_original = [inverse_label_mapping[label] for label in y_test]
y_pred_original = [inverse_label_mapping[label] for label in y_pred]

# Calculate accuracy
accuracy = accuracy_score(y_test_original, y_pred_original)
print(f"Accuracy of ML-Based Aggregation (XGBoost): {accuracy:.2%}")

# Confusion Matrix
cm = confusion_matrix(y_test_original, y_pred_original, labels=[-2, 0, 1, 2])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[-2, 0, 1, 2])
disp.plot(cmap='Blues')
disp.ax_.set_title("Confusion Matrix: Improved ML-Based Label Aggregation")
disp.figure_.savefig("improved_ml_label_aggregation_confusion_matrix.png")
