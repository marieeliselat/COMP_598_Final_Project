import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
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
        'gold_label': group['gold'].iloc[0]  # True label (gold)
    }
    task_features.append(feature_row)

# Convert to DataFrame
task_features_df = pd.DataFrame(task_features)

# Step 2: Prepare Train-Test Data
X = task_features_df.drop(columns=['topicID', 'docID', 'gold_label'])  # Features
y = task_features_df['gold_label']  # Gold labels

# Train-test split (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Train Machine Learning Model
print("Training Random Forest Classifier...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Step 4: Make Predictions and Evaluate
y_pred = rf_model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of ML-Based Aggregation (Random Forest): {accuracy:.2%}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=[-2, 0, 1, 2])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[-2, 0, 1, 2])
disp.plot(cmap='Blues')
disp.ax_.set_title("Confusion Matrix: ML-Based Label Aggregation")
disp.figure_.savefig("ml_label_aggregation_confusion_matrix.png")
