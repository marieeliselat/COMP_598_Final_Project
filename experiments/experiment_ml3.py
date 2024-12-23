import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils import class_weight

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

# Step 2: Map labels to positive integers
label_mapping = {-2: 0, -1: 1, 0: 2, 1: 3, 2: 4}
inverse_label_mapping = {v: k for k, v in label_mapping.items()}
task_features_df['gold_label'] = task_features_df['gold_label'].map(label_mapping)

# Prepare Train-Test Data
X = task_features_df.drop(columns=['topicID', 'docID', 'gold_label'])
y = task_features_df['gold_label']

# Train-test split (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Calculate Class Weights
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y), y=y)
class_weights_dict = {i: class_weights[i] for i in range(len(class_weights))}
print(f"Class Weights: {class_weights_dict}")

# Step 4: Train XGBoost with Class Weights
print("Training XGBoost Classifier with Class Weights...")
xgb = XGBClassifier(n_estimators=200, max_depth=7, learning_rate=0.01, 
                    colsample_bytree=0.7, random_state=42, scale_pos_weight=class_weights_dict)

xgb.fit(X_train, y_train)

# Step 5: Make Predictions and Convert Back to Original Labels
y_pred = xgb.predict(X_test)
y_test_original = [inverse_label_mapping[label] for label in y_test]
y_pred_original = [inverse_label_mapping[label] for label in y_pred]

# Calculate accuracy
accuracy = accuracy_score(y_test_original, y_pred_original)
print(f"Accuracy of XGBoost with Class Weights: {accuracy:.2%}")

# Confusion Matrix
cm = confusion_matrix(y_test_original, y_pred_original, labels=[-2, 0, 1, 2])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[-2, 0, 1, 2])
disp.plot(cmap='Blues')
disp.ax_.set_title("Confusion Matrix: XGBoost with Class Weights")
disp.figure_.savefig("xgboost_class_weights_confusion_matrix.png")
