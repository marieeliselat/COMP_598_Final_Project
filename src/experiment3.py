import pandas as pd
from collections import defaultdict, Counter
import numpy as np

# Load the dataset
file_path = "trec-rf10-data.txt"
data = pd.read_csv(file_path, sep="\t", header=0)

# Filter only tasks with gold labels
gold_data = data[data['gold'] != -1]

# Step 1: Compute Worker Accuracy on Gold-Labeled Tasks
def compute_worker_accuracy(gold_data):
    worker_accuracy = {}
    grouped = gold_data.groupby('workerID')
    for worker, group in grouped:
        correct = (group['label'] == group['gold']).sum()
        total = len(group)
        accuracy = correct / total if total > 0 else 0
        worker_accuracy[worker] = accuracy
    return worker_accuracy

# Step 2: Filter Workers by Accuracy Threshold
def filter_workers_by_accuracy(worker_accuracy, threshold=0.6):
    """Filter out workers below a certain accuracy threshold."""
    filtered_workers = {worker: acc for worker, acc in worker_accuracy.items() if acc >= threshold}
    return filtered_workers

# Step 3: Weighted Voting Function Using Accuracy as Weight
def weighted_majority_vote(group, worker_weights):
    """Compute weighted majority vote for a group of worker labels."""
    weighted_votes = defaultdict(float)
    for _, row in group.iterrows():
        worker = row['workerID']
        label = row['label']
        weight = worker_weights.get(worker, 0)  # Default weight is 0 for excluded workers
        weighted_votes[label] += weight ** 2  # Square the weight to amplify accuracy importance
    # Choose the label with the highest weighted score
    if weighted_votes:
        max_weight = max(weighted_votes.values())
        top_labels = [label for label, weight in weighted_votes.items() if weight == max_weight]
        return np.random.choice(top_labels)  # Resolve ties randomly
    else:
        return np.random.choice([0, 1, 2, -2])  # Fallback random label if no workers pass the filter

# Step 4: Apply Weighted Voting with Filtered Workers
# Compute worker accuracy
worker_accuracy = compute_worker_accuracy(gold_data)

# Filter workers by accuracy threshold
accuracy_threshold = 0.6  # Set a threshold (e.g., 60% accuracy)
filtered_workers = filter_workers_by_accuracy(worker_accuracy, accuracy_threshold)

print(f"Number of workers retained after filtering: {len(filtered_workers)}")

# Apply weighted voting
grouped_tasks = gold_data.groupby(['topicID', 'docID'])
weighted_results = []

for (topicID, docID), group in grouped_tasks:
    weighted_label = weighted_majority_vote(group, filtered_workers)
    gold_label = group['gold'].iloc[0]
    weighted_results.append({'topicID': topicID, 'docID': docID, 
                             'weighted_label': weighted_label, 'gold': gold_label})

# Convert results to a DataFrame
results_df = pd.DataFrame(weighted_results)

# Step 5: Evaluate Accuracy
weighted_accuracy = (results_df['weighted_label'] == results_df['gold']).mean()
print(f"Accuracy of Improved Weighted Voting with Thresholding: {weighted_accuracy:.2%}")

# Step 6: Generate a Confusion Matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(results_df['gold'], results_df['weighted_label'], labels=[-2, 0, 1, 2])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[-2, 0, 1, 2])
disp.plot(cmap='Blues')
disp.ax_.set_title("Confusion Matrix: Improved Weighted Voting with Thresholding")
disp.figure_.savefig("improved_weighted_voting_thresholding_confusion_matrix.png")
