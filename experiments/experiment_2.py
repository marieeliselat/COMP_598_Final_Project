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
    # Group by workerID
    grouped = gold_data.groupby('workerID')
    for worker, group in grouped:
        # Accuracy is the proportion of correct judgments
        correct = (group['label'] == group['gold']).sum()
        total = len(group)
        accuracy = correct / total if total > 0 else 0
        worker_accuracy[worker] = accuracy
    return worker_accuracy

# Compute worker accuracies
worker_accuracy = compute_worker_accuracy(gold_data)
print("Top 5 Workers by Accuracy:")
print(sorted(worker_accuracy.items(), key=lambda x: x[1], reverse=True)[:5])

# Step 2: Weighted Voting Function
def weighted_majority_vote(group, worker_accuracy):
    """Compute weighted majority vote for a group of worker labels."""
    weighted_votes = defaultdict(float)
    for _, row in group.iterrows():
        worker = row['workerID']
        label = row['label']
        weight = worker_accuracy.get(worker, 0)  # Default weight is 0 if worker is not in accuracy list
        weighted_votes[label] += weight
    # Choose the label with the highest weighted score
    max_weight = max(weighted_votes.values())
    top_labels = [label for label, weight in weighted_votes.items() if weight == max_weight]
    return np.random.choice(top_labels)  # Resolve ties randomly

# Step 3: Apply Weighted Voting to All Tasks with Gold Labels
grouped_tasks = gold_data.groupby(['topicID', 'docID'])
weighted_results = []

for (topicID, docID), group in grouped_tasks:
    weighted_label = weighted_majority_vote(group, worker_accuracy)
    gold_label = group['gold'].iloc[0]  # Get the gold label for comparison
    weighted_results.append({'topicID': topicID, 'docID': docID, 
                             'weighted_label': weighted_label, 'gold': gold_label})

# Convert results to a DataFrame
results_df = pd.DataFrame(weighted_results)

# Step 4: Evaluate Weighted Voting Accuracy
weighted_accuracy = (results_df['weighted_label'] == results_df['gold']).mean()
print(f"Accuracy of Weighted Voting: {weighted_accuracy:.2%}")

# Step 5: Generate a Confusion Matrix for Weighted Voting
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(results_df['gold'], results_df['weighted_label'], labels=[-2, 0, 1, 2])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[-2, 0, 1, 2])
disp.plot(cmap='Blues')
disp.ax_.set_title("Confusion Matrix: Weighted Voting vs Gold Labels")
disp.figure_.savefig("weighted_voting_confusion_matrix.png")
