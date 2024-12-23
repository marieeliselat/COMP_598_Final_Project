import pandas as pd
from collections import defaultdict, Counter
import numpy as np

# Load the dataset
file_path = "trec-rf10-data.txt"
data = pd.read_csv(file_path, sep="\t", header=0)

# Filter only tasks with gold labels
gold_data = data[data['gold'] != -1]

# Step 1: Compute Task-Specific Worker Accuracy
def compute_task_specific_accuracy(gold_data):
    """
    Compute worker accuracy per task type: highly relevant, relevant, non-relevant, broken links.
    Returns a dictionary of dictionaries with worker accuracies for each label type.
    """
    task_accuracies = defaultdict(lambda: defaultdict(list))
    
    for _, row in gold_data.iterrows():
        worker = row['workerID']
        gold_label = row['gold']
        is_correct = int(row['label'] == gold_label)
        task_accuracies[gold_label][worker].append(is_correct)
    
    worker_task_accuracy = defaultdict(lambda: defaultdict(float))
    for gold_label, workers in task_accuracies.items():
        for worker, correct_list in workers.items():
            worker_task_accuracy[gold_label][worker] = np.mean(correct_list)
    
    return worker_task_accuracy

# Step 2: Task-Specific Weighted Voting Function
def weighted_majority_vote(group, worker_task_accuracy):
    """Compute weighted majority vote for a group of worker labels using task-specific weights."""
    weighted_votes = defaultdict(float)
    gold_label = group['gold'].iloc[0]  # Gold label for reference
    for _, row in group.iterrows():
        worker = row['workerID']
        label = row['label']
        weight = worker_task_accuracy[gold_label].get(worker, 0)  # Task-specific weight
        weighted_votes[label] += weight
    # Choose the label with the highest weighted score
    if weighted_votes:
        max_weight = max(weighted_votes.values())
        top_labels = [label for label, weight in weighted_votes.items() if weight == max_weight]
        return np.random.choice(top_labels)  # Resolve ties randomly
    else:
        return np.random.choice([0, 1, 2, -2])  # Random fallback if no votes

# Step 3: Apply Task-Specific Weighted Voting
worker_task_accuracy = compute_task_specific_accuracy(gold_data)
grouped_tasks = gold_data.groupby(['topicID', 'docID'])
weighted_results = []

for (topicID, docID), group in grouped_tasks:
    weighted_label = weighted_majority_vote(group, worker_task_accuracy)
    gold_label = group['gold'].iloc[0]
    weighted_results.append({'topicID': topicID, 'docID': docID, 
                             'weighted_label': weighted_label, 'gold': gold_label})

# Convert results to a DataFrame
results_df = pd.DataFrame(weighted_results)

# Step 4: Evaluate Accuracy
weighted_accuracy = (results_df['weighted_label'] == results_df['gold']).mean()
print(f"Accuracy of Task-Specific Weighted Voting: {weighted_accuracy:.2%}")

# Step 5: Generate Confusion Matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(results_df['gold'], results_df['weighted_label'], labels=[-2, 0, 1, 2])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[-2, 0, 1, 2])
disp.plot(cmap='Blues')
disp.ax_.set_title("Confusion Matrix: Task-Specific Weighted Voting")
disp.figure_.savefig("task_specific_weighted_voting_confusion_matrix.png")
