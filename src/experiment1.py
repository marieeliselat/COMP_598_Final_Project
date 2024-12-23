import pandas as pd
from collections import Counter
import numpy as np

# Load the dataset
file_path = "trec-rf10-data.txt"
data = pd.read_csv(file_path, sep="\t", header=0)

# Set a random seed for reproducibility
np.random.seed(42)

# Filter tasks with gold labels
gold_data = data[data['gold'] != -1]

# Group the data by topicID and docID to aggregate worker labels
grouped = gold_data.groupby(['topicID', 'docID'])

# Function to apply majority voting for each group
def majority_vote(labels):
    label_counts = Counter(labels)
    # Handle ties by selecting a random label among the most frequent
    max_count = max(label_counts.values())
    top_labels = [label for label, count in label_counts.items() if count == max_count]
    return np.random.choice(top_labels)

# Apply majority vote to each task
majority_labels = grouped['label'].apply(majority_vote).reset_index()
majority_labels.columns = ['topicID', 'docID', 'majority_label']

# Merge majority vote results with the gold labels for comparison
gold_labels = gold_data[['topicID', 'docID', 'gold']].drop_duplicates()
results = pd.merge(majority_labels, gold_labels, on=['topicID', 'docID'])

# Calculate accuracy: Compare majority vote labels to gold labels
accuracy = (results['majority_label'] == results['gold']).mean()
print(f"Accuracy of Majority Voting: {accuracy:.2%}")

# Optional: Display a confusion matrix for more detailed analysis
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(results['gold'], results['majority_label'], labels=[-2, 0, 1, 2])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[-2, 0, 1, 2])
disp.plot(cmap='Blues')
disp.ax_.set_title("Confusion Matrix: Majority Voting vs Gold Labels")
disp.figure_.savefig("majority_vote_confusion_matrix.png")
