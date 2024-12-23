import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set plot style for consistency
sns.set(style="whitegrid")

# 1.Load Data 
# File path to the dataset
file_path = "trec-rf10-data.txt"

# Data has columns that are tab-separated with a header
data = pd.read_csv(file_path, sep="\t", header=0)

# Print top to see what's going on 
print("First 5 rows of the dataset:")
print(data.head())

# 2. Basic Info about the Data 
# Overview of the dataset
print("\nDataset Overview:")
print(data.info())

# Descriptive stats
print("\nDescriptive statistics:")
print(data.describe())

# 3. Data Exploration 
# Unique counts for workers and tasks
num_workers = data['workerID'].nunique()
num_tasks = data[['topicID', 'docID']].drop_duplicates().shape[0]
num_judgments = data.shape[0]
print(f"\nNumber of Workers: {num_workers}")
print(f"Number of Unique Tasks: {num_tasks}")
print(f"Total Number of Judgments: {num_judgments}")

# Distribution of Gold Labels
gold_counts = data['gold'].value_counts()
print("\nDistribution of Gold Labels:")
print(gold_counts)

# Distribution of Labels by Workers
label_counts = data['label'].value_counts()
print("\nDistribution of Worker Labels:")
print(label_counts)

# 4. Worker Participation Stats
# Number of judgments per worker
worker_counts = data['workerID'].value_counts()
print("\nJudgments per Worker (Top 5 Workers):")
print(worker_counts.head(5))

# Plot: Histogram of worker participation
plt.figure(figsize=(8, 5))
sns.histplot(worker_counts, bins=30, kde=False, color="skyblue")
plt.title("Worker Participation: Number of Judgments per Worker")
plt.xlabel("Number of Judgments")
plt.ylabel("Number of Workers")
plt.tight_layout()
plt.savefig("worker_participation.png")  # Save the figure
plt.show()

# 5. Task Coverage Statistics
# Number of workers per task
tasks = data.groupby(['topicID', 'docID']).size()
print("\nNumber of Workers per Task (Top 5 Tasks):")
print(tasks.head(5))

# Plot: Histogram of workers per task 
plt.figure(figsize=(8, 5))
bins = range(0, tasks.max() + 2)  
sns.histplot(tasks, bins=bins, kde=False, color="orange")
plt.xticks(np.arange(0, tasks.max() + 1, step=1))
plt.xlim(0, tasks.max() + 1)  # Limit x-axis to avoid extra space
plt.title("Task Coverage: Number of Workers per Task")
plt.xlabel("Number of Workers")
plt.ylabel("Number of Tasks")
plt.tight_layout()
plt.savefig("task_coverage.png")  # Save the figure
plt.show()

# 6. Gold Label Analysis 
# Proportion of tasks with gold labels
gold_present = data[data['gold'] != -1]
gold_proportion = len(gold_present) / len(data)
print(f"\nProportion of Tasks with Gold Labels: {gold_proportion:.2%}")

# Gold label Figure
plt.figure(figsize=(8, 5))
sns.countplot(x='gold', data=gold_present, palette="viridis")
plt.title("Distribution of Gold Labels")
plt.xlabel("Gold Label")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("gold_label_distribution.png")  # Save the figure
plt.show()

# 7. Worker Labels vs. Gold-Labeled Tasks
# Filter tasks with gold labels
gold_labeled_data = data[data['gold'] != -1]

# Plot the comparison of worker-provided labels to gold labels
plt.figure(figsize=(8, 5))
sns.countplot(
    x='label',
    hue='gold',
    data=gold_labeled_data,
    palette="muted"
)
plt.title("Worker Labels Compared to Gold Labels")
plt.xlabel("Worker-Provided Labels")
plt.ylabel("Count")
plt.legend(title="Gold Labels")
plt.tight_layout()
plt.savefig("worker_vs_gold_labels.png")  #Save the figure
plt.show()

# 8. Summary Stats 
# Aggregate stats per worker
worker_summary = data.groupby('workerID').agg(
    num_tasks=('workerID', 'count'),
    num_gold_tasks=('gold', lambda x: (x != -1).sum())
).reset_index()

print("\nWorker Summary Statistics (Top 5 Rows):")
print(worker_summary.head(5))

# Save to csv just in case 
worker_summary.to_csv("worker_summary_statistics.csv", index=False)
print("\nWorker summary statistics saved to 'worker_summary_statistics.csv'.")
