# TREC RF 2010 Crowdsourced Label Aggregation

This repository contains the code, data, and results for the aggregation of crowdsourced relevance judgments from the **TREC 2010 Relevance Feedback Track** dataset. The goal of this project is to evaluate different methods for combining noisy labels from Mechanical Turk workers into high-quality consensus labels. These methods range from simple statistical approaches, such as majority voting, to advanced machine learning techniques like XGBoost.

---

## **Project Overview**

The dataset consists of **98,453 worker judgments** for **20,232 unique tasks**, where workers judged the relevance of documents to search queries. Gold labels provided by NIST are available for **26.6% of the tasks** and serve as the benchmark for evaluating aggregation methods. 

This repository includes code for **seven experiments**, progressing from basic statistical techniques to advanced machine learning approaches. Each experiment is evaluated using accuracy and confusion matrices, with visualizations stored in the `figures/` directory.

---

## **Repository Structure**

```plaintext
trec-rf10-crowd/
├── data/
│   ├── trec-rf10-data.txt          # Raw dataset
│   └── trec-rf10-readme.txt        # Dataset README
├── figures/
│   ├── worker_participation.png  # Histogram of worker participation
│   ├── task_coverage.png  # Task Coverage Statistics
│   ├── gold_label_distribution.png  # Gold Label Analysis
│   ├── worker_vs_gold_labels.png  # Worker Label Analysis for Gold-Labeled Tasks
│   ├── majority_vote_confusion_matrix.png  # Confusion matrix for Experiment 1
│   ├── weighted_voting_confusion_matrix.png  # Confusion matrix for Experiment 2
│   ├── improved_weighted_voting_thresholding_confusion_matrix.png  # Confusion matrix for Experiment 3
│   ├── task_specific_weighted_voting_confusion_matrix.png  # Confusion matrix for Experiment 4
│   ├── ml_label_aggregation_confusion_matrix.png  # Confusion matrix for Experiment 5
│   ├── improved_ml_label_aggregation_confusion_matrix.png  # Confusion matrix for Experiment 6
│   └── xgboost_class_weights_confusion_matrix.png # Confusion matrix for Experiment 7
├── src/
│   ├── data_exploration.py         # Script for dataset exploration
│   ├── experiment1.py              # Majority voting
│   ├── experiment2.py              # Weighted voting
│   ├── experiment3.py              # Weighted voting with filtering
│   ├── experiment4.py              # Task-specific weighted voting
│   ├── experiment5.py              # Random Forest model
│   ├── experiment6.py              # XGBoost model
│   ├── experiment7.py              # XGBoost with class weights
│   └── utils.py                    # Shared helper functions
├── notebooks/
│   ├── data_exploration.ipynb      # Jupyter notebook for exploring the dataset
│   ├── experiment_workflow.ipynb   # Interactive notebook for reproducing all experiments
├── README.md                       # Project documentation
└── requirements.txt                # Python dependencies
