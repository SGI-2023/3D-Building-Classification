import argparse
import os
import os.path as osp
from sklearn.utils.discovery import all_displays
import torch
from collections import defaultdict, Counter
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score

# Function to calculate mean per class accuracy
def mean_per_class_acc(correct, labels):
    total_acc = 0
    for cls in torch.unique(labels):
        mask = labels == cls
        total_acc += correct[mask].sum() / mask.sum()
    return total_acc / len(torch.unique(labels))

def plot_confusion_matrix(cm, classes):
    """
    This function plots a confusion matrix using Seaborn's heatmap.
    """
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()
    

def calculate_metrics(labels, predictions):
    precision = precision_score(labels, predictions, average='macro')
    recall = recall_score(labels, predictions, average='macro')
    f1 = f1_score(labels, predictions, average='macro')
    return precision, recall, f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('folder', type=str)
    args = parser.parse_args()

    # get list of files
    files = os.listdir(args.folder)
    files = sorted([f for f in files if f.endswith('.pt')])

    sample_file_path = osp.join(args.folder, files[0])
    sample_data = torch.load(sample_file_path)
    print(sample_data.keys())
    
    # Dictionary to hold predictions and labels for each unique model ID
    grouped_predictions = defaultdict(list)
    model_labels = {}

    # preparing lists for storing all predictions and true labels for confusion matrix
    all_predictions = []
    all_labels = []

    # Load predictions and labels from .pt files
    files = os.listdir(args.folder)
    for f in tqdm(files):
        if f.endswith('.pt'):
            data = torch.load(osp.join(args.folder, f))
            unique_id = data['unique_id']  # Assuming unique_id is 'class_model'
            print(unique_id)           
            grouped_predictions[unique_id].append(data['pred'])
            model_labels[unique_id] = data['label']  # Assuming the label is the same for all views of a model

    print("=============================================================")
    #a Print out all unique IDs found
    print("All unique IDs:")
    for unique_id in grouped_predictions:
        print(unique_id)

    # Check the length of grouped_preds
    print(f"Total number of unique IDs: {len(grouped_predictions)}")
    
    print("==============================================================")
    # Perform voting for each model and calculate overall accuracy
    correct = 0
    total = 0
    model_accuracies = {}
    for unique_id, predictions in grouped_predictions.items():
        print(f"{unique_id}: {predictions}")
        # Count the most common prediction for each model
        voted_pred = Counter(predictions).most_common(1)[0][0]
        true_label = model_labels[unique_id]

        # collect for confusion matrix
        all_predictions.append(voted_pred)
        all_labels.append(true_label)

        if voted_pred == true_label:
            correct += 1
            model_accuracies[unique_id] = 100  # 100% accuracy for this model
        else:
            model_accuracies[unique_id] = 0  # 0% accuracy for this model
        total += 1
    print("================================================================")
    # Calculate and print accuracy for each model
    for unique_id, accuracy in sorted(model_accuracies.items()):
        print(f"Model {unique_id} - Accuracy: {accuracy}%")

    # Calculate and print overall accuracy
    overall_accuracy = 100 * correct / total
    print(f"\nOverall Accuracy: {overall_accuracy:.2f}%")

    # Compute additional metrics
    precision, recall, f1 = calculate_metrics(all_labels, all_predictions)
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall: {recall * 100:.2f}%")
    print(f"F1 Score: {f1 * 100:.2f}%")

    # Assuming all predictions and labels are collected...
    cm = confusion_matrix(all_labels, all_predictions)
    print("confusion matrix")
    print(cm)

    # # Assuming all predictions and labels are collected...
    # cm = confusion_matrix(all_labels, all_predictions)
    # classes = list(set(all_labels))  # Extract unique classes for labels
    # plot_confusion_matrix(cm, classes)

if __name__ == '__main__':
    main()

""" 
  If we had a list like [0, 0, 1, 1, 1], 
  the Counter would count 0 appearing twice and 1 appearing three times, 
  and most_common(1) would return [(1, 3)]. 
  The first [0] would get the tuple (1, 3), and 
  the second [0] would just get the 1 out of the tuple, 
  indicating that the most common prediction is 1
""" 