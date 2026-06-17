"""
Metrics functions for evaluating model performance.
"""

import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, precision_recall_curve, average_precision_score


def create_and_save_accuracy_curve(train_accuracies, test_accuracies, n_epochs, save_path,
                                   file_name="accuracy_plot.png"):
    # Create a plot for the accuracies
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, n_epochs + 1), train_accuracies, label='Training Accuracy')
    plt.plot(range(1, n_epochs + 1), test_accuracies, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Test Accuracies')
    plt.legend()
    plt.grid(True)

    # Save the plot as an image
    plt.savefig(save_path + file_name)
    plt.close()
    return save_path + file_name


def create_and_save_f1_curve(f1_scores, n_epochs, model_type, output_path):
    """
    Create a plot of F1 scores saved during model training
    """
    plt.figure(figsize=(10, 6))

    plt.plot(range(1, n_epochs + 1), f1_scores, marker='o', linestyle='-')

    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title(f'F1 Score Development for Model {model_type}')

    plt.grid(True)
    plt.legend(['F1 Score'], loc='lower right')

    # Save the plot as an image
    plt.savefig(output_path + "f1_score_plot.png")
    plt.close()
    return output_path + "f1_score_plot.png"


def create_and_save_confusion_matrix(predictions, true_labels, save_path, file_name="confusion_matrix.png"):
    conf_matrix = confusion_matrix(true_labels, predictions, normalize='true')

    cm_display = ConfusionMatrixDisplay(confusion_matrix = conf_matrix)

    cm_display.plot()
    plt.title("Confusion Matrix")
    plt.savefig(save_path + file_name)
    plt.close()
    return save_path + file_name