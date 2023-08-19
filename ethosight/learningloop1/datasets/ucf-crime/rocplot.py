import click
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

@click.command()
@click.argument('input_filename', type=click.Path(exists=True))
@click.argument('output_filename', type=click.Path())
def plot_roc(input_filename, output_filename):
    """
    Plot ROC curve from data in INPUT_FILENAME and save it to OUTPUT_FILENAME
    """
    
    # Read the data
    data = np.loadtxt(input_filename, delimiter=',')

    # Split the data into labels and scores
    y_true = data[:, 0]
    y_scores = data[:, 1]

    # Compute the ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    # Plot the ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    
    # Save the plot as specified
    plt.savefig(output_filename)
    
    plt.show()

    print(f'ROC curve saved to {output_filename}')

if __name__ == '__main__':
    plot_roc()

