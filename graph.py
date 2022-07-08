import matplotlib.pyplot as plt
import numpy as np

def graph_loss(title, training_filename, validation_filename, mode = 'npy'):
    if mode == 'npy':
        training_loss = np.load(training_filename)
        validation_loss = np.load(validation_filename)
    else:
        training_loss = np.genfromtxt(training_filename,delimiter=',')
        validation_loss = np.genfromtxt(validation_filename,delimiter=',')

    plt.plot(training_loss,"-b", label="Training Loss")
    plt.plot(validation_loss,"-r", label="Validation Loss")
    plt.locator_params(axis="x", integer=True, tight=True)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title + " Training and Validation Loss")
    plt.legend(loc="upper right")
    plt.savefig(title + ' Loss.png', bbox_inches='tight')
    plt.clf()

if __name__ == "__main__":
    # graph_loss("Basic Weighting", 'basic_weighting_train_losses.npy', 'basic_weighting_validation_losses.npy')
    # graph_loss("No Weighting", 'no_weighting_train_losses.npy', 'no_weighting_validation_losses.npy')
    # graph_loss("No Weighting Augmented", 'no_weighting_augmented_train_losses.npy', 'no_weighting_augmented_validation_losses.npy')
    # graph_loss("Custom Architecture", 'custom_train_losses.npy', 'custom_val_losses.npy')
    graph_loss("U-Net", 'unet_trainloss_1.csv', "unet_valloss_1.csv", 'csv')