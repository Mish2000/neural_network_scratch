import matplotlib.pyplot as plt


def plot_results():
    epochs = range(1, 11)
    training_accuracy = [0.893, 0.945, 0.961, 0.970, 0.975, 0.980, 0.984, 0.987, 0.989, 0.991]
    validation_accuracy = [0.970] * 10

    plt.plot(epochs, training_accuracy, 'bo-', label='Training accuracy')
    plt.plot(epochs, validation_accuracy, 'ro-', label='Validation accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    plot_results()
