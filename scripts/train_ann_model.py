import os
import cupy as np

from tensorflow.keras.datasets import fashion_mnist

from accuracy.accuracy_categorical import Accuracy_Categorical
from activations.activation_relu import Activation_ReLU
from activations.activation_softmax import Activation_Softmax
from layers.layer_dense import Layer_Dense
from layers.layer_dropout import Layer_Dropout
from losses.loss_categorical_crossentropy import Loss_CategoricalCrossentropy
from model.model import Model
from optimizers.optimizer_adam import Optimizer_Adam

def load_dataset():
    (X_train, y_train), (X_val, y_val) = fashion_mnist.load_data()
    X_train = X_train.astype('float32') / 255.0
    X_val = X_val.astype('float32') / 255.0
    X_train = X_train.reshape((X_train.shape[0], 784))
    X_val = X_val.reshape((X_val.shape[0], 784))
    return X_train, y_train, X_val, y_val

def main():
    X_train, y_train, X_val, y_val = load_dataset()

    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    X_val = np.asarray(X_val)
    y_val = np.asarray(y_val)

    model = Model()
    model.add(Layer_Dense(784, 128, weight_regularizer_l2=1e-4, bias_regularizer_l2=1e-4))
    model.add(Activation_ReLU())
    model.add(Layer_Dropout(rate=0.2))
    model.add(Layer_Dense(128, 64, weight_regularizer_l2=1e-4, bias_regularizer_l2=1e-4))
    model.add(Activation_ReLU())
    model.add(Layer_Dropout(rate=0.2))
    model.add(Layer_Dense(64, 10))
    model.add(Activation_Softmax())

    model.set(
        loss=Loss_CategoricalCrossentropy(),
        optimizer=Optimizer_Adam(learning_rate=0.0005),
        accuracy=Accuracy_Categorical()
    )

    model.finalize()
    model.output_layer_activation = model.layers[-1]

    print("Starting training...")
    model.train(X_train, y_train, epochs=50, batch_size=64, print_every=500)

    print("Evaluating on validation set...")
    model.evaluate(X_val, y_val)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    saves_dir = os.path.join(script_dir, '..', 'saves')
    if not os.path.exists(saves_dir):
        os.makedirs(saves_dir)

    model_full_path = os.path.join(saves_dir, 'fashion_mnist.model')
    model.save(model_full_path)
    print(f"Full model (with weights) saved to: {model_full_path}")

if __name__ == '__main__':
    main()
