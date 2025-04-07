import os
import cupy as np
from tensorflow.keras.datasets import fashion_mnist
from accuracy.accuracy_categorical import Accuracy_Categorical
from activations.activation_relu import Activation_ReLU
from activations.activation_softmax_loss_categorical_crossentropy import Activation_Softmax_Loss_CategoricalCrossentropy
from layers.layer_conv2d import Layer_Conv2D
from layers.layer_flatten import Layer_Flatten
from layers.layer_dense import Layer_Dense
from layers.layer_maxpool2d_vectorized import Layer_MaxPool2D_Vectorized
from optimizers.optimizer_adam import Optimizer_Adam
from layers.layer_dropout import Layer_Dropout
from model.model import Model

def main():
    (X_train, y_train), (X_val, y_val) = fashion_mnist.load_data()

    X_train = (X_train.astype(np.float32) - 127.5) / 127.5
    X_val   = (X_val.astype(np.float32)   - 127.5) / 127.5

    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    X_val   = np.asarray(X_val)
    y_val   = np.asarray(y_val)

    X_train = np.expand_dims(X_train, axis=1)
    X_val = np.expand_dims(X_val, axis=1)

    model = Model()

    model.add(Layer_Conv2D(
        in_channels=1,
        out_channels=32,
        kernel_size=3,
        stride=1,
        padding=1,
    ))

    model.add(Activation_ReLU())

    model.add(Layer_MaxPool2D_Vectorized())

    model.add(Layer_Conv2D(
        in_channels=32,
        out_channels=128,
        kernel_size=3,
        stride=1,
        padding=1,
    ))

    model.add(Activation_ReLU())

    model.add(Layer_MaxPool2D_Vectorized())

    model.add(Layer_Conv2D(
        in_channels=128,
        out_channels=256,
        kernel_size=3,
        stride=1,
        padding=1,
    ))

    model.add(Activation_ReLU())

    model.add(Layer_Dropout(rate=0.1))

    model.add(Layer_Flatten())

    model.add(Layer_Dense(
        256 * 7 * 7, 256,
        weight_regularizer_l2 = 1e-5
    ))

    model.add(Activation_ReLU())
    model.add(Layer_Dropout(rate=0.5))

    model.add(Layer_Dense(256, 128))
    model.add(Activation_ReLU())
    model.add(Layer_Dropout(rate=0.5))

    model.add(Layer_Dense(128, 10))

    model.set(
        loss=Activation_Softmax_Loss_CategoricalCrossentropy(),
        optimizer=Optimizer_Adam(learning_rate=0.001),
        accuracy=Accuracy_Categorical()
    )

    model.finalize()
    model.output_layer_activation = model.loss

    print("Starting CNN training on full 60k Fashion-MNIST...")

    model.train(
        X_train, y_train,
        epochs=5,
        batch_size=64,
        print_every=200,
        validation_data=(X_val, y_val)
    )

    print("Evaluating on validation set again...")
    model.evaluate(X_val, y_val)

    script_dir  = os.path.dirname(os.path.abspath(__file__))
    saves_dir   = os.path.join(script_dir, '..', 'saves')
    if not os.path.exists(saves_dir):
        os.makedirs(saves_dir)
    model_full_path = os.path.join(saves_dir, 'fashion_mnist_cnn.model')
    model.save(model_full_path)
    print(f"Model saved to: {model_full_path}")

if __name__ == '__main__':
    main()
