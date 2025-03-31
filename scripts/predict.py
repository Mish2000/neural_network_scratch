import cv2

from activations.activation_relu import Activation_ReLU
from activations.activation_softmax import Activation_Softmax
from layers.layer_dense import Layer_Dense
from losses.loss_categorical_crossentropy import Loss_CategoricalCrossentropy
from model import Model


def prepare_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (28, 28))
    img = 255 - img
    img = img.astype('float32') / 255.0
    img = img.reshape(1, 784)
    return img

def make_prediction(image_path, model_path):
    model = Model()
    model.add(Layer_Dense(784, 64))
    model.add(Activation_ReLU())
    model.add(Layer_Dense(64, 10))
    model.add(Activation_Softmax())
    model.set(loss=Loss_CategoricalCrossentropy())

    model.finalize()
    model.load_parameters(model_path)
    img = prepare_image(image_path)
    prediction = model.predict(img)
    return prediction


if __name__ == "__main__":
    test_image_path = '../pants.png'
    model_save_path = '../saves/model_parameters.pkl'
    prediction = make_prediction(test_image_path, model_save_path)
    print(f"Predicted class: {prediction}")
