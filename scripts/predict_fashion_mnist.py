import os
import cv2
import numpy as real_np
import cupy as cp
from model.model import Model


def is_cnn_model(model):
    if not model.trainable_layers:
        return False
    weights, biases = model.trainable_layers[0].get_parameters()
    if len(weights.shape) == 4:
        return True
    else:
        return False


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    img_path = os.path.join(script_dir, "bag1.png")

    image_data = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if image_data is None:
        raise FileNotFoundError(f"Could not find or open {img_path}")

    image_data = cv2.resize(image_data, (28, 28))
    image_data = 255 - image_data
    image_data = image_data.astype(real_np.float32)
    image_data = (image_data - 127.5) / 127.5

    model_path = os.path.join(script_dir, '..', 'saves', 'fashion_mnist_cnn.model')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    model = Model.load(model_path)

    cnn_flag = is_cnn_model(model)

    image_data = cp.asarray(image_data)
    if cnn_flag:
        image_data = cp.expand_dims(image_data, axis=(0, 1))  # (28,28) => (1,1,28,28)
    else:
        image_data = cp.reshape(image_data, (1, 784))  # (28,28) => (1,784)

    confidences = model.predict(image_data)

    if model.output_layer_activation is not None:
        predictions = model.output_layer_activation.predictions(confidences)
    else:
        predictions = cp.argmax(confidences, axis=1)

    predictions = int(predictions[0])

    fashion_mnist_labels = {
        0: 'T-shirt/top',
        1: 'Trouser',
        2: 'Pullover',
        3: 'Dress',
        4: 'Coat',
        5: 'Sandal',
        6: 'Shirt',
        7: 'Sneaker',
        8: 'Bag',
        9: 'Ankle boot'
    }
    label = fashion_mnist_labels[predictions]

    print("Predicted:", label)

    confidence_score = float(cp.max(confidences))
    print(f"Predicted: {label} (confidence: {confidence_score:.2%})")

