import cv2
import numpy as np
import os

from model.model import Model

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    img_path = os.path.join(script_dir, "coat1.png")

    image_data = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if image_data is None:
        raise FileNotFoundError(f"Could not find or open {img_path}")

    image_data = cv2.resize(image_data, (28, 28))
    image_data = 255 - image_data
    image_data = (image_data.reshape(1, -1).astype(np.float32) - 127.5) / 127.5

    model_path = os.path.join(script_dir, '..', 'saves', 'fashion_mnist.model')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    model = Model.load(model_path)
    confidences = model.predict(image_data)

    if model.output_layer_activation is not None:
        predictions = model.output_layer_activation.predictions(confidences)
    else:
        predictions = np.argmax(confidences, axis=1)

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

    prediction_label = fashion_mnist_labels[predictions[0]]
    print("Predicted:", prediction_label)


