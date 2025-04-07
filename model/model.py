import copy
import os
import pickle
import cv2
import cupy as np

from layers.layer_input import Layer_Input

class Model:
    def __init__(self):
        self.layers = []
        self.trainable_layers = []
        self.loss = None
        self.accuracy = None
        self.optimizer = None
        self.softmax_classifier_output = None
        self.output_layer_activation = None

    def add(self, layer):
        self.layers.append(layer)

    def set(self, *, loss=None, optimizer=None, accuracy=None):
        if loss is not None:
            self.loss = loss
        if optimizer is not None:
            self.optimizer = optimizer
        if accuracy is not None:
            self.accuracy = accuracy

    def finalize(self):
        self.input_layer = Layer_Input()
        for i, layer in enumerate(self.layers):
            if i == 0:
                layer.prev = self.input_layer
                layer.next = self.layers[i+1] if i < len(self.layers) - 1 else self.loss
            else:
                layer.prev = self.layers[i - 1]
                layer.next = self.layers[i + 1] if i < len(self.layers) - 1 else self.loss
            if hasattr(layer, 'weights'):
                self.trainable_layers.append(layer)
        if self.loss is not None:
            self.loss.remember_trainable_layers(self.trainable_layers)

    def train(self, X, y, *, epochs=1, batch_size=None, print_every=1, validation_data=None):
        self.accuracy.init(y)
        if batch_size is not None:
            steps = len(X) // batch_size
            if steps * batch_size < len(X):
                steps += 1
        else:
            steps = 1

        for epoch in range(1, epochs + 1):
            print(f"epoch: {epoch}")
            self.loss.new_pass()
            self.accuracy.new_pass()

            for step in range(steps):
                if batch_size is None:
                    batch_X = X
                    batch_y = y
                else:
                    batch_X = X[step * batch_size:(step + 1) * batch_size]
                    batch_y = y[step * batch_size:(step + 1) * batch_size]

                output = self.forward(batch_X, training=True)
                data_loss, reg_loss = self.loss.calculate(output, batch_y, include_regularization=True)
                loss_val = data_loss + reg_loss

                predictions = np.argmax(output, axis=1)

                acc_val = self.accuracy.calculate(predictions, batch_y)

                self.backward(output, batch_y)

                '''
                                for i, layer in enumerate(self.layers):
                    # If the layer has trainable parameters
                    if hasattr(layer, 'dweights'):
                        dw_mean = float(np.mean(np.abs(layer.dweights)))
                        db_mean = float(np.mean(np.abs(layer.dbiases)))
                        print(f"   layer[{i}] avg|dW|={dw_mean:.8f}, avg|dB|={db_mean:.8f}")
                '''

                layer0 = self.trainable_layers[0]

               # print("  dweights layer0 avg abs grad:", float(np.mean(np.abs(layer0.dweights))))
               # print("  dbiases layer0 avg abs grad:", float(np.mean(np.abs(layer0.dbiases))))

                self.optimizer.pre_update_params()

                # layer0 was updated:
                old_weights = layer0.weights.copy()
                self.optimizer.update_params(layer0)
                new_weights = layer0.weights
                diff = np.mean(np.abs(new_weights - old_weights))
                #print("  average update on layer0 weights:", float(diff))

                for layer in self.trainable_layers:
                    self.optimizer.update_params(layer)
                self.optimizer.post_update_params()

                '''
                                if (not step % print_every) or (step == steps - 1):
                    print(f" step: {step}, acc: {acc_val:.3f}, loss: {loss_val:.3f} "
                          f"(data_loss: {data_loss:.3f}, reg_loss: {reg_loss:.3f}), "
                          f"lr: {self.optimizer.current_learning_rate}")
                '''

            epoch_data_loss, epoch_reg_loss = self.loss.calculate_accumulated(include_regularization=True)
            epoch_loss = epoch_data_loss + epoch_reg_loss
            epoch_acc = self.accuracy.calculate_accumulated()

            print(f"training, acc: {epoch_acc:.3f}, loss: {epoch_loss:.3f} "
                  f"(data_loss: {epoch_data_loss:.3f}, reg_loss: {epoch_reg_loss:.3f}), "
                  f"lr: {self.optimizer.current_learning_rate}")

            if validation_data is not None:
                self.evaluate(*validation_data, batch_size=batch_size)

    def evaluate(self, X_val, y_val, *, batch_size=None):
        self.loss.new_pass()
        self.accuracy.new_pass()

        if batch_size is not None:
            val_steps = len(X_val) // batch_size
            if val_steps * batch_size < len(X_val):
                val_steps += 1
        else:
            val_steps = 1

        for step in range(val_steps):
            if batch_size is None:
                batch_X = X_val
                batch_y = y_val
            else:
                batch_X = X_val[step * batch_size:(step + 1) * batch_size]
                batch_y = y_val[step * batch_size:(step + 1) * batch_size]

            output = self.forward(batch_X, training=False)
            self.loss.calculate(output, batch_y)

            predictions = np.argmax(output, axis=1)

            self.accuracy.calculate(predictions, batch_y)

        val_loss = self.loss.calculate_accumulated()
        val_acc = self.accuracy.calculate_accumulated()
        print(f"validation, acc: {val_acc:.3f}, loss: {val_loss:.3f}")

    def predict(self, X, *, batch_size=None):
        if batch_size is not None:
            steps = len(X) // batch_size
            if steps * batch_size < len(X):
                steps += 1
        else:
            steps = 1

        outputs = []
        for step in range(steps):
            if batch_size is None:
                batch_X = X
            else:
                batch_X = X[step * batch_size:(step + 1) * batch_size]
            batch_out = self.forward(batch_X, training=False)
            outputs.append(batch_out)

        return np.vstack(outputs)

    def forward(self, X, training):
        self.input_layer.forward(X, training)
        for layer in self.layers:
            layer.forward(layer.prev.output, training)
        return layer.output

    def backward(self, output, y):
        self.loss.backward(output, y)
        dinputs = self.loss.dinputs
        for layer in reversed(self.layers):
            layer.backward(dinputs)
            dinputs = layer.dinputs

    def get_parameters(self):
        params = []
        for layer in self.trainable_layers:
            params.append(layer.get_parameters())
        return params

    def set_parameters(self, parameters):
        for param_set, layer in zip(parameters, self.trainable_layers):
            layer.set_parameters(*param_set)

    def save_parameters(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.get_parameters(), f)

    def load_parameters(self, path):
        with open(path, 'rb') as f:
            self.set_parameters(pickle.load(f))

    def save(self, path):
        model_copy = copy.deepcopy(self)
        model_copy.loss.new_pass()
        model_copy.accuracy.new_pass()
        model_copy.input_layer.__dict__.pop('output', None)
        model_copy.loss.__dict__.pop('dinputs', None)
        for layer in model_copy.layers:
            for prop in ['inputs', 'output', 'dinputs']:
                layer.__dict__.pop(prop, None)
        with open(path, 'wb') as f:
            pickle.dump(model_copy, f)

    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            model = pickle.load(f)
        return model

    def load_mnist_dataset(self, dataset, path):
        labels = os.listdir(os.path.join(path, dataset))
        X = []
        y = []
        for label in labels:
            for file in os.listdir(os.path.join(path, dataset, label)):
                image = cv2.imread(os.path.join(path, dataset, label, file), cv2.IMREAD_UNCHANGED)
                X.append(image)
                y.append(label)
        return np.array(X), np.array(y).astype('uint8')

    def create_data_mnist(self, path):
        X, y = self.load_mnist_dataset('train', path)
        X_test, y_test = self.load_mnist_dataset('test', path)
        return X, y, X_test, y_test

