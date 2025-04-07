import cupy as np
from losses.loss import Loss

class Activation_Softmax_Loss_CategoricalCrossentropy(Loss):
    def remember_trainable_layers(self, trainable_layers):
        self.trainable_layers = trainable_layers

    def forward(self, logits, y_true):
        exp_values = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

        samples = len(logits)
        clipped_probs = np.clip(probabilities, 1e-7, 1 - 1e-7)
        if len(y_true.shape) == 1:
            row_ids = np.arange(samples, dtype=np.int32)
            correct_confidences = clipped_probs[row_ids, y_true]
        else:
            correct_confidences = np.sum(clipped_probs * y_true, axis=1)

        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        self.dinputs = self.output.copy()
        row_ids = np.arange(samples, dtype=np.int32)
        self.dinputs[row_ids, y_true] -= 1
        self.dinputs /= samples

    def predictions(self, outputs):
        return np.argmax(self.output, axis=1)
