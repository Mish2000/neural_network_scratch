from losses.loss import Loss
import cupy as np

class Loss_BinaryCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        sample_losses = -(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))
        sample_losses = np.mean(sample_losses, axis=1)
        return sample_losses

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        outputs = len(dvalues[0])
        clip_values = np.clip(dvalues, 1e-7, 1 - 1e-7)
        self.dinputs = -(y_true / clip_values - (1 - y_true) / (1 - clip_values)) / outputs
        self.dinputs = self.dinputs / samples

