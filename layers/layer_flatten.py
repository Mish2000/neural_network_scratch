import cupy as np

class Layer_Flatten:
    def forward(self, inputs, training):
        self.inputs = inputs
        batch_size = inputs.shape[0]
        self.output = inputs.reshape(batch_size, -1)

    def backward(self, dvalues):
        self.dinputs = dvalues.reshape(self.inputs.shape)