import cupy as cp
from layers.conv_utils import im2col, col2im

class Layer_Conv2D:
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, padding=0,
                 weight_regularizer_l1=0, weight_regularizer_l2=0,
                 bias_regularizer_l1=0, bias_regularizer_l2=0):

        if isinstance(kernel_size, int):
            self.kernel_h = kernel_size
            self.kernel_w = kernel_size
        else:
            self.kernel_h, self.kernel_w = kernel_size
        self.stride = stride
        self.padding = padding
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2

        fan_in = in_channels * self.kernel_h * self.kernel_w
        limit = cp.sqrt(2.0 / fan_in)
        self.weights = cp.random.randn(out_channels, in_channels,
                                       self.kernel_h, self.kernel_w) * limit

        self.biases = cp.zeros((out_channels,), dtype=cp.float32)

    def forward(self, inputs, training):
        self.inputs = inputs
        self.batch_size, _, self.in_h, self.in_w = inputs.shape
        col, self.out_h, self.out_w = im2col(inputs, self.kernel_h, self.kernel_w, self.stride, self.padding)
        self.W_2d = self.weights.reshape(self.out_channels, -1)
        out = col.dot(self.W_2d.T) + self.biases
        out = out.reshape(self.batch_size, self.out_h, self.out_w, self.out_channels)
        self.output = out.transpose(0, 3, 1, 2)

    def backward(self, dvalues):
        dvalues_reshaped = dvalues.transpose(0, 2, 3, 1).reshape(-1, self.out_channels)
        col, _, _ = im2col(self.inputs, self.kernel_h, self.kernel_w, self.stride, self.padding)
        self.dweights_2d = dvalues_reshaped.T.dot(col)
        self.dweights_2d = self.dweights_2d.reshape(self.out_channels, self.in_channels, self.kernel_h, self.kernel_w)
        self.dbiases = cp.sum(dvalues_reshaped, axis=0)
        col_grads = dvalues_reshaped.dot(self.W_2d)
        image_shape = (self.batch_size, self.in_channels, self.in_h, self.in_w)
        dX = col2im(
            col_grads,
            image_shape,
            self.kernel_h,
            self.kernel_w,
            self.stride,
            self.padding,
            self.out_h,
            self.out_w
        )
        self.dinputs = dX
        self.dweights = self.dweights_2d
        if self.weight_regularizer_l1 > 0:
            dL1 = cp.sign(self.weights)
            self.dweights += self.weight_regularizer_l1 * dL1
        if self.weight_regularizer_l2 > 0:
            self.dweights += 2 * self.weight_regularizer_l2 * self.weights
        if self.bias_regularizer_l1 > 0:
            dL1 = cp.sign(self.biases)
            self.dbiases += self.bias_regularizer_l1 * dL1
        if self.bias_regularizer_l2 > 0:
            self.dbiases += 2 * self.bias_regularizer_l2 * self.biases

    def get_parameters(self):
        return (self.weights, self.biases)

    def set_parameters(self, weights, biases):
        self.weights = weights
        self.biases = biases
