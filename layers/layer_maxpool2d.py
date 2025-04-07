import cupy as cp

def pool2col(inputs, pool_h, pool_w, stride):
    N, C, H, W = inputs.shape
    out_h = (H - pool_h) // stride + 1
    out_w = (W - pool_w) // stride + 1
    col = cp.zeros((N * C * out_h * out_w, pool_h * pool_w), dtype=inputs.dtype)
    idx = 0
    for oh in range(out_h):
        h_start = oh * stride
        for ow in range(out_w):
            w_start = ow * stride
            patch = inputs[:, :, h_start:h_start + pool_h, w_start:w_start + pool_w]
            patch_2d = patch.reshape(-1, pool_h * pool_w)
            col[idx:idx + patch_2d.shape[0], :] = patch_2d
            idx += patch_2d.shape[0]
    return col, out_h, out_w

def col2pool(col, N, C, H, W, pool_h, pool_w, stride, out_h, out_w):
    outputs = cp.zeros((N, C, H, W), dtype=col.dtype)
    idx = 0
    for oh in range(out_h):
        h_start = oh * stride
        for ow in range(out_w):
            w_start = ow * stride
            count = N * C
            patch_2d = col[idx:idx + count, :]
            idx += count
            patch_4d = patch_2d.reshape(N, C, pool_h, pool_w)
            outputs[:, :, h_start:h_start + pool_h, w_start:w_start + pool_w] += patch_4d
    return outputs

class Layer_MaxPool2D:
    def __init__(self, pool_size=2, stride=2):
        if isinstance(pool_size, int):
            self.pool_h = pool_size
            self.pool_w = pool_size
        else:
            self.pool_h, self.pool_w = pool_size
        self.stride = stride

    def forward(self, inputs, training):
        self.inputs = inputs
        N, C, H, W = inputs.shape
        col, out_h, out_w = pool2col(inputs, self.pool_h, self.pool_w, self.stride)
        max_vals = col.max(axis=1)
        argmax_vals = col.argmax(axis=1)
        self.output = max_vals.reshape(N, C, out_h, out_w)
        self.col = col
        self.argmax_vals = argmax_vals
        self.out_h, self.out_w = out_h, out_w
        self.N, self.C, self.H, self.W = N, C, H, W

    def backward(self, dvalues):
        dvalues_flat = dvalues.ravel()
        col_grad = cp.zeros_like(self.col)
        col_indices = cp.arange(len(self.argmax_vals))
        col_grad[col_indices, self.argmax_vals] = dvalues_flat
        dX = col2pool(col_grad, self.N, self.C, self.H, self.W, self.pool_h, self.pool_w, self.stride, self.out_h, self.out_w)
        self.dinputs = dX
