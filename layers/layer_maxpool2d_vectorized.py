import cupy as np

class Layer_MaxPool2D_Vectorized:
    def __init__(self):
        self.pool_h = 2
        self.pool_w = 2
        self.stride = 2

    def forward(self, inputs, training):
        self.inputs = inputs
        N, C, H, W = inputs.shape
        out_h = H // 2
        out_w = W // 2

        reshaped = inputs.reshape(N, C, out_h, self.pool_h, out_w, self.pool_w)
        max_vals = np.max(reshaped, axis=(3,5))
        self.output = max_vals

        reshaped_4d = reshaped.reshape(N, C, out_h, out_w, 4)
        self.argmax_local = np.argmax(reshaped_4d, axis=4).astype(np.int32)

    def backward(self, dvalues):
        N, C, out_h, out_w = dvalues.shape
        dX = np.zeros_like(self.inputs)

        dvalues_exp = dvalues[..., np.newaxis]

        offset_hw = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.int32)

        row_offset = offset_hw[self.argmax_local][...,0]
        col_offset = offset_hw[self.argmax_local][...,1]

        n_idx = np.arange(N)[:,None,None,None]
        c_idx = np.arange(C)[None,:,None,None]
        n_idx = np.broadcast_to(n_idx, dvalues.shape)
        c_idx = np.broadcast_to(c_idx, dvalues.shape)

        oh_idx = np.arange(out_h)[None,None,:,None]
        oh_idx = np.broadcast_to(oh_idx, dvalues.shape)
        ow_idx = np.arange(out_w)[None,None,None,:]
        ow_idx = np.broadcast_to(ow_idx, dvalues.shape)

        abs_h = oh_idx * 2 + row_offset
        abs_w = ow_idx * 2 + col_offset

        dX[n_idx, c_idx, abs_h, abs_w] += dvalues_exp[...,0]

        self.dinputs = dX
