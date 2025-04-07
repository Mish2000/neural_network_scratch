import cupy as cp

def im2col(images, kernel_h, kernel_w, stride, padding):
    N, C, H, W = images.shape
    pad_h, pad_w = padding, padding
    out_h = (H + 2 * pad_h - kernel_h) // stride + 1
    out_w = (W + 2 * pad_w - kernel_w) // stride + 1

    if pad_h > 0 or pad_w > 0:
        images_padded = cp.pad(images, ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)))
    else:
        images_padded = images

    Np, Cp, Hp, Wp = images_padded.shape
    i0 = cp.repeat(cp.arange(kernel_h), kernel_w)
    i0 = cp.tile(i0, C)
    i1 = stride * cp.repeat(cp.arange(out_h), out_w)
    j0 = cp.tile(cp.arange(kernel_w), kernel_h)
    j0 = cp.tile(j0, C)
    j1 = stride * cp.tile(cp.arange(out_w), out_h)
    c = cp.repeat(cp.arange(C), kernel_h * kernel_w)

    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)
    c = c.reshape(-1, 1)

    cols = images_padded[:, c, i, j]
    cols = cols.transpose(0, 2, 1)
    cols = cols.reshape(N * out_h * out_w, -1)

    return cols, out_h, out_w

def col2im(cols, images_shape, kernel_h, kernel_w, stride, padding, out_h, out_w):
    N, C, H, W = images_shape
    pad_h, pad_w = padding, padding
    Hp = H + 2 * pad_h
    Wp = W + 2 * pad_w

    out = cp.zeros((N, C, Hp, Wp), dtype=cols.dtype)

    i0 = cp.repeat(cp.arange(kernel_h), kernel_w)
    i0 = cp.tile(i0, C)
    i1 = stride * cp.repeat(cp.arange(out_h), out_w)
    j0 = cp.tile(cp.arange(kernel_w), kernel_h)
    j0 = cp.tile(j0, C)
    j1 = stride * cp.tile(cp.arange(out_w), out_h)
    c = cp.repeat(cp.arange(C), kernel_h * kernel_w)

    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)
    c = c.reshape(-1, 1)

    cols_reshaped = cols.reshape(N, out_h * out_w, -1).transpose(0, 2, 1)

    for n in range(N):
        out[n, c, i, j] += cols_reshaped[n]

    if pad_h > 0 or pad_w > 0:
        out = out[:, :, pad_h:-pad_h, pad_w:-pad_w]
    return out
