from builtins import range
import numpy as np


def get_im2col_indices(x_shape, field_height, field_width, padding=1, stride=1):
    # First figure out what the size of the output should be
    N, C, H, W = x_shape
    #print(x_shape)
    #print("************ testing\n",field_height,field_width,padding,stride)
    assert (H + 2 * padding - field_height) % stride == 0
    assert (W + 2 * padding - field_height) % stride == 0
    out_height = (H + 2 * padding - field_height) / stride + 1
    out_width = (W + 2 * padding - field_width) / stride + 1
    #print("out he: ", out_height, "  out width: ",out_width)
    #print("input he: ", field_height, "input width: ",field_width)
    i0 = np.repeat(np.arange(field_height), field_width)
    #print("1: \n",i0)
    i0 = np.tile(i0, C)
    #print("2: \n",i0)
    i1 = stride * np.repeat(np.arange(int(out_height)), int(out_width))
    #print("3: \n",i1)
    j0 = np.tile(np.arange(field_width), field_height * C)
    #print("4: \n",j0)
    j1 = stride * np.tile(np.arange(int(out_width)), int(out_height))
    #print("5: \n",j1)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    #print("6: \n",i)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)
    #print("7: \n",j)
    k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)
    #print("8: \n",k)

    return (k, i, j)


def im2col_indices(x, field_height, field_width, padding=1, stride=1):
    """ An implementation of im2col based on some fancy indexing """
    # Zero-pad the input
    p = padding
    x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')

    k, i, j = get_im2col_indices(x.shape, field_height, field_width, padding,
                                 stride)
   # print("shapes\n k={0}\ni={1}\nj={2}".format(k,i,j))
    #print("x shape : ",x.shape)
  #  print("x image : \n",x_padded)
   # print("shapes\n k={0}\ni={1}\nj={2}".format(k.shape,i.shape,j.shape))
    #print("shapes\n k={0}\ni={1}\nj={2}".format(k,i,j))
    cols = x_padded[:, k, i, j]
    #print("cols shape {0}".format(cols.shape))
    #print("cols \n {0}".format(cols))
    C = x.shape[1]
    #print("C shape {0}".format(C))
    cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
    #print("Col shape after transpose \n{0}".format(cols))
    #print("cols shape {0}".format(cols.shape))
    #print("cols: \n{0}".format(cols))
    return cols


def col2im_indices(cols, x_shape, field_height=3, field_width=3, padding=1,
                   stride=1):
    """ An implementation of col2im based on fancy indexing and np.add.at """
    N, C, H, W = x_shape
    H_padded, W_padded = H + 2 * padding, W + 2 * padding
    x_padded = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)
    k, i, j = get_im2col_indices(x_shape, field_height, field_width, padding,
                                 stride)
    cols_reshaped = cols.reshape(C * field_height * field_width, -1, N)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)
    np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)
    if padding == 0:
        return x_padded
    return x_padded[:, :, padding:-padding, padding:-padding]

pass
