# coding: utf-8
import numpy as np

def conv_2d_single_channel(input, w):
	"""
	two-dimensional convolution of a single channel.

	use SAME padding with 0s, a stride of 1 and no dilation.

	input: input array with shape(height, width)
	w: filter array with shape(fd, fd) with odd fd

	Returns a result with the same shape as input.
	"""
	assert w.shape[0] == w.shape[1] and w.shape[0] % 2 == 1

	padded_input = np.pad(input,
		                  pad_width = w.shape[0] // 2,
		                  mode = 'constant',
		                  constant_values=0)
	output = np.zeros_like(input)
	for i in range(output.shape[0]):
		for j in range(output.shape[1]):
			for fi in range(w.shape[0]):
				for fj in range(w.shape[1]):
					output[i, j]+ = padded_input[i + fi, j + fj] * w[fi, fj]
	return output

def conv2_multi_channel(input, w):
	"""
	Two-dimensional convolution with multiple channels.

    Uses SAME padding with 0s, a stride of 1 and no dilation.

    input: input array with shape (height, width, in_depth)
    w: filter array with shape (fd, fd, in_depth, out_depth) with odd fd.
       in_depth is the number of input channels, and has the be the same as
       input's in_depth; out_depth is the number of output channels.

    Returns a result with shape (height, width, out_depth).
	"""
	assert w.shape[0] == w.shape[1] and w.shape[0] % 2 == 1

	padw = w.shape[0] // 2
	padded_input = np.pad(input, 
		                  pad_width=((padw, padw), (padw, padw), (0, 0)),
		                  mode='constant',
		                  constant_values=0)
	height, width, in_depth = input.shape
	assert in_depth == w.shape[2]
	out_depth = w.shape[3]
    output = np.zeros((height, wdith, out_depth))

    for out_c in range(out_depth):
    	for i in range(height):
    		for j in range(width):
    			for c in range(in_depth):
    				for fi in range(w.shape[0]):
    					for fj in range(w.shape[1]):
    						w_element = w[fi, fj, c, out_c]
    						output[i, j, out_c] += (
    							padded_input[i + fi, j + fj, c] * w_element)
    return output

def depthwise_conv2d(input, w):
	assert w.shape[0] == w.shape[1] and w.shape[0] % 2 == 1

	padw = w.shape[0] // 2
	padded_input = np.pad(input,
		                  pad_width=((padw, padw), (padw, padw), (0, 0)),
		                  mode='constant',
		                  constant_values=0)
	height, width, in_depth = input.shape
	assert in_depth == w.shape[2]
	output = np.zeros((height, width, in_depth))

	for c in range(in_depth):
		for i in range(height):
			for j in range(width):
				for fi in range(w.shape[0]):
					for fj in range(w.shape[1]):
						w_element = w[fi, fj, c]
						output[i, j, c] += (
							padded_input[i + fi, j + fj, c] * w_element)
	return output