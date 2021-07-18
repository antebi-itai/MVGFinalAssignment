import numpy as np
import utils


def get_flatten_index(width, i, j):
	return i * width + j


def get_edges(height, width):
	i, j = np.indices((height, width))
	indices = i * width + j

	left_right_edges = np.stack((indices, shift_image(indices, disparity_x=-1, disparity_y=0)), axis=-1).reshape(height * width, 2)
	left_right_edges = left_right_edges[~np.isnan(left_right_edges).any(axis=1)]
	top_bottom_edges = np.stack((indices, shift_image(indices, disparity_x=0, disparity_y=-1)), axis=-1).reshape(height * width, 2)
	top_bottom_edges = top_bottom_edges[~np.isnan(top_bottom_edges).any(axis=1)]
	return np.concatenate((left_right_edges, top_bottom_edges), axis=0).astype(np.int32)


def shift_image(image, disparity_x=0, disparity_y=0):
	non = lambda s: s if s < 0 else None
	mom = lambda s: max(0, s)
	shifted_image = np.zeros_like(image) * np.nan
	shifted_image[mom(disparity_y):non(disparity_y), mom(disparity_x):non(disparity_x)] = \
		image[mom(-disparity_y):non(-disparity_y), mom(-disparity_x):non(-disparity_x)]
	return shifted_image


def pairwise_cost(K=15, max_disp=80):
	pairwise_cost_mat = K * np.ones((max_disp, max_disp))
	pairwise_cost_mat = pairwise_cost_mat * (~np.eye(max_disp).astype(np.bool))
	return pairwise_cost_mat.astype(np.int32)


def unary_cost(right_image, left_image, max_disp=80):
	assert (right_image.shape == left_image.shape) and (len(right_image.shape) == 3) and (right_image.shape[-1] == 3)
	h, w, _ = right_image.shape

	gray_right_image = utils.rgb2grey(right_image)
	gray_left_image = utils.rgb2grey(left_image)
	unary_cost_box = np.stack([np.abs(gray_right_image - shift_image(gray_left_image, disparity_x=-disparity))
							   for disparity in range(1, max_disp + 1)],
							  axis=-1)
	return unary_cost_box.reshape(h * w, max_disp).astype(np.int32)
