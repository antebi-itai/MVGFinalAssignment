import numpy as np
import utils


def evaluate_disparity(gt_disparity, pred_disparity, c=3):
	assert gt_disparity.shape == pred_disparity.shape, "disparity maps must be of same shape"
	assert len(gt_disparity.shape) == 2, "disparity maps must be 2D"
	mask = np.logical_and(gt_disparity != 0, pred_disparity != 0)

	abs_diff = np.abs(gt_disparity[mask] - pred_disparity[mask])
	accuracy = abs_diff.mean()
	outlier_ration = (abs_diff > c).mean()
	return accuracy, outlier_ration


def get_disparity_rgb_distances(img_l, img_r, d):
	assert img_l.shape == img_r.shape, "images must be of same shape"
	assert (len(img_l.shape) == 3) and (img_l.shape[-1] == 3), "images must be in HxWx3 form"
	h, w, _ = img_l.shape

	# turn the two images to greyscale
	img_l_gray, img_r_gray = utils.rgb2grey(img_l), utils.rgb2grey(img_r)
	differences_image = np.zeros_like(img_l_gray)
	# loop over all the points in "img_R"
	for i in range(h):
		for j in range(w):
			# disparity of zero means unknown disparity due to occlusions
			if d[i][j] != 0:
				img_l_index = j + d[i][j]
				if 0 <= img_l_index < w:
					# compute the absolute difference between the point and the matching point in "img_L" according to formula 1
					differences_image[i][j] = np.abs(img_r_gray[i][j] - img_l_gray[i][img_l_index])
	return differences_image
