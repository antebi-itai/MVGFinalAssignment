import numpy as np


def get_image_center(h, w):
	x_0 = w // 2 - 0.5 * (w % 2)
	y_0 = h // 2 - 0.5 * (h % 2)
	return x_0, y_0


def get_3d_points_from_disparities(disparity_map, focal_length, base_line, x_0, y_0, right_image=True):
	# extract consts
	h, w = disparity_map.shape

	# calculate Z
	disparity_map = np.where(disparity_map == 0, np.nan, disparity_map)
	Z = (base_line * focal_length) / disparity_map

	# calculate Y
	X, Y = np.meshgrid(range(w), range(h))
	Y = ((Y - y_0) / focal_length) * Z
	X = ((X - x_0) / focal_length) * Z
	if right_image: X += base_line

	# build 3D points
	points = np.stack((X, Y, Z, np.ones_like(Y)), axis=0)
	return points


def get_camera_matrices(focal_length, base_line, x_0, y_0):
	K = np.array([[focal_length, 0, x_0],
				  [0, focal_length, y_0],
				  [0, 0, 1]])
	calibrated_P = np.concatenate((np.eye(3), np.array([[-base_line, 0, 0]]).T), axis=1)
	P = np.matmul(K, calibrated_P)
	return P, K
