import numpy as np
import utils


# ---------- 3.2  3D Plot ---------- #

def get_image_center(h, w):
	x_0 = w // 2 - 0.5 * ((w + 1) % 2)
	y_0 = h // 2 - 0.5 * ((h + 1) % 2)
	return x_0, y_0


def get_camera_matrices(focal_length, base_line, x_0, y_0):
	K = np.array([[focal_length, 0, x_0],
				  [0, focal_length, y_0],
				  [0, 0, 1]])
	calibrated_P = np.concatenate((np.eye(3), np.array([[-base_line, 0, 0]]).T), axis=1)
	P = np.matmul(K, calibrated_P)
	return P, K


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


# ---------- 3.3  Novel View ---------- #

def project_points(points, P):
	return utils.pflat(np.matmul(P, points))


def novel_view_image(points, point_colors, P, h, w):
	"""
	Generate a 2D image viewing a scene with 3D points
	:param points: ndarray of shape [4, n_points], the 3D points
	:param point_colors. ndarray of shapes [n_points, 3]
	:param P: ndarray of shape [3, 4], the camera observing the novel view
	"""
	assert (len(points.shape) == 2) and (points.shape[0] == 4)
	assert (len(point_colors.shape) == 2) and (point_colors.shape[1] == 3)
	assert points.shape[1] == point_colors.shape[0]
	# project 3d points onto image plane
	projected_points = project_points(points, P)[:2]
	# sort points from farthest to nearest
	point_sort_order = np.flip(np.argsort(points[-2]))
	sorted_projected_points = projected_points.T[point_sort_order, :]
	sorted_point_colors = point_colors[point_sort_order, :]
	# paint the new image
	image = np.zeros((h, w, 3))
	for (x, y), color in zip(sorted_projected_points, sorted_point_colors):
		x, y = np.rint(x).astype(np.int32), np.rint(y).astype(np.int32)
		if (0 <= x < w) and (0 <= y < h):
			image[y][x] = color
	return image


def remove_black_stripes(image):
	image = np.copy(image)
	h, w, _ = image.shape
	for i in range(h):
		for j in range(w):
			# if pixel is empty
			if (image[i][j] == np.zeros(3)).all():
				surrounding_pixels = []
				for ii in [-1, 0, 1]:
					for jj in [-1, 0, 1]:
						if (0 <= i + ii < h) and (0 <= j + jj < w) and not (ii == 0 and jj == 0):
							surrounding_pixels.append(image[i + ii][j + jj])
				surrounding_pixels = np.stack(surrounding_pixels, axis=0)
				# and most of his surrounding is not empty
				if np.count_nonzero((surrounding_pixels != 0).all(axis=1)) > len(surrounding_pixels) / 2:
					# fill the pixel with mean of non-empty surroundings
					image[i][j] = np.mean(surrounding_pixels[(surrounding_pixels != 0).all(axis=1)], axis=0)
	return image
