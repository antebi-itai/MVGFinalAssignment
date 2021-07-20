import numpy as np
from pygco import cut_from_graph
import utils
import data
import evaluation
import visualization
import geometry

K = 15
MAX_DEPTH_IN_IMAGE = 50


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


def images_to_disparity_map_and_3d_scene(scene, get_edges_fn, get_pairwise_cost_fn, get_unary_cost_fn,
										 P_l, P_r, K_l, K_r, subsection="4.2"):
	print("Scene {0}:".format(scene))
	# read images of scene
	img_l, img_r = data.get_scene_images(scene)
	assert (img_l.shape == img_r.shape) and (len(img_l.shape) == 3) and (img_l.shape[-1] == 3)
	h, w, _ = img_l.shape
	x_0, y_0 = geometry.get_image_center(h=h, w=w)

	# obtain disparity map using graph-cuts
	edges = get_edges_fn(height=h, width=w)
	pairwise_cost_mat = get_pairwise_cost_fn(K=K)
	unary_cost_mat = get_unary_cost_fn(right_image=img_r, left_image=img_l)
	disparity = cut_from_graph(edges=edges,
							   unary_cost=unary_cost_mat, #np.zeros((157990, 80), dtype=np.int32),
							   pairwise_cost=pairwise_cost_mat, #np.zeros((80, 80), dtype=np.int32),
							   algorithm="expansion")
	disparity = disparity.reshape(h, w)

	# calculate accuracy / outlier ratio if possible
	if scene in data.SCENES_WITH_GT_DISPARITIES:
		d_l, d = data.get_scene_disparities(scene)
		accuracy, outlier_ration = evaluation.evaluate_disparity(gt_disparity=d, pred_disparity=disparity)
		print("accuracy = {:.2f} \noutlier_ration = {:.2f}".format(accuracy, outlier_ration))

	# show disparity map
	visualization.plot_image(disparity)

	# calculate 3d points
	points_r = geometry.get_3d_points_from_disparities(disparity_map=disparity,
													   focal_length=data.FOCAL_LENGTH, base_line=data.BASELINE,
													   x_0=x_0, y_0=y_0,
													   right_image=True)

	# filter out points that are not seen by the left camera / very far points
	mask = geometry.mask_for_3d_points(points_1=points_r, P_2=P_l, max_depth=MAX_DEPTH_IN_IMAGE, h=h, w=w)

	# plot the 3D points and the camera
	utils.plot_cameras(P=np.stack((P_l, P_r), axis=0),
					   K=np.stack((K_l, K_r), axis=0),
					   X=points_r.reshape(4, h*w)[:, mask.reshape(h*w)],
					   title="{subsection}_3D_plot_{scene}".format(subsection=subsection, scene=scene),
					   point_colors=img_r.reshape(h*w, 3)[mask.reshape(h*w)])
