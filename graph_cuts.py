import numpy as np
import torch
import torch.nn.functional as F
from skimage import color
from pygco import cut_from_graph
import utils
import data
import evaluation
import visualization
import geometry

MAX_DISP = 80
DEFAULT_PAIRWISE_K = 15
MAX_DEPTH_IN_IMAGE = 50


def shift_image(image, disparity_x=0, disparity_y=0):
	non = lambda s: s if s < 0 else None
	mom = lambda s: max(0, s)
	shifted_image = np.zeros_like(image) * np.nan
	shifted_image[mom(disparity_y):non(disparity_y), mom(disparity_x):non(disparity_x)] = \
		image[mom(-disparity_y):non(-disparity_y), mom(-disparity_x):non(-disparity_x)]
	return shifted_image


def disparity_mat_to_int32(disparity_mat):
	if disparity_mat.dtype == np.int32:
		return disparity_mat
	disparity_mat[np.isnan(disparity_mat)] = np.inf
	# np.array(np.inf).astype(np.int32) == -2147483648
	# but this works and 2147483647 causes pygco to crash RuntimeError: Unknown exception
	return disparity_mat.astype(np.int32)


def unfold(np_image, kernel_size):
	h, w, _ = np_image.shape
	tensor_image = torch.tensor(np_image).permute(2, 0, 1)
	tensor_patches = F.unfold(tensor_image.unsqueeze(0), kernel_size=kernel_size, padding=kernel_size//2).squeeze()
	tensor_patches = tensor_patches.t().reshape(h, w, 3, kernel_size, kernel_size)
	tensor_patches = tensor_patches.permute(0, 1, 3, 4, 2). reshape(h, w, kernel_size*kernel_size*3)
	np_patches = np.array(tensor_patches)
	return np_patches


# ---------- 4.2  edges ---------- #

def get_edges(height, width):
	i, j = np.indices((height, width))
	indices = i * width + j

	left_right_edges = np.stack((indices, shift_image(indices, disparity_x=-1, disparity_y=0)), axis=-1).reshape(height * width, 2)
	left_right_edges = left_right_edges[~np.isnan(left_right_edges).any(axis=1)]
	top_bottom_edges = np.stack((indices, shift_image(indices, disparity_x=0, disparity_y=-1)), axis=-1).reshape(height * width, 2)
	top_bottom_edges = top_bottom_edges[~np.isnan(top_bottom_edges).any(axis=1)]
	return np.concatenate((left_right_edges, top_bottom_edges), axis=0).astype(np.int32)


# ---------- 4.2  pairwise cost ---------- #

def pairwise_cost(K=DEFAULT_PAIRWISE_K, max_disp=MAX_DISP):
	pairwise_cost_mat = K * np.ones((max_disp, max_disp))
	pairwise_cost_mat = pairwise_cost_mat * (~np.eye(max_disp).astype(bool))
	return disparity_mat_to_int32(pairwise_cost_mat)


def pairwise_cost_l1(K=DEFAULT_PAIRWISE_K, max_disp=MAX_DISP):
	i, j = np.indices((max_disp, max_disp))
	return disparity_mat_to_int32(K * np.abs(i-j))


def pairwise_cost_l1_saturated(K=DEFAULT_PAIRWISE_K, max_disp=MAX_DISP, **kwargs):
	if "M" not in kwargs:
		M = (np.ones((max_disp, max_disp)) * 10*K).astype(np.int32)
	return np.minimum(pairwise_cost_l1(K=K, max_disp=max_disp), M)


# ---------- 4.2  unary cost ---------- #

def unary_cost(right_image, left_image, max_disp=MAX_DISP):
	assert (right_image.shape == left_image.shape) and (len(right_image.shape) == 3) and (right_image.shape[-1] == 3)
	h, w, _ = right_image.shape

	gray_right_image = utils.rgb2grey(right_image)
	gray_left_image = utils.rgb2grey(left_image)
	unary_cost_box = np.stack([np.abs(gray_right_image - shift_image(gray_left_image, disparity_x=-disparity))
							   for disparity in range(1, max_disp + 1)],
							  axis=-1)
	return disparity_mat_to_int32(unary_cost_box.reshape(h * w, max_disp))


def unary_cost_colored(right_image, left_image, max_disp=MAX_DISP, scale=2):
	assert (right_image.shape == left_image.shape) and (len(right_image.shape) == 3) and (right_image.shape[-1] == 3)
	h, w, _ = right_image.shape

	lab_right_image = color.rgb2lab(right_image)
	lab_left_image = color.rgb2lab(left_image)
	unary_cost_box = np.stack([np.linalg.norm(lab_right_image - shift_image(lab_left_image, disparity_x=-disparity), axis=2)
							   for disparity in range(1, max_disp + 1)],
							  axis=-1)
	unary_cost_box *= scale
	return disparity_mat_to_int32(unary_cost_box.reshape(h * w, max_disp))


def unary_cost_patches(right_image, left_image, max_disp=MAX_DISP, scale=2, kernel_size=3):
	assert (right_image.shape == left_image.shape) and (len(right_image.shape) == 3) and (right_image.shape[-1] == 3)
	h, w, _ = right_image.shape

	lab_right_image_patches = unfold(color.rgb2lab(right_image), kernel_size=kernel_size)
	lab_left_image_patches = unfold(color.rgb2lab(left_image), kernel_size=kernel_size)
	unary_cost_box = np.stack([np.linalg.norm(lab_right_image_patches - shift_image(lab_left_image_patches, disparity_x=-disparity), axis=2)
							   for disparity in range(1, max_disp + 1)],
							  axis=-1)
	unary_cost_box *= scale
	return disparity_mat_to_int32(unary_cost_box.reshape(h * w, max_disp))


# ---------- 4.2  calculate disparity map ---------- #

def mask_for_3d_points(points_3d, P, max_depth, h, w):
	projected_points = geometry.project_points(points_3d.reshape(4, h * w), P).reshape(3, h, w)
	x_s, y_s, ones = projected_points
	mask_seen_by_camera = np.logical_and.reduce((0 <= x_s, x_s < w, 0 <= y_s, y_s < h))
	mask_close_points = (points_3d[2] < max_depth)
	mask = np.logical_and(mask_close_points, mask_seen_by_camera)
	return mask


def images_to_disparity_map_and_3d_scene(scene, get_edges_fn, get_pairwise_cost_fn, get_unary_cost_fn,
										 P_l, P_r, K_l, K_r, subsection="4.2",
										 get_edges_kwargs=dict({}), get_pairwise_cost_kwargs=dict({}), get_unary_cost_kwargs=dict({})):
	print("-" * 50)
	print("Scene {0}:".format(scene))
	# read images of scene
	img_l, img_r = data.get_scene_images(scene)
	assert (img_l.shape == img_r.shape) and (len(img_l.shape) == 3) and (img_l.shape[-1] == 3)
	h, w, _ = img_l.shape
	x_0, y_0 = geometry.get_image_center(h=h, w=w)

	# obtain disparity map using graph-cuts
	edges = get_edges_fn(height=h, width=w, **get_edges_kwargs)
	pairwise_cost_mat = get_pairwise_cost_fn(**get_pairwise_cost_kwargs)
	unary_cost_mat = get_unary_cost_fn(right_image=img_r, left_image=img_l, **get_unary_cost_kwargs)
	disparity = cut_from_graph(edges=edges,
							   unary_cost=unary_cost_mat,
							   pairwise_cost=pairwise_cost_mat,
							   algorithm="expansion")
	disparity = disparity.reshape(h, w)

	# calculate 3d points
	points_r = geometry.get_3d_points_from_disparities(disparity_map=disparity,
													   focal_length=data.FOCAL_LENGTH, base_line=data.BASELINE,
													   x_0=x_0, y_0=y_0,
													   right_image=True)

	# filter out points that are not seen by the left camera / very far points
	mask = mask_for_3d_points(points_3d=points_r, P=P_l, max_depth=MAX_DEPTH_IN_IMAGE, h=h, w=w)
	filtered_disparity = np.where(mask, disparity, 0)
	filtered_points_r = points_r.reshape(4, h*w)[:, mask.reshape(h*w)]
	filtered_points_r_colors = img_r.reshape(h*w, 3)[mask.reshape(h*w), :]

	# calculate accuracy / outlier ratio if possible
	if scene in data.SCENES_WITH_GT_DISPARITIES:
		d_l, d = data.get_scene_disparities(scene)
		accuracy, outlier_ration = evaluation.evaluate_disparity(gt_disparity=d, pred_disparity=filtered_disparity)
		print("accuracy = {:.3f} \noutlier_ration = {:.1f}%".format(accuracy, outlier_ration * 100))

	# show disparity map
	visualization.plot_image(filtered_disparity)

	# plot the 3D points and the camera
	utils.plot_cameras(P=np.stack((P_l, P_r), axis=0),
					   K=np.stack((K_l, K_r), axis=0),
					   X=filtered_points_r,
					   title="{subsection}_3D_plot_{scene}".format(subsection=subsection, scene=scene),
					   point_colors=filtered_points_r_colors)
