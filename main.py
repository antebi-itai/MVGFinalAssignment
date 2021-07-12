from matplotlib import pyplot as plt
import numpy as np

import data
import visualization
import evaluation
import geometry
import utils

"""			3.1  Sanity Check			"""
# choose scene
chosen_scene = data.SCENES_WITH_GT_DISPARITIES[data.CHOSEN_GT_SCENE_INDEX]
# get images and gt disparities of scene
img_l, img_r = data.get_scene_images(chosen_scene)
d_l, d = data.get_scene_disparities(chosen_scene)
# visualize images and gt disparities
visualization.plot_scene(img_l, img_r, d_l, d)
# visualize a validation of gt disparities
differences_image = evaluation.get_disparity_rgb_distances(img_l, img_r, d)
visualization.plot_image(differences_image)

"""			3.2  3D Plot				"""
# extract constants
h, w = img_r.shape[:2]
x_0, y_0 = geometry.get_image_center(h=h, w=w)
# calculate 3d points from the two views
points_r = geometry.get_3d_points_from_disparities(disparity_map=d,
												   focal_length=data.FOCAL_LENGTH, base_line=data.BASELINE,
												   x_0=x_0, y_0=y_0,
												   right_image=True)
points_l = geometry.get_3d_points_from_disparities(disparity_map=d_l,
												   focal_length=data.FOCAL_LENGTH, base_line=data.BASELINE,
												   x_0=x_0, y_0=y_0,
												   right_image=False)
# get camera matrices for the two views
P_r, K_r = geometry.get_camera_matrices(focal_length=data.FOCAL_LENGTH, base_line=data.BASELINE,
										x_0=x_0, y_0=y_0)
P_l, K_l = geometry.get_camera_matrices(focal_length=data.FOCAL_LENGTH, base_line=0,
										x_0=x_0, y_0=y_0)
# plot the 3D points and the camera
utils.plot_cameras(P=np.stack((P_l, P_r), axis=0),
				   K=np.stack((K_l, K_r), axis=0),
				   X=np.concatenate((points_l.reshape(4, -1), points_r.reshape(4, -1)), axis=1),
				   title="3.2_3D_plot_reconstruction",
				   point_colors=np.concatenate((img_l.reshape(-1, 3), img_r.reshape(-1, 3)), axis=0))

"""				Finish					"""
plt.show()