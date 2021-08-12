from PIL import Image
import numpy as np
import os

FOCAL_LENGTH = 3740
BASELINE = 0.16
SCENES_WITH_GT_DISPARITIES = ["Aloe", "Flowerpots", "Bowling1"]
SCENES_WITHOUT_GT_DISPARITIES = [scene for scene in os.listdir(os.path.join(".", "data")) if scene not in SCENES_WITH_GT_DISPARITIES]
CHOSEN_GT_SCENE_INDEX = 0
CHOSEN_WITHOUT_GT_SCENE_INDEX = -4


def read_image(image_path):
	pil_image = Image.open(image_path)
	np_image = np.array(pil_image)
	return np_image


def get_scene_images(scene_name):
	img_l = read_image(os.path.join(".", "data", scene_name, "img_L.png"))
	img_r = read_image(os.path.join(".", "data", scene_name, "img_R.png"))
	return img_l, img_r


def get_scene_disparities(scene_name):
	assert scene_name in SCENES_WITH_GT_DISPARITIES
	d_l = read_image(os.path.join(".", "data", scene_name, "D_L.png"))
	d = read_image(os.path.join(".", "data", scene_name, "D.png"))
	return d_l, d
