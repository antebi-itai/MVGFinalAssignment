import matplotlib.pyplot as plt


def plot_image(np_image, title=None):
	plt.figure()
	cmap = "gray" if len(np_image.shape)==2 else None
	plt.imshow(np_image, cmap=cmap)
	if title is not None:
		plt.title(title)


def plot_scene(img_l, img_r, d_l, d):
	# plot
	fig, axs = plt.subplots(2, 2)
	axs[0, 0].imshow(img_l)
	axs[0, 0].set_title("img_L")
	axs[0, 1].imshow(img_r)
	axs[0, 1].set_title("img_R")
	axs[1, 0].imshow(d_l, cmap="gray")
	axs[1, 0].set_title("d_L")
	axs[1, 1].imshow(d, cmap="gray")
	axs[1, 1].set_title("d")
