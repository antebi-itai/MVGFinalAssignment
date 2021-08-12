import matplotlib.pyplot as plt

FIG_SIZE = (10, 10)


# ---------- plot single image ---------- #

def plot_image(np_image, title=None):
	plt.figure(figsize=FIG_SIZE)
	cmap = "gray" if len(np_image.shape) == 2 else None
	plt.imshow(np_image, cmap=cmap)
	if title is not None:
		plt.title(title)
	plt.show()


# ---------- plot scene ---------- #

def plot_axes(axes, np_image, title=None):
	cmap = "gray" if len(np_image.shape) == 2 else None
	axes.imshow(np_image, cmap=cmap)
	if title is not None:
		axes.set_title(title)


def plot_scene(img_l, img_r, d_l, d):
	# plot
	fig, axs = plt.subplots(nrows=2, ncols=2, figsize=FIG_SIZE)
	plot_axes(axs[0, 0], img_l, title="Left Image")
	plot_axes(axs[0, 1], img_r, title="Right Image")
	plot_axes(axs[1, 0], d_l, title="Left Disparity Map")
	plot_axes(axs[1, 1], d, title="Right Disparity Map")
	fig.tight_layout()
	plt.show()
