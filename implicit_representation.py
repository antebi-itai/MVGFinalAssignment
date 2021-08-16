import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm
from torch.autograd import backward
import graph_cuts
import wandb
import data


class CostDataGenerator(Dataset):

	def __init__(self, cost_volume):
		self.cost_volume = cost_volume
		self.size = self.cost_volume.numel()

		i, j, d = torch.meshgrid(*[torch.arange(dim_size) for dim_size in self.cost_volume.shape])
		self.input_coordinates = torch.stack([i, j, d], dim=0).reshape(3, self.size).type(torch.FloatTensor)
		self.output_costs = self.cost_volume.flatten().type(torch.FloatTensor)

	def __len__(self):
		return self.size

	def __getitem__(self, idx):
		return self.input_coordinates[:, idx], self.output_costs[idx]


class LinearNet(nn.Module):

	def __init__(self, hidden_size=500):
		super(LinearNet, self).__init__()
		# kernel
		self.fc1 = nn.Linear(3, hidden_size)  # 5*5 from image dimension
		self.fc2 = nn.Linear(hidden_size, hidden_size)
		self.fc3 = nn.Linear(hidden_size, hidden_size)
		self.fc4 = nn.Linear(hidden_size, hidden_size)
		self.fc5 = nn.Linear(hidden_size, 1)

	def forward(self, x):
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = F.relu(self.fc3(x))
		x = F.relu(self.fc4(x))
		x = F.relu(self.fc5(x))
		return x


class Experiment:

	def __init__(self, conf):
		self.scene = "Aloe"
		self.batch_size = 10000
		self.epochs = 1
		self.lr = 0.1
		self.hidden_size = 500
		self.device = "cpu"

		# read images of scene
		self.left_image, self.right_image = data.get_scene_images(self.scene)
		self.h, self.w, _ = self.left_image.shape
		# get GT cost volume for images
		self.cost_volume = \
			graph_cuts.unary_cost_colors_features_combined(right_image=self.right_image,
														   left_image=self.left_image,
														   **dict({"colored_scale" : 3,
																   "features_scale" : 0.3,
																   "model": "vgg"}))
		self.max_disp = self.cost_volume.shape[-1]
		self.cost_volume = torch.tensor(self.cost_volume).reshape(self.h, self.w, self.max_disp)
		# generate a dataset from the cost volume
		self.dataset = CostDataGenerator(cost_volume=self.cost_volume)
		self.data_loader = torch.utils.data.DataLoader(dataset=self.dataset, batch_size=self.batch_size, shuffle=True)
		# model and training parameters
		self.model = LinearNet(hidden_size=self.hidden_size)
		self.criterion = torch.nn.MSELoss()
		self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)

	def train(self):
		for epoch in tqdm(range(self.epochs)):
			for input_coordinates, output_costs in self.data_loader:
				print("Drawn")
				# Move to device
				input_coordinates, output_costs = input_coordinates.to(device=self.device), output_costs.to(device=self.device)
				# Run the model on the input batch
				pred_costs = self.model(input_coordinates)
				# Calculate the loss for this batch
				loss = self.criterion(pred_costs, output_costs)
				wandb.log({"loss": loss})
				# Update gradients
				self.optimizer.zero_grad()
				backward(loss)
				self.optimizer.step()
