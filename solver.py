
import random
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from tower import Tower

class Solver:
	"""docstring for Solver"""
	def __init__(self, height, width):
		"""Initialize a solver

		Args:
		    height (int): Height of the rectangle
		    width (int): Width of the rectangle
		"""
		self.height = int(height)
		self.width = int(width)

		self.coverage = np.zeros((self.height, self.width), dtype=np.uint)
		self.num_tower = 0
		self.tower_list = []

	def create_tower(self, x1, x2, y1, y2):
		"""Create a tower instance for the solver

		Args:
		    x1 (int): x1 coordinate
		    x2 (int): x2 coordinate
		    y1 (int): y1 coordinate
		    y2 (int): y2 coordinate

		Returns:
		    Tower: A Tower instance
		"""
		# Argument checks are done by Tower.__init__
		return Tower(self, x1, x2, y1, y2)

	def create_random_tower(self):
		"""Create a random tower. Sizes are uniformly distributed, coordinates
		are uniformly distributed.

		Returns:
		    Tower: The generated tower
		"""
		# Get a random width
		width = random.randint(1, self.width)
		# Get a random x1
		x1 = random.randint(0, self.width - width)
		# Get x2
		x2 = x1 + width

		# Get a random height
		height = random.randint(1, self.height)
		# Get a random y1
		y1 = random.randint(0, self.height - height)
		# Get y2
		y2 = y1 + height

		return self.create_tower(x1, x2, y1, y2)

	def add_tower(self, tower):
		"""Add a tower to the solution. Assign a rank to it and update solver's
		states.

		Args:
		    tower (Tower): The tower to be added

		Raises:
		    RuntimeError: Varies reasons blocking the tower to be added
		    TypeError: Wrong argument types
		"""
		if not isinstance(tower, Tower):
			raise TypeError ("tower %s is not a Tower object" % tower)

		if not tower.is_for(self):
			raise RuntimeError ("The tower %s was not created for the solver %s"
				% (tower, self))

		if np.any(self.coverage[tower.mask]):
			raise RuntimeError ("New tower overlaps with the current coverage")

		self.num_tower += 1
		tower.rank = self.num_tower
		self.tower_list.append(tower)
		self.coverage[tower.mask] = tower.rank

	def plot(self):
		plt.imshow(self.coverage, 'hot', vmax=8)
		plt.colorbar()


