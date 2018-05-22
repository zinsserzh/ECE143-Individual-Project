"""solver.py

Jinzhe Zhang
A99000241

Provides Solver class as a wrapper/container/handle for the entire problem.

"""
import random
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from tower import Tower

class Solver:
	"""docstring for Solver

	Attributes:
	    coverage (np.ndarray): The current coverage. 0 means uncovered, other
	                           values means covered by the tower of that rank
	    height (int): The height of the problem size
	    width (int): The width of the problem size
	    num_tower (int): Number of online towers
	    tower_list (list): A list of online towers
	"""
	def __init__(self, height, width):
		"""Initialize a solver

		Args:
		    height (int): Height of the rectangle
		    width (int): Width of the rectangle
		"""
		self.height = int(height)
		self.width = int(width)

		self.clear()

	def clear(self):
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

	def dump_towers(self):
		"""Render a string the rank and coordiantes of the towers"""
		return "\n".join(tower.dump() for tower in self.tower_list)

	def generate_random_tower(self):
		"""Generate a random tower. Sizes are uniformly distributed, coordinates
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

	def generate_random_valid_tower_untrimmed(self):
		"""Generate a random tower whose area is not totally covered yet.

		Returns:
		    Tower: The generated tower

		Raises:
		    RuntimeError: The entire space has been covered
		"""
		if np.all(self.coverage):
			raise RuntimeError ("The entire space has been covered")

		tower = self.generate_random_tower()
		while np.all(self.coverage[tower.mask]):
			tower = self.generate_random_tower()
		return tower

	def generate_random_valid_tower_trimmed(self):
		"""Generate a random tower whose area is not totally covered yet, and
		then trim its coverage.

		Returns:
		    Tower: The generated tower
		"""
		tower = self.generate_random_valid_tower_untrimmed()
		tower.trim()
		return tower

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

	def plot_coverage(self, im=None):
		"""Plot the current coverage in a binary form.
		Black means uncovered.
		Dark red means covered.

		Args:
		    im (AxiImage, optional): If set, update the data instead of creating
		                             a new plot.

		Returns:
		    AxiImage: The figure used to plot
		"""
		data = self.coverage > 0
		if im:
			im.set_data(data)
		else:
			im = plt.imshow(data, 'hot', vmin=0, vmax=4)
		return im

	def plot_coverage_history(self, im=None):
		"""Plot the current coverage with different color on each tower.

		Args:
		    im (AxiImage, optional): If set, update the data instead of creating
		                             a new plot.

		Returns:
		    AxiImage: The figure used to plot
		"""
		data = self.coverage.copy()
		data[data != 0] += 8
		vmax = max(8, np.max(data))
		data[data == vmax] += 8
		vmax += 8
		if im:
			im.set_data(data)
			im.norm.vmax = vmax
		else:
			im = plt.imshow(data, 'hot', vmin=0, vmax=vmax)
		return im

	def plot_coverage_overlay(self, tower_high=None, tower_low=None, im=None):
		"""Plot the current coverage with two extra towers overlayed.

		Args:
		    tower_high (Tower, optional): Tower that will be strongly highlighted.
		    tower_low (Tower, optional): Tower that will be lightly highlighted.
		    im (AxiImage, optional): If set, update the data instead of creating
		                             a new plot.

		Returns:
		    AxiImage: The figure used to plot
		"""
		data = np.zeros_like(self.coverage)
		data[self.coverage != 0] += 2
		if tower_high:
			data[tower_high.mask] += 7
		if tower_low:
			data[tower_low.mask] += 1
		if im:
			im.set_data(data)
		else:
			im = plt.imshow(data, 'hot', vmin=0, vmax=8)
		return im

	def solve_once(self):
		"""Solve the problem once, return the number of towers.

		Returns:
		    int: Number of tower used to cover the entire space.
		"""
		self.clear()
		while not np.all(self.coverage):
			self.add_tower(self.generate_random_valid_tower_trimmed())
		return self.num_tower

	def solve(self, times=1):
		"""Solve the problem for multiple times, return a list of results

		Args:
		    times (int, optional): Number of times to solve the problem

		Returns:
		    np.array: A list of results (number of towers)
		"""
		result = np.empty(times, dtype=np.int)
		for index in xrange(times):
			result[index] = self.solve_once()
		return result
