"""tower.py

Jinzhe Zhang
A99000241

Provides Tower class to represent towers in the problem. Tower.trim is most
important function.
"""
import weakref
import numpy as np
import matplotlib.pyplot as plt
import itertools

class Tower:

	"""Tower class to represent towers for a specific solver instance

	Attributes:
	    rank (int): The nth tower for the solver. None mean not yet assigned
	    x1 (int): x1 coordinate
	    x2 (int): x2 coordinate
	    y1 (int): y1 coordinate
	    y2 (int): y2 coordinate
	"""

	def __init__(self, solver, x1, x2, y1, y2, rank=None):
		"""Verify coordinates and create a tower instance for a specific solver

		Args:
		    solver (Solver): The solver this tower is created for
		    x1 (int): x1 coordinate
		    x2 (int): x2 coordinate
		    y1 (int): y1 coordinate
		    y2 (int): y2 coordinate
		    rank (int, optional): The nth tower for the solver.

		Raises:
		    TypeError: Wrong argument types
		    ValueError: Invalid arguments
		"""
		self._solver_weakref = weakref.ref(solver)

		if not all(isinstance(coordinate, int) for coordinate in (x1, y1, x2, y2)):
			raise TypeError ("All coordinates should be of type int")

		if not 0 <= x1 < x2 <= solver.width:
			raise ValueError ("Invalid x coordinates")
		if not 0 <= y1 < y2 <= solver.height:
			raise ValueError ("Invalid y coordinates")

		self.x1 = x1
		self.x2 = x2
		self.y1 = y1
		self.y2 = y2

		self.rank = rank

	def __copy__(self):
		"""Create a copy of self

		Returns:
		    Tower: A copy of the tower
		"""
		return Tower(self.solver, self.x1, self.x2, self.y1, self.y2, self.rank)

	def copy(self):
		"""A wrapper to __copy__

		Returns:
		    Tower: A copy of the tower
		"""
		return self.__copy__()

	def dump(self):
		"""Return a string of the rank and coordianates of the current tower

		Returns:
		    TYPE: Description
		"""
		return "Rank: %(rank)s\tx1: %(x1)d\tx2: %(x2)d\ty1: %(y1)d\ty2: %(y2)d" % self.__dict__

	@property
	def mask(self):
		"""Returns the slice object to mask Solver's coverage array

		For example, solver.coverage[tower.mask] returns the coverage area of
		the tower

		Returns:
		    tuple: A index tuple
		"""
		return np.s_[self.y1:self.y2, self.x1:self.x2]

	@property
	def solver(self):
		"""Returns the solver which the tower was created for

		Returns:
		    Solver: The solver which the tower was created for
		"""
		return self._solver_weakref()

	def is_for(self, solver):
		"""Check if the tower was created for the solver

		Args:
		    solver (Solver): The solver to be tested

		Returns:
		    bool: Whether the tower was created for the solver
		"""
		return self.solver is solver

	def trim(self):
		"""Trim the tower's coverage area so that it doesn't overlap with other
		existing tower.

		This function is based on the last O(MN) algorithm found at this site:
		http://www.drdobbs.com/database/the-maximal-rectangle-problem/184410529

		Changes are made to adapt different orientations and indices. Its bugs

		are also fixed:
			1.  When re-pushing the active opening, height0 instead of
				current_height should be used.
			2.  When closing to a opening with the same height, re-pushing is
				not needed.

		Raises:
		    RuntimeError: Tower can not be trimmed. Totally covered.
		"""
		# Get sub-area
		sub_array = self.solver.coverage[self.mask]

		# Verify the sub-area
		if np.all(sub_array):
			raise RuntimeError ("The entire current space has been covered.")

		# Initialize algorithm variables
		cache = np.zeros_like(sub_array[0])
		max_area = -1
		max_coordinates = (None, None, None, None)
		stack = []

		# Main loop
		for yy, row in enumerate(sub_array):
			cache += 1
			cache[row != 0] = 0

			current_height = 0
			for xx, item in enumerate(itertools.chain(cache, (0,))):
				if item > current_height:	# Opening new rectangle(s)?
					stack.append((xx, current_height))
					current_height = item
				elif item < current_height:	# Closing rectangle(s)?
					while item < current_height:
						xx0, height0 = stack.pop()
						area = current_height * (xx - xx0)
						if area > max_area:
							max_area = area
							max_coordinates = (xx0, xx, int(yy - current_height + 1), yy + 1)
						current_height = height0
					if current_height != item:
						current_height = item
						if current_height:	# Popped an active opening?
							stack.append((xx0, height0))

		# Update coordinates
		self.x2 = self.x1 + max_coordinates[1]
		self.x1 = self.x1 + max_coordinates[0]
		self.y2 = self.y1 + max_coordinates[3]
		self.y1 = self.y1 + max_coordinates[2]

		# Verify the output is valid
		assert 0 <= self.x1 < self.x2 <= self.solver.width
		assert 0 <= self.y1 < self.y2 <= self.solver.height

	def trim_animation(self):
		"""A generator to step through trim algorithm and plot the animation.

		Yields:
		    None: Each time when yield happens, the plot it updated.
		"""
		def render_animation(examine=False, new_max=False):
			"""Render one frame of the animation

			These render functions are set to be a nested function so that state
			variables can be accessed without making them global or object's
			attributes. In this manner, the render routine can be isolated to a
			function while the state variables are also isolated within the
			generator.

			This function read the algorithm state variables and render a frame
			of animation.

			Args:
			    examine (bool, optional): True when examining a new rectangle
			    new_max (bool, optional): True when a larger rectangle is found.
			"""
			render_cache()

			if not examine and not new_max:
				render_opening()

			render_examine(examine)

			if new_max:
				render_max()

		def render_cache():
			"""Render im_cache layer
			"""
			cache_data = np.zeros_like(empty_space)
			for x, height in enumerate(cache):
				cache_data[int(yy - height + 1) : yy + 1, x] = 1
			cache_data = np.ma.masked_where(cache_data == 0, cache_data)
			im_cache.set_data(cache_data)

		def render_opening():
			"""Render im_opening layer
			"""
			if stack:
				opening_data = np.zeros_like(empty_space)
				x_list, h_list = (list(t) for t in zip(*stack))
				del h_list[0]
				h_list.append(current_height)
				xh_list = zip(x_list, h_list)
				for x, height in xh_list:
					opening_data[int(yy - height + 1) : yy + 1, x] = 1
				opening_data = np.ma.masked_where(opening_data == 0, opening_data)
				im_opening.set_data(opening_data)
			else:
				im_opening.set_data(np.ma.masked_all_like(empty_space))

		def render_examine(examine):
			"""Render im_examine highlight layer

			Args:
			    examine (bool): True when examining a new rectangle. I.e.
			                    highlight the area.
			"""
			examine_data = np.ma.masked_all_like(empty_space)
			if examine:
				examine_data[int(yy - current_height + 1) : yy + 1, xx0 : xx] = 1
			im_examine.set_data(examine_data)

		def render_max():
			"""Render im_max layer
			"""
			max_data = np.ma.masked_all_like(empty_space)
			if max_coordinates[0] is not None:
				max_data[max_coordinates[2]:max_coordinates[3], max_coordinates[0]:max_coordinates[1]] = 1
			im_max.set_data(max_data)


		# Get subarea
		empty_space = self.solver.coverage[self.mask] == 0

		# Initialize algorithm variables
		cache = np.zeros_like(empty_space[0], dtype=np.uint)
		max_area = -1
		max_coordinates = (None, None, None, None)
		stack = []

		# Initialize animation layers
		im_background = plt.imshow(np.zeros_like(empty_space), 'gray', vmin=0, vmax=1)
		im_empty = plt.imshow(np.ma.masked_where(empty_space == False, empty_space), 'tab10', alpha=0.2, vmin=0, vmax=1)
		im_cache = plt.imshow(np.ma.masked_all_like(empty_space), 'tab10', alpha=0.7, vmin=0, vmax=1)
		im_max = plt.imshow(np.ma.masked_all_like(empty_space), 'Purples', alpha=0.7, vmin=0, vmax=1)
		im_opening = plt.imshow(np.ma.masked_all_like(empty_space), 'spring', alpha=0.7, vmin=0, vmax=1)
		im_examine = plt.imshow(np.ma.masked_all_like(empty_space), 'gray', alpha=1.0, vmin=0, vmax=1)

		# Main loop
		for yy, row in enumerate(empty_space):
			cache += 1
			cache[row == False] = 0

			current_height = 0
			yield render_animation()
			for xx, item in enumerate(itertools.chain(cache, (0,))):
				if item > current_height:	# Opening new rectangle(s)?
					stack.append((xx, current_height))
					current_height = item
					yield render_animation()
				elif item < current_height:	# Closing rectangle(s)?
					while item < current_height:
						xx0, height0 = stack.pop()
						area = current_height * (xx - xx0)
						yield render_animation(examine=True)
						if area > max_area:
							max_area = area
							max_coordinates = (xx0, xx, int(yy - current_height + 1), yy + 1)
							yield render_animation(new_max=True)
						current_height = height0
						yield render_animation()
					if current_height != item:
						current_height = item
						if current_height:	# Popped an active opening?
							stack.append((xx0, height0))
						yield render_animation()
