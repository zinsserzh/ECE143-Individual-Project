
import weakref
import numpy as np
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

	def __init__(self, solver, x1, x2, y1, y2):
		"""Verify coordinates and create a tower instance for a specific solver

		Args:
		    solver (Solver): The solver this tower is created for
		    x1 (int): x1 coordinate
		    x2 (int): x2 coordinate
		    y1 (int): y1 coordinate
		    y2 (int): y2 coordinate

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

		self.rank = None

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
		sub_array = self.solver.coverage[self.mask]

		if np.all(sub_array):
			raise RuntimeError ("The entire current space has been covered.")

		cache = np.zeros_like(sub_array[0])
		max_area = -1
		max_coordinates = (None, None, None, None)
		stack = []

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
							max_coordinates = (xx0, xx, yy - current_height + 1, yy + 1)
						current_height = height0
					if current_height != item:
						current_height = item
						if current_height:	# Popped an active opening?
							stack.append((xx0, height0))

		self.x2 = int(self.x1 + max_coordinates[1])
		self.x1 = int(self.x1 + max_coordinates[0])
		self.y2 = int(self.y1 + max_coordinates[3])
		self.y1 = int(self.y1 + max_coordinates[2])

		assert 0 <= self.x1 < self.x2 <= self.solver.width
		assert 0 <= self.y1 < self.y2 <= self.solver.height

