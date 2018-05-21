
import weakref
import numpy as np

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

		if not 0 <= x1 < x2 < solver.width:
			raise ValueError ("Invalid x coordinates")
		if not 0 <= y1 < y2 < solver.height:
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
		return np.s_[self.y1:self.y2+1, self.x1:self.x2+1]

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
