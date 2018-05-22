"""solve_multiprocessing.py

Jinzhe Zhang
A99000241

This module provides a function to solve for a given problem size multiple times
with all available processors.
"""
import numpy as np
import multiprocessing
from solver import Solver

def _initialize_worker(height, width):
	"""Initialize a solver for each worker process

	Args:
	    height (int): The height of the problem size
	    width (int): The width of the problem size
	"""
	global solver
	solver = Solver(height, width)

def _solve_once_multiprocessing(_):
	"""Run the solver once and return the result

	Args:
	    _ (Any Type): Ignored. This is the input value from imap_unordered

	Returns:
	    int: The number of towers to provide full coverage
	"""
	global solver
	return solver.solve_once()

def solve_multiprocessing(height, width, times):
	"""Solve for a given problem size multiple times with all processors.

	Args:
	    height (int): The height of the problem size
	    width (int): The width of the problem size
	    times (int): The number of times to repeat solving the problem

	Returns:
	    np.array: An array of results
	"""
	pool = multiprocessing.Pool(initializer=_initialize_worker,
		initargs=(height, width))
	results = np.empty(times, dtype=np.int)
	for index, result in enumerate(pool.imap_unordered(_solve_once_multiprocessing, xrange(times))):
		results[index] = result
	pool.close()
	return results
