import numpy as np

# Revised from: https://github.com/rlcode/per

class SumTree:
	write = 0
	'''
	tree index management
	    0    <- sum of all priority
	  1  2
	 3 4 5 6 <- priority data
	'''
	def __init__(self, capacity):
		self.capacity = capacity
		self.tree = np.zeros(2 * capacity -1)
		self.data = np.zeros(capacity, dtype=object)
		self.n_entries = 0
	
	# update to the root node
	def _propagate(self, idx, change):
		parent = (idx - 1) // 2
		self.tree[parent] += change
		if parent != 0: # propagate until meet the root node (with index 0)
			self._propagate(parent, change)
	
	# find sample on leaf node
	'''
	if the target is smaller than the left node, keep search in the left subtree, 
	else search in the right subtree with target subtract the data of the left node
	until you can't traverse deeper
	'''
	def _retrieve(self, idx, target):
		left = 2 * idx + 1
		right = left + 1
		if left >= len(self.tree):
			return idx
		if target <= self.tree[left]:
			return self._retrieve(left, target)
		else:
			return self._retrieve(right, target - self.tree[left])
	
	@property
	def total(self):
		return self.tree[0]
	
	@property
	def length(self):
		return self.n_entries
		
	# store priority and sample
	def add(self, p, data):
		idx = self.write + self.capacity - 1
		self.data[self.write] = data
		# print("[Memory] Index {} with priority {} add to tree".format(idx, p))
		self.update(idx, p)
		self.write += 1
		if self.write >= self.capacity:
			self.write = 0 # clear to zero (queue)
		if self.n_entries < self.capacity:
			self.n_entries += 1
	
	# update priority
	def update(self, idx, p):
		# print(self.tree.shape, idx)
		change = p - self.tree[idx]
		self.tree[idx] = p # Set leaf node priority value
		self._propagate(idx, change) # Update parent nodes value until root node
		
	# get priority and sample
	def get(self, target):
		idx = self._retrieve(0, target)
		dataIdx = idx - self.capacity + 1
		return idx, self.tree[idx], self.data[dataIdx]
		
	def reset_priority(self):
		for i in range(self.write):
			idx = i + self.capacity - 1
			self.update(idx, 1.0)
