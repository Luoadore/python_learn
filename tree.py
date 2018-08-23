# coding: utf-8

# 树的先中后序遍历，深度优先遍历，广度优先遍历
class tree:
	def __init__(self, x):
		self.val = x
		self.left = None
		self.right = None

class tree_op:

	def preOrder_traversal(self, root):
		l = []
		if root != None:
			l.extend(root.val)
			l.extend(self.preOrder_traversal(root.left))
			l.extend(self.preOrder_traversal(root.right))
		return l

	def midOrder_traversal(self, root):
		l = []
		if root != None:
			l.extend(self.midOrder_traversal(root.left))
			l.extend(root.val)
			l.extend(self.midOrder_traversal(root.right))
		return l

	def laterOrder_traversal(self, root):
		l = []
		if root != None:
			l.extend(self.laterOrder_traversal(root.left))
			l.extend(self.laterOrder_traversal(root.right))
			l.extend(root.val)
		return l

	def depth_search(self, root):
		stack = []
		l = []
		stack.append(root)
		while len(stack) != 0:
			node = stack.pop()
			l.append(node.val)
			if node.right != None:
				stack.append(node.right)
			if node.left != None:
				stack.append(node.left)
		return l

	def width_search(root):
		queue = []
		l = []
		queue.append(root)
		while len(queue) != 0:
			node = queue.pop(0)
			l.append(node.val)
			if node.left != None:
				queue.append(node.left)
			if node.right:
				queue.append(node.right)
		return l

