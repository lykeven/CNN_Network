# -*- coding: utf-8 -*-
__author__ = 'keven'

import numpy as np
import networkx as nx
import os, sys, timeit, random
from scipy.spatial import distance

import theano
import theano.tensor as T
from theano.tensor.signal import pool
from theano.tensor.nnet import conv2d

from scoring import multi_label_classification
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

sys.setrecursionlimit(16000)


def read_graph(filename, K, T):
	"""Read graph from file and construct features for cnn"""
	with open(filename, 'rb') as f:
		G = nx.read_weighted_edgelist(f)
	node2id = dict([(node, vid) for vid, node in enumerate(G.nodes())])
	id2node = dict(zip(node2id.values(), node2id.keys()))
	n = len(id2node)
	# edges = np.asarray(nx.adjacency_matrix(G).todense())
	# edges = np.asarray((np.eye(n) - 0.1 * np.mat(edges)).I - np.eye(n))
	edges = np.asarray([(node2id[s], node2id[t]) for s, t in G.edges()], dtype=np.int).T
	neg_edges = []
	X = np.zeros((n, 2, K), dtype=np.double)
	# obtain normalized neighbors for each vertex
	for i in range(n):
		neighbors = G.neighbors(id2node[i])
		neighbors_dict = dict(zip(neighbors, [1] * len(neighbors)))
		neighbors_dict[id2node[i]] = 0
		for j in range(T):
			index = random.randint(0, n - 1)
			if id2node[index] in neighbors_dict:
				continue
			neg_edges.append((i, index))
		heap = neighbors[:]
		# top K neighbor with BFS strategy
		while len(neighbors_dict) < K:
			node = heap.pop()
			new_neighbors = G.neighbors(node)
			new_neighbors_dict = dict(zip(new_neighbors, [neighbors_dict[node] + 1] * len(new_neighbors)))
			for key, val in new_neighbors_dict.items():
				if key not in neighbors_dict:
					neighbors_dict[key] = val
			heap[0:0] = new_neighbors
		# construct subgraph and compute local structure attributes like degree centrality
		sub_g = G.subgraph(neighbors_dict.keys())
		measurement = nx.degree_centrality(sub_g)
		dis_degree = dict([(v, [dis, 1.0 / measurement[v]]) for v, dis in neighbors_dict.items()])
		sorted_neighbor = sorted(dis_degree.items(), key=lambda d: d[1], reverse=False)[:K]
		norm_sub_g = sub_g.subgraph([v for v, measure in sorted_neighbor])
		sub_degree = norm_sub_g.degree()
		sub_betweenness = nx.degree_centrality(norm_sub_g)
		# construct features as channels by local structure
		X[i, :] = np.asarray([[sub_degree[v], sub_betweenness[v]] for v, measure in sorted_neighbor]).T
	for i in range(2):
		m = np.mean(X[:, i, :])
		std = np.std(X[:, i, :])
		X[:, i, :] = (X[:, i, :] - m) / std
	X = np.reshape(X, (n, 2, 1, K))
	neg_edges = np.asarray(neg_edges, dtype=np.int).T
	return X, edges, neg_edges, node2id


class ConvPoolLayer(object):
	"""Pool Layer of a convolutional network """
	def __init__(self, rng, input, filter_shape, feature_image_shape, poolsize=(1, 2)):
		assert feature_image_shape[1] == filter_shape[1]
		self.input = input
		fan_in = np.prod(filter_shape[1:])
		fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) // np.prod(poolsize))
		# initialize weights with random weights
		W_bound = np.sqrt(6. / (fan_in + fan_out))
		self.W = theano.shared(
			np.asarray(rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
			dtype=theano.config.floatX),
			borrow=True
		)

		# the bias is a 1D tensor -- one bias per output feature map
		b_values = np.zeros((filter_shape[0],), dtype=theano.config.floatX)
		self.b = theano.shared(value=b_values, borrow=True)

		# convolve input feature maps with filters
		conv_out = conv2d(
			input=input,
			filters=self.W,
			filter_shape=filter_shape,
			input_shape=feature_image_shape
		)

		# pool each feature map individually, using maxpooling
		pooled_out = pool.pool_2d(
			input=conv_out,
			ds=poolsize,
			ignore_border=True
		)
		self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
		self.params = [self.W, self.b]
		self.input = input


class HiddenLayer(object):
	def __init__(self, rng, input, n_in, n_out, W=None, b=None, activation=T.tanh):
		self.input = input
		if W is None:
			W_values = np.asarray(
				rng.uniform(
					low=-np.sqrt(6. / (n_in + n_out)),
					high=np.sqrt(6. / (n_in + n_out)),
					size=(n_in, n_out)
				),
				dtype=theano.config.floatX
			)
		if activation == theano.tensor.nnet.sigmoid:
			W_values *= 4

		W = theano.shared(value=W_values, name='W', borrow=True)

		if b is None:
			b_values = np.zeros((n_out,), dtype=theano.config.floatX)
			b = theano.shared(value=b_values, name='b', borrow=True)

		self.W = W
		self.b = b

		lin_output = T.dot(input, self.W) + self.b
		self.output = (
			lin_output if activation is None
			else activation(lin_output)
		)
		# parameters of the model
		self.params = [self.W, self.b]


class NetworkPreserve(object):
	def __init__(self, input, edges, neg_edges):
		# S = T.dot(input, input.T)
		# n = edges.shape[0]
		# # N = S.norm(2, axis=1)
		# N = [0] * n
		# for i in range(n):
		# 	N[i] = T.sqrt(T.sum(input[i] ** 2))
		# N2 = np.zeros((n, n), dtype=float)
		# for i in range(n):
		# 	for j in range(n):
		# 		N2 = 1./ (N[i] * N[j])
		# S2 = S * N2
		# S2 = distance.cdist(input, input, 'cosine')
		# N = np.linalg.norm(input, axis=1)
		# self.loss = T.mean((edges - S2) ** 2)
		# self.loss = T.mean((edges - T.dot(input, input.T)) ** 2)
		# self.loss = T.mean((edges - 1./(1 + T.exp(-T.dot(input, input.T)))) ** 2)
		self.loss = -T.mean(T.log(1. / (1 + T.exp(-T.batched_dot(input[edges[0]], input[edges[1]]))))) - \
					10 * T.mean(T.log(1. / (1 + T.exp(T.batched_dot(input[neg_edges[0]], input[neg_edges[1]])))))

	def first_order_proximity(self):
		return self.loss


def build_cnn(X=None, edges=None, neg_edges=None,alpha=0.1, n_epochs=20, nkerns=[10, 20], kerns_size=[101, 25], dimension=100):
	rng = np.random.RandomState(23455)
	batch_size = X.shape[0]
	n_channels = X.shape[1]
	n_dimension = X.shape[3]
	sec_conv_input_size = (n_dimension - kerns_size[0] + 1) / 2
	sec_conv_output_size = (sec_conv_input_size - kerns_size[1] + 1) / 2
	hidden_input_size = nkerns[1] * sec_conv_output_size
	hidden_output_size = dimension

	x = T.dmatrix('x')
	layer0_input = x.reshape((batch_size, n_channels, 1, n_dimension))

	print('... building the model')
	# construct the first Convolution-Pooling Layer
	layer0 = ConvPoolLayer(
		rng,
		input=layer0_input,
		feature_image_shape=(batch_size, n_channels, 1, n_dimension),
		filter_shape=(nkerns[0], n_channels, 1, kerns_size[0]),
		poolsize=(1, 2)
	)
	# construct the second Convolution-Pooling Layer
	layer1 = ConvPoolLayer(
		rng,
		input=layer0.output,
		feature_image_shape=(batch_size, nkerns[0], 1, sec_conv_input_size),
		filter_shape=(nkerns[1], nkerns[0], 1, kerns_size[1]),
		poolsize=(1, 2)
	)
	# flatten output of second Convolution-Pooling Layer to 2D features
	layer2_input = layer1.output.flatten(2)

	# construct a fully-connected layer
	layer2 = HiddenLayer(
		rng,
		input=layer2_input,
		n_in=hidden_input_size,
		n_out=hidden_output_size,
		activation=T.tanh
	)

	# compute loss of preserve Network property of first-order proximity
	layer3 = NetworkPreserve(input=layer2.output, edges=edges, neg_edges=neg_edges)
	cost = layer3.first_order_proximity()

	# create a list of all model parameters to be fit by gradient descent
	params = layer2.params + layer1.params + layer0.params
	grads = T.grad(cost, params)

	updates = [
		(param_i, param_i - alpha * grad_i)
		for param_i, grad_i in zip(params, grads)
	]

	train_model = theano.function(
		[layer0_input], [cost, layer2.W, layer2.output],
		updates=updates
	)

	print('... training')
	# training model and print progress
	embeddings = np.zeros((batch_size, dimension), dtype=float)
	prev = None
	for epoch in range(n_epochs):
		cost, params, embeddings = train_model(X)
		product = embeddings.dot(embeddings.T)
		prev = params
		if epoch % 2 == 0:
			print 'Iter:%d, cost:%.8f' % (epoch, cost)
	return embeddings


def write_file(embeddings, node2id, label_file, embedding_file, label_matrix_file):
	m = embeddings.shape[0]
	d = embeddings.shape[1]
	label_map = dict()
	label_list = []
	fp = open(label_file)
	for i, line in enumerate(fp):
		segs = line.strip().split(' ')
		label = int(segs[1])
		if label not in label_map:
			label_map[label] = len(label_map)
		label_list.append((node2id[segs[0]], label_map[label]))
	c = len(label_map)
	label_matrix = np.zeros((m, c), dtype=int)
	for vid, label in label_list:
		label_matrix[vid, label] = 1
	# write embeddings into file
	f = open(embedding_file, 'w')
	f.write(str(m) + " " + str(d) + " " + str(c) + "\n")
	for i in range(m):
		f.write(str(i) + ' ' + ' '.join(str(s) for s in embeddings[i, :].tolist()) + '\n')
	f.close()
	# write label_matrix into file
	f2 = open(label_matrix_file, 'w')
	for i in range(m):
		f2.write(' '.join(str(s) for s in label_matrix[i, :].tolist()) + '\n')
	f2.close()


def main():
	parser = ArgumentParser('CNN_Node', formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler='resolve')
	parser.add_argument('--input', required=True, help='Input graph file')
	parser.add_argument('--label', required=True, help='Input label file')
	parser.add_argument('--output', required=True, help='Output Embedding file')
	parser.add_argument('--neighbor', default=200, type=int, help='Number of neighbor in constructing features')
	parser.add_argument('--negative', default=200, type=int, help='Number of negative sampling edges')
	parser.add_argument('--iteration', default=10, type=int, help='Number of iteration')
	parser.add_argument('--alpha', default=0.1, type=float, help='learning rate for SGD')
	parser.add_argument('--num_kernel', nargs='+', type=int, help='Number of channel in convolutional Layers')
	parser.add_argument('--kernel_size', nargs='+', type=int, help='Size of kernel in convolutional Layers')
	parser.add_argument('--dimension', default=100, type=int, help='Dimension of Output Embedding')
	args = parser.parse_args()

	os.chdir(os.path.dirname(os.path.realpath(__file__)))
	# for path in [args.input, args.label, args.output]:
	# 	os.chdir(os.path.dirname(os.path.abspath(path)))

	label_matrix_file = 'temp_label.txt'
	start_time_1 = timeit.default_timer()
	features, edges, neg_edges, node2id = read_graph(args.input, args.neighbor, args.negative)
	end_time_1 = timeit.default_timer()
	print 'Run for constructing %.2fs' % (end_time_1 - start_time_1)
	start_time_2 = timeit.default_timer()
	embeddings = build_cnn(X=features, edges=edges, neg_edges=neg_edges, alpha=args.alpha, n_epochs=args.iteration,
						nkerns=args.num_kernel, kerns_size=args.kernel_size, dimension=args.dimension)
	end_time_2 = timeit.default_timer()
	print 'Run for training %.2fs' % (end_time_2 - start_time_2)
	start_time_3 = timeit.default_timer()
	write_file(embeddings, node2id, args.label, args.output, label_matrix_file)
	multi_label_classification(args.output, label_matrix_file)
	end_time_3 = timeit.default_timer()
	print 'Run for testing %.2fs' % (end_time_3 - start_time_3)

if __name__ == '__main__':
	sys.exit(main())
