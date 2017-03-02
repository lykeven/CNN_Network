import graph_tool.all as gt
import numpy as np
import cPickle as cp
import networkx as nx

def read_graph_tool(filename, rate, length):
	"""Read graph from file and construct features for cnn"""
	f = open(filename, 'rb')
	node2id = dict()
	for line in f:
		seg = line.strip().split(' ')
		for v in seg:
			if v not in node2id:
				node2id[v] = len(node2id)
	n = len(node2id)
	g = gt.Graph(directed=False)
	g.add_vertex(n)
	f = open(filename, 'rb')
	for line in f:
		seg = line.strip().split(' ')
		g.add_edge(g.vertex(node2id[seg[0]]), g.vertex(node2id[seg[1]]))
	num = int(rate * n)
	vp, ep = gt.betweenness(g)
	id2node = dict(zip(node2id.values(), node2id.keys()))


def parse_nci(graph_name='nci1.graph', with_structural_features=False):
	path = "../data/"
	if graph_name == 'nci1.graph':
		maxval = 37
	elif graph_name == 'nci109.graph':
		maxval = 38

	with open(path + graph_name, 'r') as f:
		raw = cp.load(f)
		n_classes = 2
		n_graphs = len(raw['graph'])

		A = []
		rX = []
		Y = np.zeros((n_graphs, n_classes), dtype='int32')

		for i in range(n_graphs):
			# Set label
			Y[i][raw['labels'][i]] = 1

			# Parse graph
			G = raw['graph'][i]
			n_nodes = len(G)
			a = np.zeros((n_nodes, n_nodes), dtype='float32')
			x = np.zeros((n_nodes, maxval), dtype='float32')

			for node, meta in G.iteritems():
				x[node, meta['label'][0] - 1] = 1
				for neighbor in meta['neighbors']:
					a[node, neighbor] = 1

			A.append(a)
			rX.append(x)
			g = nx.from_numpy_matrix(A[i])

	if with_structural_features:
		for i in range(len(rX)):
			struct_feat = np.zeros((rX[i].shape[0], 3))
			# degree
			struct_feat[:, 0] = A[i].sum(1)

			G = nx.from_numpy_matrix(A[i])
			# pagerank
			prank = nx.pagerank_numpy(G)
			struct_feat[:, 1] = np.asarray([prank[k] for k in range(A[i].shape[0])])

			# clustering
			clust = nx.clustering(G)
			struct_feat[:, 2] = np.asarray([clust[k] for k in range(A[i].shape[0])])

			rX[i] = np.hstack((rX[i], struct_feat))

	return A, rX, Y


def read_graph(filename, K, T):
	"""Read graph from file and construct features for cnn"""
	with open(filename, 'rb') as f:
		G = nx.read_weighted_edgelist(f)
	node2id = dict([(node, vid) for vid, node in enumerate(G.nodes())])
	id2node = dict(zip(node2id.values(), node2id.keys()))
	n = len(id2node)
	X = np.zeros((n, 2, K), dtype=np.double)
	# obtain normalized neighbors' features for each vertex
	for i in range(n):
		neighbors = G.neighbors(id2node[i])
		neighbors_dict = dict(zip(neighbors, [1] * len(neighbors)))
		neighbors_dict[id2node[i]] = 0
		heap = neighbors[:]
		# assemble over K neighbors with BFS strategy
		while len(neighbors_dict) < K:
			node = heap.pop()
			new_neighbors = G.neighbors(node)
			new_neighbors_dict = dict(zip(new_neighbors, [neighbors_dict[node] + 1] * len(new_neighbors)))
			for key, val in new_neighbors_dict.items():
				if key not in neighbors_dict:
					neighbors_dict[key] = val
			heap[0:0] = new_neighbors
		# construct subgraph and sort neighbors with a labeling measurement like degree centrality
		sub_g = G.subgraph(neighbors_dict.keys())
		measurement = nx.degree_centrality(sub_g)
		dis_degree = dict([(v, [dis, 1.0 / measurement[v]]) for v, dis in neighbors_dict.items()])
		sorted_neighbor = sorted(dis_degree.items(), key=lambda d: d[1], reverse=False)[:K]
		# construct normalized neighbor graph and obtain features as channels for CNN
		norm_sub_g = sub_g.subgraph([v for v, measure in sorted_neighbor])
		sub_degree = norm_sub_g.degree()
		# sub_betweenness = nx.betweenness_centrality(norm_sub_g)
		sub_betweenness = nx.degree_centrality(norm_sub_g)
		X[i, :] = np.asarray([[sub_degree[v], sub_betweenness[v]] for v, measure in sorted_neighbor]).T
	# features normalization
	for i in range(2):
		m = np.mean(X[:, i, :])
		std = np.std(X[:, i, :])
		X[:, i, :] = (X[:, i, :] - m) / std
	X = np.reshape(X, (n, 2, 1, K))
	return X, node2id


def obtain_features(G, features, num_channel, num_sample, num_neighbor, stride):
	"""construct features for cnn"""
	X = np.zeros((num_channel, 1, num_sample, num_neighbor), dtype=np.double)
	node2id = dict([(node, vid) for vid, node in enumerate(G.nodes())])
	id2node = dict(zip(node2id.values(), node2id.keys()))
	betweenness = nx.betweenness_centrality(G)
	sorted_nodes = sorted(betweenness.items(), key=lambda d: d[1], reverse=False)

	i = 0
	j = 0
	# obtain normalized neighbors' features for each vertex
	while j < num_sample:
		if i < len(sorted_nodes):
			node = sorted_nodes[i][0]
			neighbors = G.neighbors(id2node[node])
			neighbors_dict = dict(zip(neighbors, [1] * len(neighbors)))
			neighbors_dict[id2node[node]] = 0
			heap = neighbors[:]
			# assemble over K neighbors with BFS strategy
			while len(neighbors_dict) < num_neighbor:
				node_heap = heap.pop()
				new_neighbors = G.neighbors(node_heap)
				new_neighbors_dict = dict(zip(new_neighbors, [neighbors_dict[node_heap] + 1] * len(new_neighbors)))
				for key, val in new_neighbors_dict.items():
					if key not in neighbors_dict:
						neighbors_dict[key] = val
				heap[0:0] = new_neighbors
			# construct subgraph and sort neighbors with a labeling measurement like degree centrality
			sub_g = G.subgraph(neighbors_dict.keys())
			measurement = nx.degree_centrality(sub_g)
			dis_degree = dict([(v, [dis, 1.0 / measurement[v]]) for v, dis in neighbors_dict.items()])
			sorted_neighbor = sorted(dis_degree.items(), key=lambda d: d[1], reverse=False)[:num_neighbor]
			# construct receptive field with CNN
			X[:, 0, j, :] = features[[v for v, measure in sorted_neighbor]].T
		else:
			# zero receptive field
			X[:, 0, j, :] = np.zeros((num_channel, num_neighbor), dtype=np.double)
		i += stride
		j += 1
	return X.reshape(num_channel, 1, num_sample * num_neighbor)