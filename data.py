# -*- coding: utf-8 -*-
__author__ = 'keven'

import numpy as np
import networkx as nx
import os, sys, timeit, random
import pickle


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


def assemble_neighbor(G, node, num_neighbor, sorted_nodes):
	"""assemble neighbors for node with BFS strategy"""
	neighbors_dict = dict()
	new_neighbors_dict = dict()
	neighbors_dict[node] = 0
	new_neighbors_dict[node] = 0
	# assemble over K neighbors with BFS strategy
	while len(neighbors_dict) < num_neighbor and len(new_neighbors_dict) > 0:
		temp_neighbor_dict = dict()
		for v, d in new_neighbors_dict.items():
			for new_v in G.neighbors(v):
				if new_v not in temp_neighbor_dict:
					temp_neighbor_dict[new_v] = d + 1
		n = len(neighbors_dict)
		for v, d in temp_neighbor_dict.items():
			if v not in neighbors_dict:
				neighbors_dict[v] = d
		new_neighbors_dict = temp_neighbor_dict
		# break if the number of neighbors do not increase
		if n == len(neighbors_dict):
			break

	# add dummy disconnected nodes if number is not suffice
	while len(neighbors_dict) < num_neighbor:
		rand_node = sorted_nodes[random.randint(0, len(sorted_nodes) - 1)]
		if rand_node not in neighbors_dict:
			neighbors_dict[rand_node] = 10
	return neighbors_dict


def node_selection(filename, num_channel, num_neighbor, num_negative):
	"""node selection and construct features for cnn"""
	with open(filename, 'rb') as f:
		G = nx.read_weighted_edgelist(f)
	node2id = dict([(node, vid) for vid, node in enumerate(G.nodes())])
	id2node = dict(zip(node2id.values(), node2id.keys()))
	n = len(id2node)

	features = np.zeros((n, num_channel), dtype=np.double)
	pagerank = nx.pagerank_numpy(G)
	degree = nx.degree_centrality(G)
	features[:, 0] = np.asarray([pagerank[id2node[k]] for k in range(n)])
	features[:, 1] = np.asarray([degree[id2node[k]] for k in range(n)])
	X = np.zeros((n, num_channel, num_neighbor), dtype=np.double)
	edges = np.asarray([(node2id[s], node2id[t]) for s, t in G.edges()], dtype=np.int).T
	neg_edges = []
	# obtain normalized neighbors' features for each vertex
	for i in range(n):
		if i % 100 == 0:
			print "process node %d" % (i,)
		neighbors_dict = assemble_neighbor(G, id2node[i], num_neighbor, node2id.keys())
		for j in range(num_negative):
			index = random.randint(0, n - 1)
			if id2node[index] in neighbors_dict:
				continue
			neg_edges.append((i, index))
		# construct subgraph and sort neighbors with a labeling measurement like degree centrality
		sub_g = G.subgraph(neighbors_dict.keys())
		# measurement = nx.betweenness_centrality(sub_g)
		measurement = nx.degree_centrality(sub_g)
		dis_degree = dict([(v, [dis, 1.0 / (1 + measurement[v])]) for v, dis in neighbors_dict.items()])
		sorted_neighbor = sorted(dis_degree.items(), key=lambda d: d[1], reverse=False)[:num_neighbor]
		# construct receptive field with CNN
		X[i, :] = features[[node2id[v] for v, measure in sorted_neighbor]].T

	for i in range(num_channel):
		m = np.mean(X[:, i, :])
		std = np.std(X[:, i, :])
		X[:, i, :] = (X[:, i, :] - m) / std
	X = np.reshape(X, (n, num_channel, 1, num_neighbor))
	neg_edges = np.asarray(neg_edges, dtype=np.int).T
	return X, edges, neg_edges, node2id


def save_data(features, edges, neg_edges, node2id, file):
	object = dict()
	object["features"] = features
	object["edges"] = edges
	object["neg_edges"] = neg_edges
	object["node2id"] = node2id
	with open(file, 'w') as f:
		pickle.dump(object, f)


def load_data(file):
	with open(file, 'r') as f:
		object = pickle.load(f)
	features = object["features"]
	edges = object["edges"]
	neg_edges = object["neg_edges"]
	node2id = object["node2id"]
	return features, edges, neg_edges, node2id