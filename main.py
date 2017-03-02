# -*- coding: utf-8 -*-
__author__ = 'keven'

import numpy as np
import os, sys, timeit

from data import node_selection, read_graph
from model import build_cnn
from scoring import multi_label_classification
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


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
	parser.add_argument('--negative', default=5, type=int, help='Number of negative sampling edges')
	parser.add_argument('--iteration', default=10, type=int, help='Number of iteration')
	parser.add_argument('--alpha', default=0.1, type=float, help='learning rate for SGD')
	parser.add_argument('--num_kernel', nargs='+', type=int, help='Number of channel in convolutional Layers')
	parser.add_argument('--kernel_size', nargs='+', type=int, help='Size of kernel in convolutional Layers')
	parser.add_argument('--dimension', default=100, type=int, help='Dimension of Output Embedding')
	args = parser.parse_args()

	os.chdir(os.path.dirname(os.path.realpath(__file__)))

	label_matrix_file = 'temp_label.txt'
	start_time_1 = timeit.default_timer()
	# features, edges, neg_edges, node2id = read_graph(args.input, args.neighbor, args.negative)
	features, edges, neg_edges, node2id = node_selection(args.input, 2, args.neighbor, args.negative)
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
