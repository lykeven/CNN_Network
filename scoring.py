#!/usr/bin/env python

"""scoring.py: Script that demonstrates the multi-label classification used."""

import os
import numpy
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MultiLabelBinarizer
from itertools import izip
from sklearn.metrics import f1_score
from scipy.io import loadmat
from scipy.sparse import csc_matrix
from sklearn.utils import shuffle as skshuffle

from collections import defaultdict
from gensim.models import Word2Vec

class TopKRanker(OneVsRestClassifier):
    def predict(self, X, top_k_list):
        assert X.shape[0] == len(top_k_list)
        probs = numpy.asarray(super(TopKRanker, self).predict_proba(X))
        all_labels = []
        for i, k in enumerate(top_k_list):
            probs_ = probs[i, :]
            labels = self.classes_[probs_.argsort()[-k:]].tolist()
            all_labels.append(labels)
        return all_labels

def transformlabel(labels, classes):
    y = numpy.zeros((len(labels), classes), dtype=int)
    for i, line in enumerate(labels):
        for e in line:
            y[i, e] = 1
    return y


def multi_label_classification(embeddings_file, label_file):
    m = 0
    d = 0
    c = 0
    fp = open(embeddings_file)
    for i, line in enumerate(fp):
        seg = line.strip().split(' ')
        if i == 0:
            m = int(seg[0])
            d = int(seg[1])
            c = int(seg[2])
            break

    features_matrix = numpy.zeros((m, d), dtype=float)
    for i, line in enumerate(fp):
        segs = line.strip().split(' ')
        index = int(segs[0])
        segs = segs[1:]
        if i != 0:
            for j in range(d):
                features_matrix[index, j] = float(segs[j])

    fp = open(label_file)
    labels_matrix = numpy.zeros((m, c), dtype=int)
    for i, line in enumerate(fp):
        segs = line.strip().split(' ')
        for j in range(c):
            labels_matrix[i, j] = int(segs[j])

    labels_matrix = csc_matrix(labels_matrix)

    # 2. Shuffle, to create train/test groups
    shuffles = []
    number_shuffles = 5
    for x in range(number_shuffles):
        shuffles.append(skshuffle(features_matrix, labels_matrix))

    # 3. to score each train/test group
    all_results = defaultdict(list)

    training_percents = [0.1* i for i in range(1,10)]
    # uncomment for all training percents
    #training_percents = numpy.asarray(range(1,10))*.1
    for train_percent in training_percents:
        for shuf in shuffles:
            X, y = shuf

            training_size = int(train_percent * X.shape[0])
            X_train = X[:training_size, :]
            y_train_ = y[:training_size]
            y_train = [[] for x in xrange(y_train_.shape[0])]

            cy =  y_train_.tocoo()
            for i, j in izip(cy.row, cy.col):
                y_train[i].append(j)

            assert sum(len(l) for l in y_train) == y_train_.nnz

            X_test = X[training_size:, :]
            y_test_ = y[training_size:]

            y_test = [[] for x in xrange(y_test_.shape[0])]

            cy =  y_test_.tocoo()
            for i, j in izip(cy.row, cy.col):
                y_test[i].append(j)
            y_train = transformlabel(y_train, c)
            clf = TopKRanker(LogisticRegression())
            clf.fit(X_train, y_train)

            # find out how many labels should be predicted
            top_k_list = [len(l) for l in y_test]
            preds = clf.predict(X_test, top_k_list)
            preds = transformlabel(preds, c)
            y_test = transformlabel(y_test, c)
            results = [0] * 2
            results[0] = f1_score(y_test,  preds, average="micro")
            results[1] = f1_score(y_test,  preds, average="macro")
            all_results[train_percent].append(results)

    print 'Results of ', embeddings_file
    print '-------------------'
    for train_percent in sorted(all_results.keys()):
        micro_f1 = sum([a[0] for a in all_results[train_percent]])/len(all_results[train_percent])
        macro_f1 = sum([a[1] for a in all_results[train_percent]])/len(all_results[train_percent])
        print '%.2f: micro_f1:%.4f\tmacro_f1:%.3f' %(train_percent, micro_f1, macro_f1)


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.realpath(__file__)) )
    embeddings_file = "embeddings"
    label_file = "label"
    num_algorithm = 3
    # for i in range(num_algorithm):
    #     multi_label_classification(embeddings_file + str(i+1), label_file + str(i+1))
    multi_label_classification(embeddings_file + "1", label_file + "1")
   

