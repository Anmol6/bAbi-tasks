
from __future__ import absolute_import
import os
import re
import numpy as np


def load_glove(dim):

    glove = {}
    with open("glove.6B/glove.6B."+str(dim) + "d.txt") as file:
        for line in file:
            l = line.split()
            glove[l[0]] = map(float, l[1:])

    print "loaded glove vectors"
    return glove


def open_files(direc, task_id):
	files = os.listdir(direc)
	files = [os.path.join(direc, f) for f in files]
	s = 'qa{}_'.format(task_id)
	train_file = [tf in files if s in tf and 'train' in tf][0]
	test_file = [tf in files if s in tf and 'test' in tf][0]

	return parse_stories(open(train_file).readlines()), parse_stories(open(test_file).readlines())


def parse_stories(file):
   	data = []
    # read_file(file.read())
    for line in file:
        # story = []
        line = line.strip()
        idx, line = line.split(' ', 1)
        line = line.replace('.', ' . ')
        if (idx == 1):
            story = []

        if "\t" in line:
            q, a = line.split("\t", 1)
            a, ida = a.split(' ', 1)
            data.append((q, a, story))

        else:
            substory = line.split(' ')
            story.append(substory)

    return data


def vectorize_data(data, word2idx, sentence_length, buffer_size):
    Q = []
    A = []
    S = []

    for q, a, story in data:
        ss = []
        for substory in story:
            extra_lens = max(0, sentence_length - len(substory))
            sss = [word2idx[word] for word in substory] + [0]*extra_lens
            ss.append(sss)

        ss = ss[::-1][:buffer_size][::-1]
        ss_r = max(0, sentence_length - len(ss))
        for _ in range( ss_r ):
        	ss.append([0]*sentence_length)

        extra_lenq = max(0, sentence_length - len(q))
            q.append([word2idx[word] for word in q] + [0]*extra_lenq)

            ans = np.zeros(len(word2idx)+1)
            ans[word2idx[a]] = 1

            Q.append(q)
            A.append(ans)
            S.append(ss)

    return np.array(S), np.array(Q), np.array(A)
