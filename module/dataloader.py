#!/usr/bin/python
# -*- coding: utf-8 -*-

# __author__="Danqing Wang"

#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""This file contains code to read the train/eval/test data from file and process it, and read the vocab data from file and process it"""

import re
import os
from nltk.corpus import stopwords

from itertools import cycle
import glob
import copy
import random
import time
import json
import pickle
import nltk
import collections
from collections import Counter
from itertools import combinations
import numpy as np
from random import shuffle

import torch
import torch.utils.data
import torch.nn.functional as F

from tools.logger import logger

import dgl
from dgl.data.utils import save_graphs, load_graphs


class IterDataset(torch.utils.data.IterableDataset):
    """ Constructor: Dataset of example(object) for single document summarization"""

    def __init__(self, hps):
        """ Initializes the IterDataset with the path of data
        """
        self.hps = hps

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            num_workers, worker_id = 1, 0
        else:
            num_workers, worker_id = worker_info.num_workers, worker_info.id
        graph_dir = os.path.join(self.hps.cache_dir, 'graph')
        return ExampleSet(num_workers, worker_id, graph_dir, self.hps)


class ExampleSet():
    def __init__(self, num_workers, worker_id, graph_dir, hps):
        self.num_workers = num_workers
        self.worker_id = worker_id
        self.graph_dir = graph_dir
        self.graph_data_folder_num = len([f for f in os.listdir(graph_dir) if 'train' in f])
        self.folder = None
        self.hps = hps
        self.folder_i = -1
        self.data_no = -1
        self.folder_records = -1
        logger.info("[INFO] itering train dataset to %d" % hps.start_iteration)
        start_index = hps.start_iteration * hps.batch_size
        while start_index > 0:
            self.folder_i += 1
            if self.folder_i > self.graph_data_folder_num:
                self.folder_i = 0
            self.folder_records = len(os.listdir(self.folder))
            if start_index >= self.folder_records:
                start_index -= self.folder_records
        self.graph_i = start_index - 1
        logger.info("[INFO] starting at: data_no=%d, folder_i=%d, graph_i=%d" % (self.data_no, self.folder_i, self.graph_i))

    def __next__(self):
        time1 = time.time()
        while True:
            self.data_no += 1
            self.graph_i += 1
            while self.graph_i >= self.folder_records:
                self.folder_i += 1
                self.graph_i = 0
                if self.folder_i >= self.graph_data_folder_num:
                    raise StopIteration
                self.folder = os.path.join(self.graph_dir, 'train'+str(self.folder_i))
                self.folder_records = len(os.listdir(self.folder))
            if self.data_no % self.num_workers == self.worker_id:
                break
        time2 = time.time()
        logger.debug('[DEBUG] dataloader %d start reading graph folder %d, file %d. using time %.5f' % (self.worker_id, self.folder_i, self.graph_i, time2-time1))
        try:
            graphs, labels = load_graphs(os.path.join(self.folder, str(self.graph_i)+'.bin'))
            graph = graphs[0]
            # filter erroneous graphs which may cause exception
            graph.filter_nodes(lambda nodes: nodes.data["dtype"] == 1)
            graph.filter_nodes(lambda nodes: nodes.data["unit"] == 1)
            if self.hps.model == 'HDSG':
                graph.filter_nodes(lambda nodes: nodes.data["extractable"] == 1)
            return graph, self.data_no
        except Exception as e:
            logger.warning('[WARNING] dataloader %d failed reading graph folder %d, file %d.' % (self.worker_id, self.folder_i, self.graph_i))
            logger.warning(str(e))
            return self.__next__()


class Example():
    def __init__(self, summary, ori_text):
        self.original_abstract = "\n".join(summary)
        if isinstance(ori_text, list) and isinstance(ori_text[0], list):
            self.original_article_sents = []
            for chap in ori_text:
                self.original_article_sents.extend(chap)
        else:
            self.original_article_sents = ori_text


class MapDataset(torch.utils.data.Dataset):
    """ Constructor: Dataset of example(object) for single document summarization"""

    def __init__(self, hps, mode='val'):
        self.hps = hps
        self.mode = mode
        self.size = len(os.listdir(os.path.join(hps.cache_dir, 'graph', mode)))
        self.example_list = readJson(os.path.join(hps.data_dir, mode+'.json'))

    def get_example(self, index):
        e = self.example_list[index]
        return Example(e['summary'], e['ori_text'])

    def __getitem__(self, index):
        """
        :param index: int; the index of the example
        :return
            G: graph for the example
            index: int; the index of the example in the dataset
        """
        try:
            graphs, labels = load_graphs(os.path.join(self.hps.cache_dir, 'graph', self.mode, str(index)+'.bin'))
            graph = graphs[0]
            # filter erroneous graphs which may cause exception
            graph.filter_nodes(lambda nodes: nodes.data["dtype"] == 1)
            graph.filter_nodes(lambda nodes: nodes.data["unit"] == 1)
            if self.hps.model == 'HDSG':
                graph.filter_nodes(lambda nodes: nodes.data["extractable"] == 1)
            return graph, index
        except Exception as e:
            logger.warning('[WARNING] failed reading graph %d.' % (index))
            logger.warning(str(e))
            if index < self.size - 1:
                return self.__getitem__(index + 1)
            else:
                return self.__getitem__(0)

    def __len__(self):
        return self.size


def readJson(fname):
    data = []
    with open(fname, encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def graph_collate_fn(samples):
    '''
    :param batch: (G, input_pad)
    :return: 
    '''
    graphs, index = map(list, zip(*samples))
    graph_len = [len(g.filter_nodes(lambda nodes: nodes.data["dtype"] == 1)) for g in graphs]  # sent node of graph
    sorted_len, sorted_index = torch.sort(torch.LongTensor(graph_len), dim=0, descending=True)
    batched_graph = dgl.batch([graphs[idx] for idx in sorted_index])
    return batched_graph, [index[idx] for idx in sorted_index]
