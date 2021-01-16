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

import os
import time
import json
import numpy as np
import torch
import torch.utils.data

from tools.logger import logger

import dgl
from dgl.data.utils import load_graphs


class IterDataset(torch.utils.data.IterableDataset):
    """ Constructor: Dataset of example(object) for single document summarization"""

    def __init__(self, hps):
        """ Initializes the IterDataset with the path of data
        """
        self.hps = hps
        self.first_iter = True

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            num_workers, worker_id = 1, 0
        else:
            num_workers, worker_id = worker_info.num_workers, worker_info.id
        graph_dir = os.path.join(self.hps.cache_dir, 'graph')
        if self.first_iter:
            self.first_iter = False
            logger.info("init the train dataset for the first time")
            # need to fast forward training data if the training procesjust begins
            return ExampleSet(num_workers, worker_id, graph_dir, self.hps, fast_woward=True)
        else:
            logger.info("init the train dataset not for the first time")
            # else, no need to ff data, start from the first data
            return ExampleSet(num_workers, worker_id, graph_dir, self.hps)


class ExampleSet():
    def __init__(self, num_workers, worker_id, graph_dir, hps, fast_woward=False):
        self.num_workers = num_workers
        self.worker_id = worker_id
        self.graph_dir = graph_dir
        self.graph_data_folder_num = len([f for f in os.listdir(graph_dir) if 'train' in f])
        self.folder = None
        self.hps = hps
        self.folder_i = -1
        self.data_no = 0
        self.folder_records = -1
        start_index = 0
        if fast_woward:
            start_index = hps.start_iteration * hps.batch_size
            logger.info("[INFO] fast-fowarding train dataset to %d" % start_index)
            while start_index > 0:
                self.folder_i += 1
                if self.folder_i >= self.graph_data_folder_num:
                    self.folder_i = 0
                self.folder = os.path.join(self.graph_dir, 'train'+str(self.folder_i))
                self.folder_records = len(os.listdir(self.folder))
                logger.info("[INFO] fast-forwarding data, folder %d has %d files" % (self.folder_i, self.folder_records))
                if start_index >= self.folder_records:
                    start_index -= self.folder_records
                    self.data_no += self.folder_records
                else:
                    break
        self.graph_i = start_index - 1
        self.data_no += self.graph_i
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
                    self.folder_i, self.graph_i, self.data_no, self.folder_records = -1, -1, -1, 0
                    break
                self.folder = os.path.join(self.graph_dir, 'train'+str(self.folder_i))
                self.folder_records = len(os.listdir(self.folder))
            if (self.data_no % self.num_workers == self.worker_id) and self.data_no >= 0:
                break
        try:
            graphs, labels = load_graphs(os.path.join(self.folder, str(self.graph_i)+'.bin'))
            graph = graphs[0]

            # filter erroneous graphs which may cause exception
            meta = {}
            meta['dtype_0'] = graph.filter_nodes(lambda nodes: nodes.data["dtype"] == 0)
            meta['dtype_1'] = graph.filter_nodes(lambda nodes: nodes.data["dtype"] == 1)
            meta['unit_0'] = graph.filter_nodes(lambda nodes: nodes.data["unit"] == 0)
            meta['unit_1'] = graph.filter_nodes(lambda nodes: nodes.data["unit"] == 1)
            if self.hps.model == 'HDSG':
                meta['extractable_1'] = graph.filter_nodes(lambda nodes: nodes.data["extractable"] == 1)
                meta['dtype_2'] = graph.filter_nodes(lambda nodes: nodes.data["dtype"] == 2)
        except Exception as e:
            logger.warning('[WARNING] dataloader %d failed reading graph folder %d, file %d.' % (self.worker_id, self.folder_i, self.graph_i))
            logger.warning(str(e))
            return self.__next__()

        if self.hps.use_bert_embedding:
            # load sent and chap features
            sent_features = np.load(os.path.join(self.hps.cache_dir, 'features', 'train' + str(self.folder_i), str(self.graph_i), 'sent_features' + self.hps.bert_finetune + '.npy'))
            sent_features = torch.FloatTensor(sent_features)
            chap_features = np.load(os.path.join(self.hps.cache_dir, 'features', 'train' + str(self.folder_i), str(self.graph_i), 'chap_features' + self.hps.bert_finetune + '.npy'))
            chap_features = torch.FloatTensor(chap_features)
        else:
            sent_features, chap_features = None, None

        time2 = time.time()
        logger.debug('[DEBUG] dataloader %d start reading graph folder %d, file %d. using time %.5f' % (self.worker_id, self.folder_i, self.graph_i, time2-time1))
        return graph, sent_features, chap_features, self.data_no


class Example():
    def __init__(self, summary, ori_text, labels=None):
        self.labels = labels
        self.original_abstract = "\n".join(summary)
        if isinstance(ori_text, list) and isinstance(ori_text[0], list):
            self.original_article_sents = ori_text[len(ori_text)//2]        # only the center chapter can be extracted
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
        if 'ori_text' in e.keys():
            text_key = 'ori_text'
        else:
            text_key = 'text'
        if 'label' in e.keys():
            return Example(e['summary'], e[text_key], e['label'])
        else:
            return Example(e['summary'], e[text_key])

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
            meta = {}
            # filter erroneous graphs which may cause exception
            meta['dtype_1'] = graph.filter_nodes(lambda nodes: nodes.data["dtype"] == 1)
            meta['unit_1'] = graph.filter_nodes(lambda nodes: nodes.data["unit"] == 1)
            if self.hps.model == 'HDSG':
                meta['extractable_1'] = graph.filter_nodes(lambda nodes: nodes.data["extractable"] == 1)
            
            if self.hps.use_bert_embedding:
                # load sent and chap features
                sent_features = np.load(os.path.join(self.hps.cache_dir, 'features', self.mode, str(index), 'sent_features' + self.hps.bert_finetune + '.npy'))
                sent_features = torch.FloatTensor(sent_features)
                chap_features = np.load(os.path.join(self.hps.cache_dir, 'features', self.mode, str(index), 'chap_features' + self.hps.bert_finetune + '.npy'))
                chap_features = torch.FloatTensor(chap_features)
            else:
                sent_features, chap_features = None, None

            return graph, sent_features, chap_features, index
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
    graphs, sent_embeddings, chap_embeddings, index = map(list, zip(*samples))
    graph_len = [len(g.filter_nodes(lambda nodes: nodes.data["dtype"] == 1)) for g in graphs]  # sent node of graph
    sorted_len, sorted_index = torch.sort(torch.LongTensor(graph_len), dim=0, descending=True)
    batched_graph = dgl.batch([graphs[idx] for idx in sorted_index])
    batched_index = [index[idx] for idx in sorted_index]

    if sent_embeddings[0] is not None:
        batched_sent_embeddings = [sent_embeddings[idx] for idx in sorted_index]
        batched_sent_embeddings = torch.cat(batched_sent_embeddings, dim=0)
    else:
        batched_sent_embeddings = None

    if chap_embeddings[0] is not None:
        batched_chap_embeddings = [chap_embeddings[idx] for idx in sorted_index]
        batched_chap_embeddings = torch.cat(batched_chap_embeddings, dim=0)
    else:
        batched_chap_embeddings = None
    return batched_graph, batched_sent_embeddings, batched_chap_embeddings, batched_index
