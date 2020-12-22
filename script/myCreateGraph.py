# create dgl graph

import os
import sys
import argparse
import json
import threading
import collections
from collections import Counter
import numpy as np
import torch
import dgl
from dgl.data.utils import save_graphs, load_graphs

sys.path.append('.')
from module.vocabulary import Vocab

######################################### util functions and resources #########################################

def readText(fname):
    data = []
    with open(fname, encoding="utf-8") as f:
        for line in f:
            data.append(line.strip())
    return data

def catDoc(textlist):
    res = []
    for tlist in textlist:
        res.extend(tlist)
    return res

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='qidian_1118_winsize1', help='dataset name')
parser.add_argument('--doc_max_timesteps', type=int, default=100, help='max sentences in input data')
parser.add_argument('--sent_max_len', type=int, default=100, help='max tokens in a sentence')
parser.add_argument('--model', type=str, default='HSG', help='model structure, [HSG|HDSGn]')
parser.add_argument('--vocab_size', type=int, default=100000, help='Size of vocabulary. [default: 50000]')
parser.add_argument('--num_proc', type=int, default=1, help='num of processes.')
parser.add_argument('--no_proc', type=int, default=1, help='no. of this process.')
args = parser.parse_args()

data_folder = os.path.join('./data', args.dataset)
cache_folder = os.path.join('./cache', args.dataset)

# Load Public Resources
VOCAB = Vocab(os.path.join(cache_folder, 'vocab'), args.vocab_size)
TFIDF_W = readText(os.path.join(cache_folder, 'filter_word.txt'))
print('[graph] VOCAB and TFIDF_W loaded.')

FILTERWORDS = [line.strip() for line in open('baidu_stopwords.txt', encoding='utf-8').readlines()]
punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%', '\'\'', '\'', '`', '``',
                '-', '--', '|', '\/']
punctuations.extend(['，', '。', '、', '【', '】', '《', '》', '？', '！', '“', '”', '‘', '’', '*', '—', '…', '\r', '\n', '\t'])
FILTERWORDS.extend(punctuations)

FILTERIDS = [VOCAB.word2id(w.lower()) for w in FILTERWORDS]
FILTERIDS.append(VOCAB.word2id("[PAD]"))   # keep "[UNK]" but remove "[PAD]"

lowtfidf_num = 0
pattern = r"^[0-9]+$"
for w in TFIDF_W:
    if VOCAB.word2id(w) != VOCAB.word2id('[UNK]'):
        FILTERWORDS.append(w)
        FILTERIDS.append(VOCAB.word2id(w))
        # if re.search(pattern, w) == None:  # if w is a number, it will not increase the lowtfidf_num
            # lowtfidf_num += 1
        lowtfidf_num += 1
    if lowtfidf_num > 5000:
        break

######################################### Exapmles #########################################

class Example(object):
    """Class representing a train/val/test example for single-document extractive summarization."""

    def __init__(self, article_sents, abstract_sents, sent_max_len, label):
        """ Initializes the Example, performing tokenization and truncation to produce the encoder, decoder and target sequences, which are stored in self.

        :param article_sents: list(strings) for single document or list(list(string)) for multi-document; one per article sentence. each token is separated by a single space.
        :param abstract_sents: list(strings); one per abstract sentence. In each sentence, each token is separated by a single space.
        :param sent_max_len: int, max length of each sentence
        :param label: list, the No of selected sentence, e.g. [1,3,5]
        """

        self.sent_max_len = sent_max_len
        self.enc_sent_len = []
        self.enc_sent_input = []
        self.enc_sent_input_pad = []

        # Store the original strings
        self.original_article_sents = article_sents
        self.original_abstract = "\n".join(abstract_sents)

        # Process the article
        if isinstance(article_sents, list) and isinstance(article_sents[0], list):  # multi document
            self.original_article_sents = []
            for doc in article_sents:
                self.original_article_sents.extend(doc)
        for sent in self.original_article_sents:
            article_words = sent.split()
            self.enc_sent_len.append(len(article_words))  # store the length before padding
            self.enc_sent_input.append([VOCAB.word2id(w.lower()) for w in article_words])  # list of word ids; OOVs are represented by the id for UNK token
        self._pad_encoder_input(VOCAB.word2id('[PAD]'))

        # Store the label
        self.label = label
        label_shape = (len(self.original_article_sents), len(label))  # [N, len(label)]
        # label_shape = (len(self.original_article_sents), len(self.original_article_sents))
        self.label_matrix = np.zeros(label_shape, dtype=int)
        if label != []:
            self.label_matrix[np.array(label), np.arange(len(label))] = 1  # label_matrix[i][j]=1 indicate the i-th sent will be selected in j-th step

    def _pad_encoder_input(self, pad_id):
        """
        :param pad_id: int; token pad id
        :return: 
        """
        max_len = self.sent_max_len
        for i in range(len(self.enc_sent_input)):
            article_words = self.enc_sent_input[i].copy()
            if len(article_words) > max_len:
                article_words = article_words[:max_len]
            if len(article_words) < max_len:
                article_words.extend([pad_id] * (max_len - len(article_words)))
            self.enc_sent_input_pad.append(article_words)

class Example2(Example):
    """Class representing a train/val/test example for multi-document extractive summarization."""

    def __init__(self, article_sents, abstract_sents, extractable_labels, sent_max_len, label):
        """ Initializes the Example, performing tokenization and truncation to produce the encoder, decoder and target sequences, which are stored in self.

        :param article_sents: list(list(string)) for multi-document; one per article sentence. each token is separated by a single space.
        :param abstract_sents: list(strings); one per abstract sentence. In each sentence, each token is separated by a single space.
        :param extractable_labels: list(int); one int(0/1) per article sentence. 1 if the sentence can be extracted and 0 if the sentence can not.
        :param sent_max_len: int, max length of each sentence
        :param label: list, the No of selected sentence, e.g. [1,3,5]
        """

        super().__init__(article_sents, abstract_sents, sent_max_len, label)
        cur = 0
        self.original_articles = []
        self.article_len = []
        self.enc_doc_input = []
        self.extractable_labels = extractable_labels
        for doc in article_sents:
            if len(doc) == 0:
                continue
            docLen = len(doc)
            self.original_articles.append(" ".join(doc))
            self.article_len.append(docLen)
            self.enc_doc_input.append(catDoc(self.enc_sent_input[cur:cur + docLen]))
            cur += docLen


######################################### Graph Processers #########################################

class GraphPreprocesser(object):

    def __init__(self, data_path, w2s_path, doc_max_timesteps, sent_max_len, dest_folder):
        self.data_fd = open(data_path, encoding='utf-8')
        self.w2s_fd = open(w2s_path, encoding='utf-8')
        self.doc_max_timesteps = doc_max_timesteps
        self.sent_max_len = sent_max_len
        self.dest_folder = dest_folder
        self.processed_graphs = os.listdir(dest_folder)
        self.size = int(os.popen('wc -l {}'.format(data_path)).read().split()[0])

    def process(self):
        for i in range(self.size):
            if str(i) + '.bin' in self.processed_graphs:
                try:
                    _, _ = load_graphs(os.path.join(self.dest_folder, str(i) + '.bin'))
                    yield None, None, None, i
                    continue
                except Exception as e:
                    print('folder %s, graph %d, error when loading.' % (self.dest_folder, i))
            item, bookid, chapno = self.get_example()
            w2s_w = self.get_w2s()
            input_pad = item.enc_sent_input_pad[:self.doc_max_timesteps]
            label = self.pad_label_m(item.label_matrix)
            G = self.CreateGraph(input_pad, label, w2s_w)
            yield G, bookid, chapno, i

    def get_example(self):
        e = json.loads(self.data_fd.readline())
        e["summary"] = e.setdefault("summary", [])
        example = Example(e["text"], e["summary"], self.sent_max_len, e["label"])
        return example, e["bookid"], e["chapno"]

    def get_w2s(self):
        return json.loads(self.w2s_fd.readline())

    def CreateGraph(self, input_pad, label, w2s_w):
        """ Create a graph for each document
        
        :param input_pad: list(list); [sentnum, wordnum]
        :param label: list(list); [sentnum, sentnum]
        :param w2s_w: dict(dict) {str: {str: float}}; for each sentence and each word, the tfidf between them
        :return: G: dgl.DGLGraph
            node:
                word: unit=0, dtype=0, id=(int)wordid in vocab
                sentence: unit=1, dtype=1, words=tensor, position=int, label=tensor
            edge:
                word2sent, sent2word:  tffrac=int, dtype=0
        """
        G = dgl.DGLGraph()
        wid2nid, nid2wid = self.AddWordNode(G, input_pad)
        w_nodes = len(nid2wid)

        N = len(input_pad)
        G.add_nodes(N)
        G.ndata["unit"][w_nodes:] = torch.ones(N)
        G.ndata["dtype"][w_nodes:] = torch.ones(N)
        sentid2nid = [i + w_nodes for i in range(N)]

        G.set_e_initializer(dgl.init.zero_initializer)
        for i in range(N):
            c = Counter(input_pad[i])
            sent_nid = sentid2nid[i]
            sent_tfw = w2s_w[str(i)]
            for wid in c.keys():
                if wid in wid2nid.keys() and VOCAB.id2word(wid) in sent_tfw.keys():
                    tfidf = sent_tfw[VOCAB.id2word(wid)]
                    tfidf_box = np.round(tfidf * 9)  # box = 10
                    G.add_edges(wid2nid[wid], sent_nid,
                                data={"tffrac": torch.LongTensor([tfidf_box]), "dtype": torch.Tensor([0])})
                    G.add_edges(sent_nid, wid2nid[wid],
                                data={"tffrac": torch.LongTensor([tfidf_box]), "dtype": torch.Tensor([0])})

            # The two lines can be commented out if you use the code for your own training, since HSG does not use sent2sent edges. 
            # However, if you want to use the released checkpoint directly, please leave them here.
            # Otherwise it may cause some parameter corresponding errors due to the version differences.

            G.add_edges(sent_nid, sentid2nid, data={"dtype": torch.ones(N)})
            G.add_edges(sentid2nid, sent_nid, data={"dtype": torch.ones(N)})
        G.nodes[sentid2nid].data["words"] = torch.LongTensor(input_pad)  # [N, seq_len]
        G.nodes[sentid2nid].data["position"] = torch.arange(1, N + 1).view(-1, 1).long()  # [N, 1]
        G.nodes[sentid2nid].data["label"] = torch.LongTensor(label)  # [N, doc_max]
        return G

    def pad_label_m(self, label_matrix):
        label_m = label_matrix[:self.doc_max_timesteps, :self.doc_max_timesteps]
        N, m = label_m.shape
        if m < self.doc_max_timesteps:
            pad_m = np.zeros((N, self.doc_max_timesteps - m))
            return np.hstack([label_m, pad_m])
        return label_m

    def AddWordNode(self, G, inputid):
        wid2nid = {}
        nid2wid = {}
        nid = 0
        for sentid in inputid:
            for wid in sentid:
                if wid not in FILTERIDS and wid not in wid2nid.keys():
                    wid2nid[wid] = nid
                    nid2wid[nid] = wid
                    nid += 1

        w_nodes = len(nid2wid)
        G.add_nodes(w_nodes)
        G.set_n_initializer(dgl.init.zero_initializer)
        G.ndata["unit"] = torch.zeros(w_nodes)
        G.ndata["id"] = torch.LongTensor(list(nid2wid.values()))
        G.ndata["dtype"] = torch.zeros(w_nodes)
        return wid2nid, nid2wid


class MultiGraphPreprocesser(GraphPreprocesser):
    def __init__(self, data_path, w2s_path, w2d_path, doc_max_timesteps, sent_max_len, dest_folder):
        super().__init__(data_path, w2s_path, doc_max_timesteps, sent_max_len, dest_folder)
        self.w2d_fd = open(w2d_path, encoding='utf-8')

    def process(self):
        for i in range(self.size):
            if str(i) + '.bin' in self.processed_graphs:
                try:
                    _, _ = load_graphs(os.path.join(self.dest_folder, str(i) + '.bin'))
                    yield None, None, None, i
                    continue
                except Exception as e:
                    print('folder %s, graph %d, error when loading.' % (self.dest_folder, i))
            item, bookid, chapno = self.get_example()
            w2s_w = self.get_w2s()
            w2d_w = self.get_w2d()
            sent_pad = item.enc_sent_input_pad[:self.doc_max_timesteps]
            extractable_labels = item.extractable_labels[:self.doc_max_timesteps]
            enc_doc_input = item.enc_doc_input
            article_len = item.article_len
            label = self.pad_label_m(item.label_matrix)
            G = self.CreateGraph(article_len, sent_pad, enc_doc_input, label, extractable_labels, w2s_w, w2d_w)
            yield G, bookid, chapno, i

    def get_example(self):
        e = json.loads(self.data_fd.readline())
        e["summary"] = e.setdefault("summary", [])
        example = Example2(e["text"], e["summary"], e["extractable"], self.sent_max_len, e["label"])
        return example, e["bookid"], e["chapno"]

    def get_w2d(self):
        return json.loads(self.w2d_fd.readline())

    def MapSent2Doc(self, article_len, sentNum):
        sent2doc = {}
        doc2sent = {}
        sentNo = 0
        for i in range(len(article_len)):
            doc2sent[i] = []
            for j in range(article_len[i]):
                sent2doc[sentNo] = i
                doc2sent[i].append(sentNo)
                sentNo += 1
                if sentNo > sentNum:
                    return sent2doc
        return sent2doc

    def CreateGraph(self, docLen, sent_pad, doc_pad, label, extractable_labels, w2s_w, w2d_w):
        """ Create a graph for each document

        :param docLen: list; the length of each document in this example
        :param sent_pad: list(list), [sentnum, wordnum]
        :param doc_pad: list, [document, wordnum]
        :param label: list(list), [sentnum, sentnum]
        :param extractable_labels: list(0/1), [sentnum] whether the sentence can be extracted
        :param w2s_w: dict(dict) {str: {str: float}}, for each sentence and each word, the tfidf between them
        :param w2d_w: dict(dict) {str: {str: float}}, for each document and each word, the tfidf between them
        :return: G: dgl.DGLGraph
            node:
                word: unit=0, dtype=0, id=(int)wordid in vocab
                sentence: unit=1, dtype=1, words=tensor, position=int, label=tensor
                document: unit=1, dtype=2
            edge:
                word2sent, sent2word: tffrac=int, dtype=0
                word2doc, doc2word: tffrac=int, dtype=0
                sent2doc: dtype=2
        """
        # add word nodes
        G = dgl.DGLGraph()
        wid2nid, nid2wid = self.AddWordNode(G, sent_pad)
        w_nodes = len(nid2wid)

        # add sent nodes
        N = len(sent_pad)
        G.add_nodes(N)
        G.ndata["unit"][w_nodes:] = torch.ones(N)
        G.ndata["dtype"][w_nodes:] = torch.ones(N)
        sentid2nid = [i + w_nodes for i in range(N)]
        ws_nodes = w_nodes + N

        # add doc nodes
        sent2doc = self.MapSent2Doc(docLen, N)
        article_num = len(set(sent2doc.values()))
        G.add_nodes(article_num)
        G.ndata["unit"][ws_nodes:] = torch.ones(article_num)
        G.ndata["dtype"][ws_nodes:] = torch.ones(article_num) * 2
        docid2nid = [i + ws_nodes for i in range(article_num)]

        # add sent edges
        for i in range(N):
            c = Counter(sent_pad[i])
            sent_nid = sentid2nid[i]
            sent_tfw = w2s_w[str(i)]
            for wid, cnt in c.items():
                if wid in wid2nid.keys() and VOCAB.id2word(wid) in sent_tfw.keys():
                    tfidf = sent_tfw[VOCAB.id2word(wid)]
                    tfidf_box = np.round(tfidf * 9)  # box = 10
                    # w2s s2w
                    G.add_edge(wid2nid[wid], sent_nid,
                               data={"tffrac": torch.LongTensor([tfidf_box]), "dtype": torch.Tensor([0])})
                    G.add_edge(sent_nid, wid2nid[wid],
                               data={"tffrac": torch.LongTensor([tfidf_box]), "dtype": torch.Tensor([0])})
            # s2d
            docid = sent2doc[i]
            docnid = docid2nid[docid]
            G.add_edge(sent_nid, docnid, data={"dtype": torch.Tensor([2])})

        # add doc edges
        for i in range(article_num):
            c = Counter(doc_pad[i])
            doc_nid = docid2nid[i]
            doc_tfw = w2d_w[str(i)]
            for wid, cnt in c.items():
                if wid in wid2nid.keys() and VOCAB.id2word(wid) in doc_tfw.keys():
                    # w2d d2w
                    tfidf = doc_tfw[VOCAB.id2word(wid)]
                    tfidf_box = np.round(tfidf * 9)  # box = 10
                    G.add_edge(wid2nid[wid], doc_nid,
                               data={"tffrac": torch.LongTensor([tfidf_box]), "dtype": torch.Tensor([0])})
                    G.add_edge(doc_nid, wid2nid[wid],
                               data={"tffrac": torch.LongTensor([tfidf_box]), "dtype": torch.Tensor([0])})

        G.nodes[sentid2nid].data["words"] = torch.LongTensor(sent_pad)  # [N, seq_len]
        G.nodes[sentid2nid].data["position"] = torch.arange(1, N + 1).view(-1, 1).long()  # [N, 1]
        G.nodes[sentid2nid].data["label"] = torch.LongTensor(label)  # [N, doc_max]
        G.nodes[sentid2nid].data["extractable"] = torch.LongTensor(extractable_labels).view(-1, 1)  # [N, 1]
        return G


######################################### Threads #########################################

class ProcessThread(threading.Thread):
    def __init__(self, src_data_file, src_w2s_file, dest_folder, hps, _id=-1):
        threading.Thread.__init__(self)
        self.src_data_file = src_data_file
        self.src_w2s_file = src_w2s_file
        self.dest_folder = dest_folder
        self.hps = hps
        self._id = _id

    def run(self):
        print('[graph] start thread: %d' % self._id)
        gp = GraphPreprocesser(self.src_data_file, self.src_w2s_file, self.hps.doc_max_timesteps, self.hps.sent_max_len, dest_folder=self.dest_folder)
        for i, (graph, bookid, chapno, index) in enumerate(gp.process()):
            assert i == index, "data id != graph id"
            if graph is None:
                continue
            # graph_label ={"content": torch.tensor([int(bookid), int(chapno)])}
            graph_label ={"content": torch.tensor([int(chapno)])}
            save_graphs(os.path.join(self.dest_folder, str(i)+'.bin'), [graph], graph_label)
        print('[graph] finish thread: %d' % self._id)

class MultiProcessThread(ProcessThread):
    def __init__(self, src_data_file, src_w2s_file, src_w2d_file, dest_folder, hps, _id=-1):
        super().__init__(src_data_file, src_w2s_file, dest_folder, hps, _id)
        self.src_w2d_file = src_w2d_file

    def run(self):
        print('[graph] start thread: %d' % self._id)
        mgp = MultiGraphPreprocesser(self.src_data_file, self.src_w2s_file, self.src_w2d_file, self.hps.doc_max_timesteps, self.hps.sent_max_len, dest_folder=self.dest_folder)
        for i, (graph, bookid, chapno, index) in enumerate(mgp.process()):
            assert i == index, "data id != graph id"
            if graph is None:
                continue
            # graph_label = {"content": torch.tensor([int(bookid), int(chapno)])}
            graph_label = {"content": torch.tensor([int(chapno)])}
            save_graphs(os.path.join(self.dest_folder, str(i)+'.bin'), [graph], graph_label)
        print('[graph] finish thread: %d' % self._id)

def main():
    if not os.path.exists(os.path.join(cache_folder, 'graph')):
        os.mkdir(os.path.join(cache_folder, 'graph'))

    # build graph
    threads = []
    filenames = [f for f in os.listdir(data_folder)]
    filenames.sort()
    for i, filename in enumerate(filenames):
        if i % args.num_proc != args.no_proc - 1:
            continue
        if not os.path.exists(os.path.join(cache_folder, 'graph', filename.replace('.json', ''))):
            os.mkdir(os.path.join(cache_folder, 'graph', filename.replace('.json', '')))
        if args.model == 'HSG':
            threads.append(ProcessThread(
                src_data_file=os.path.join(data_folder, filename),
                src_w2s_file=os.path.join(cache_folder, 'w2s', filename),
                dest_folder=os.path.join(cache_folder, 'graph', filename.replace('.json', '')),
                hps=args, _id=i
            ))
        elif args.model.startswith('HDSG'):
            threads.append(MultiProcessThread(
                src_data_file=os.path.join(data_folder, filename),
                src_w2s_file=os.path.join(cache_folder, 'w2s', filename),
                src_w2d_file=os.path.join(cache_folder, 'w2d', filename),
                dest_folder=os.path.join(cache_folder, 'graph', filename.replace('.json', '')),
                hps=args, _id=i
            ))

    for i, thread in enumerate(threads):
        thread.start()
    for i, thread in enumerate(threads):
        thread.join()

if __name__ == '__main__':
    main()
