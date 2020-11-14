import os
import argparse
import json
import threading

from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

def GetType(path):
    if 'train' in path:
        return 'train'
    if 'val' in path:
        return 'val'
    if 'test' in path:
        return 'test'
    return None

def get_tfidf_embedding(text):
    """

    :param text: list, doc_number * word
    :return: 
        vectorizer: 
            vocabulary_: word2id
            get_feature_names(): id2word
        tfidf: array [sent_number, max_word_number]
    """
    vectorizer = CountVectorizer(lowercase=True)
    word_count = vectorizer.fit_transform(text)
    tfidf_transformer = TfidfTransformer()
    tfidf = tfidf_transformer.fit_transform(word_count)
    tfidf_weight = tfidf.toarray()
    return vectorizer, tfidf_weight


def compress_array(a, id2word):
    d = {}
    for i in range(len(a)):
        d[i] = {}
        for j in range(len(a[i])):
            if a[i][j] != 0:
                d[i][id2word[j]] = a[i][j]
    return d

def process_file(srcFile, saveFile):
    fout = open(saveFile, "w")
    with open(srcFile, encoding='utf-8') as f:
        for line in f:
            e = json.loads(line)
            if isinstance(e["text"], list) and isinstance(e["text"][0], list):
                docs = [" ".join(doc) for doc in e["text"]]
            else:
                docs = [e["text"]]
            cntvector, tfidf_weight = get_tfidf_embedding(docs)
            id2word = {}
            for w, tfidf_id in cntvector.vocabulary_.items():
                id2word[tfidf_id] = w
            tfidfvector = compress_array(tfidf_weight, id2word)
            fout.write(json.dumps(tfidfvector) + "\n")


class ProcessThread(threading.Thread):
    def __init__(self, srcFile, saveFile, _id=-1):
        threading.Thread.__init__(self)
        self.srcFile = srcFile
        self.saveFile = saveFile
        self._id = _id
        print('init thread: %d' % self._id)

    def run(self):
        print('start thread: %d' % self._id)
        process_file(self.srcFile, self.saveFile)
        print('finish thread: %d' % self._id)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data/CNNDM/train.label.jsonl', help='File to deal with')
    parser.add_argument('--dataset', type=str, default='CNNDM', help='dataset name')
    args = parser.parse_args()

    save_dir = os.path.join("cache", args.dataset)
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    fname = GetType(args.data_path) + ".w2d.tfidf.jsonl"
    saveFile = os.path.join(save_dir, fname)
    print("Save word2sent features of dataset %s to %s" % (args.dataset, saveFile))

    if os.path.isfile(args.data_path):
        process_file(args.data_path, saveFile)
    else:
        filelist = os.listdir(args.data_path)
        threads = []
        if not os.path.exists(os.path.join(save_dir, 'train_split')):
            os.makedirs(os.path.join(save_dir, 'train_split'))
        for i, filename in enumerate(filelist):
            new_thread = ProcessThread(
                os.path.join(args.data_path, filename),
                os.path.join(save_dir, 'train_split', 'train.w2d.tfidf.jsonl_' + filename[-2:]),
                i)
            threads.append(new_thread)
        for i, thread in enumerate(threads):
            thread.start()
        for i, thread in enumerate(threads):
            thread.join()

if __name__ == '__main__':
    main()

