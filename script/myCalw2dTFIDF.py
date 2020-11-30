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
    processed_lines = 0
    if os.path.exists(saveFile):
        fout = open(saveFile, "r")
        for line in fout:
            processed_lines += 1
        fout = open(saveFile, "a")
    else:
        fout = open(saveFile, "w")

    with open(srcFile, encoding='utf-8') as f:
        for i in range(processed_lines):
            f.readline()

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
    def __init__(self, src_file, dest_file, _id=-1):
        threading.Thread.__init__(self)
        self.src_file = src_file
        self.dest_file = dest_file
        self._id = _id
        print('[w2d] init thread: %d' % self._id)

    def run(self):
        print('[w2d] start thread: %d' % self._id)
        process_file(self.src_file, self.dest_file)
        print('[w2d] finish thread: %d' % self._id)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='qidian_1118_winsize1', help='dataset name')
    args = parser.parse_args()

    data_path = os.path.join("data", args.dataset)
    save_dir = os.path.join("cache", args.dataset, "w2d")
    if not os.path.exists(save_dir): os.makedirs(save_dir)

    filelist = os.listdir(data_path)
    threads = []
    for i, filename in enumerate(filelist):
        new_thread = ProcessThread(
            src_file=os.path.join(data_path, filename),
            dest_file=os.path.join(save_dir, filename), _id=i)
        threads.append(new_thread)
    for i, thread in enumerate(threads):
        thread.start()
    for i, thread in enumerate(threads):
        thread.join()

if __name__ == '__main__':
    main()

