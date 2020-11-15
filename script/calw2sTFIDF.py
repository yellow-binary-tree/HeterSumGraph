import os
import argparse
import json
import threading

from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer

def GetType(path):
    if 'train' in path:
        return 'train'
    if 'val' in path:
        return 'val'
    if 'test' in path:
        return 'test'
    return None


def catDoc(textlist):
    res = []
    for tlist in textlist:
        res.extend(tlist)
    return res

def get_tfidf_embedding(text):
    """
    
    :param text: list, sent_number * word
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
    """
    
    :param a: matrix, [N, M], N is document number, M is word number
    :param id2word: word id to word
    :return: 
    """
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
                sents = catDoc(e["text"])
            else:
                sents = e["text"]
            cntvector, tfidf_weight = get_tfidf_embedding(sents)
            id2word = {}
            for w, tfidf_id in cntvector.vocabulary_.items():   # word -> tfidf matrix row number
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
    fname = GetType(args.data_path) + ".w2s.tfidf.jsonl"
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
                os.path.join(save_dir, 'train_split', 'train.w2s.tfidf.jsonl_' + filename[-2:]),
                i)
            threads.append(new_thread)
        for i, thread in enumerate(threads):
            thread.start()
        for i, thread in enumerate(threads):
            thread.join()


if __name__ == '__main__':
    main()