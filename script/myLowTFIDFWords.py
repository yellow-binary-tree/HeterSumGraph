#!/bin/env python
#coding:utf-8
#Author:brxx122@gmail.com

import os
import json
import argparse
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer


def catDoc(textlist):
    res = []
    for tlist in textlist:
        res.extend(tlist)
    return res

def calTFidf(text):
    vectorizer = CountVectorizer(lowercase=True)
    wordcount = vectorizer.fit_transform(text)
    tf_idf_transformer = TfidfTransformer()
    tfidf_matrix = tf_idf_transformer.fit_transform(wordcount)
    return vectorizer, tfidf_matrix

            
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='qidian_1118_winsize1', help='dataset name')
    args = parser.parse_args()

    data_path = os.path.join("data", args.dataset)
    save_dir = os.path.join("cache", args.dataset)
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    saveFile = os.path.join(save_dir, "filter_word.txt")
    print("Save low tfidf words in dataset %s to %s" % (args.dataset, saveFile))

    documents = []
    filenames = [os.path.join(data_path, f) for f in os.listdir(data_path) \
                 if os.path.isfile(os.path.join(data_path, f)) and 'train' in f]

    for filename in filenames:
        for line in open(filename, "r", encoding="utf-8"):
            e = json.loads(line)
            if isinstance(e["text"], list) and isinstance(e["text"][0], list):
                text = catDoc(e["text"])
            else:
                text = e["text"]
            documents.append(" ".join(text))
            
    vectorizer, tfidf_matrix = calTFidf(documents)
    print("The number of example is %d, and the TFIDF vocabulary size is %d" % (len(documents), len(vectorizer.vocabulary_)))
    word_tfidf = np.array(tfidf_matrix.mean(0))
    del tfidf_matrix
    word_order = np.argsort(word_tfidf[0])

    id2word = vectorizer.get_feature_names()
    with open(saveFile, "w") as fout:
        for idx in word_order:
            w = id2word[idx]
            string = w + "\n"
            try:
                fout.write(string)
            except:
                pass

if __name__ == '__main__':
    main()
