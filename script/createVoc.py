import os
import json
import nltk
import random
import argparse


def catDoc(textlist):
    res = []
    for tlist in textlist:
        res.extend(tlist)
    return res

def PrintInformation(keys, allcnt):
    # Vocab > 10
    cnt = 0
    first = 0.0
    for key, val in keys:
        if val >= 10:
            cnt += 1
            first += val
    print("appearance > 10 cnt: %d, percent: %f" % (cnt, first / allcnt))  # 416,303

    # first 30,000, last freq 31
    if len(keys) > 30000:
        first = 0.0
        for k, v in keys[:30000]:
            first += v
        print("First 30,000 percent: %f, last freq %d" % (first / allcnt, keys[30000][1]))

    # first 50,000, last freq 383
    if len(keys) > 50000:
        first = 0.0
        for k, v in keys[:50000]:
            first += v
        print("First 50,000 percent: %f, last freq %d" % (first / allcnt, keys[50000][1]))

    # first 100,000, last freq 107
    if len(keys) > 100000:
        first = 0.0
        for k, v in keys[:100000]:
            first += v
        print("First 100,000 percent: %f, last freq %d" % (first / allcnt, keys[100000][1]))

def add_vocab(vocab, fdist):
    keys = fdist1.most_common()
    for key, val in keys:
        if key not in vocab.keys():
            vocab[key] = 0
        vocab[key] += val
    return vocab

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type=str, default='data/CNNDM/train.label.jsonl', help='File to deal with')
    parser.add_argument('--dataset', type=str, default='CNNDM', help='dataset name')

    args = parser.parse_args()

    save_dir = os.path.join("cache", args.dataset)
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    saveFile = os.path.join(save_dir, "vocab")
    print("Save vocab of dataset %s to %s" % (args.dataset, saveFile))

    cnt = 0
    vocab = {}

    if os.path.isdir(args.data_path):
        filenames = [os.path.join(args.data_path, i) for i in os.listdir(args.data_path) if os.path.isfile(os.path.join(args.data_path, i))]
    else:
        filenames = [args.data_path]
    print('training data filenames:', filenames)

    for filename in filenames:
        allword = []
        with open(filename, encoding='utf8') as f:
            for line in f:
                e = json.loads(line)
                if isinstance(e["text"], list) and isinstance(e["text"][0], list):
                    sents = catDoc(e["text"])
                else:
                    sents = e["text"]
                text = " ".join(sents)
                summary = " ".join(e["summary"])
                allword.extend(text.split())
                allword.extend(summary.split())
                cnt += 1

        print('stated: %d' % cnt)
        fdist1 = nltk.FreqDist(allword)
        vocab = add_vocab(vocab, fdist1)

    print("Training set of dataset has %d example" % cnt)

    vocab_tuple = sorted(vocab.items(), key=lambda x: x[1], reverse=True)
    fout = open(saveFile, "w")
    for key, val in vocab_tuple:
        try:
            fout.write("%s\t%d\n" % (key, val))
        except UnicodeEncodeError as e:
            # print(repr(e))
            # print(key, val)
            continue

    fout.close()

    allcnt = sum([i[1] for i in vocab_tuple]) # 788,159,121
    allset = len(vocab_tuple)
    print("All appearance %d, unique word %d" % (allcnt, allset))

    # PrintInformation(keys, allcnt)
