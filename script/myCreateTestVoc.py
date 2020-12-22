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

def add_vocab(vocab, fdist):
    keys = fdist1.most_common()
    for key, val in keys:
        if key not in vocab.keys():
            vocab[key] = 0
        vocab[key] += val
    return vocab

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='qidian_1118_winsize1', help='dataset name')
    args = parser.parse_args()

    data_path = os.path.join("data", args.dataset)
    save_dir = os.path.join("cache", args.dataset)
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    saveFile = os.path.join(save_dir, "test_vocab")
    print("[vocab] Save vocab of dataset %s to %s" % (args.dataset, saveFile))

    cnt = 0
    vocab = {}
    filenames = [os.path.join(data_path, f) for f in os.listdir(data_path) \
                 if os.path.isfile(os.path.join(data_path, f)) and 'val' in f or 'test' in f]

    for filename in filenames:
        allword = []
        with open(filename, encoding='utf8') as f:
            for line in f:
                e = json.loads(line)
                if "ori_text" in e.keys():
                    text_key = 'ori_text'
                else: 
                    text_key = 'text'

                if isinstance(e[text_key], list) and isinstance(e[text_key][0], list):
                    sents = catDoc(e[text_key])
                else:
                    sents = e[text_key]
                text = " ".join(sents)
                summary = " ".join(e["summary"])
                allword.extend(text.split())
                allword.extend(summary.split())
                cnt += 1

        print('[vocab] stated: %d' % cnt)
        fdist1 = nltk.FreqDist(allword)
        vocab = add_vocab(vocab, fdist1)

    print("[vocab] Training set of dataset has %d example" % cnt)

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

    allcnt = sum([i[1] for i in vocab_tuple])
    allset = len(vocab_tuple)
    print("[vocab] All appearance %d, unique word %d" % (allcnt, allset))

