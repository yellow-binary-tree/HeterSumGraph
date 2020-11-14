import os
import json

def create_win_size3_corpus(src_folder, dest_folder):
    for filename in [filename for filename in os.listdir(src_folder) if os.path.isfile(os.path.join(src_folder, filename))]:
        f_in = open(os.path.join(src_folder, filename), encoding='utf-8')
        f_out = open(os.path.join(dest_folder, filename), 'w', encoding='utf-8')
        for line in f_in:
            dict_in = json.loads(line)
            dict_out = {
                'chapno': dict_in['chapno'], 'summary': dict_in['summary'],
                'text': dict_in['text'][1:-1], 'label': [],
                'extractable': dict_in['extractable']
            }
            chap_lens = [len(chap) for chap in dict_in['text']]
            del dict_out['extractable'][sum(chap_lens[:-1]): sum(chap_lens)]
            del dict_out['extractable'][0: chap_lens[0]]
            assert len(dict_out['extractable']) == sum([len(chap) for chap in dict_out['text']]), 'extractable_length_error'
            for label in dict_in['label']:
                dict_out['label'].append(label - chap_lens[0])

            write_str = json.dumps(dict_out, ensure_ascii=False)
            f_out.write(write_str + '\n')

def check_long_corpus(data_file):
    f = open(data_file, 'r', encoding='utf-8')
    for i, line in enumerate(f):
        if i < 200*24:
            continue
        data_dict = json.loads(line)
        text_num_sentences = sum([len(chap) for chap in data_dict['text'][:-1]])
        if text_num_sentences > 560:
            print(i, text_num_sentences)
        if i > 250*24:
            break


if __name__ == "__main__":
    # create_win_size3_corpus('./data/qidian_1109_seq', './data/qidian_1109_seq_winsize3')
    check_long_corpus('./data/qidian_1109_seq/train.label.jsonl')