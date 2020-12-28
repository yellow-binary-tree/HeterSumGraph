# coding=utf-8
# temp.py

import os
import json
import time

import torch
import numpy as np
from transformers import BertTokenizerFast, BertModel, BertConfig


def bert_base_encode_chapters(model_name='bert-base-uncased', dataset='wiki_winsize1'):
    model_base_folder = '/share/wangyq/.cache/huggingface/transformers/' + model_name
    graph_base_folder = '/share/wangyq/project/HeterSumGraph/cache/' + dataset + '/features'
    data_base_folder = '/share/wangyq/project/HeterSumGraph/data/' + dataset

    print('bert_base_encode_sentences')
    print('data_base_folder: ', data_base_folder)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('device: %s' % device)

    print('loading model...')
    tokenizer = BertTokenizerFast.from_pretrained(model_base_folder)
    config = BertConfig.from_pretrained(model_base_folder)
    model = BertModel.from_pretrained(model_base_folder, config=config)
    model.to(device)
    print('finish loading model')

    data_sections = [f.replace('.json', '') for f in os.listdir(data_base_folder)]
    data_sections.sort()
    print('data_sections:', data_sections)

    for data_section in data_sections:
        data_fd = open(os.path.join(data_base_folder, data_section+'.json'), encoding='utf-8')
        print('start data_section: %s' % data_section)
        start_time = time.time()
        for i, line in enumerate(data_fd):
            data_folder = os.path.join(graph_base_folder, data_section, str(i))
            if not os.path.exists(data_folder):
                os.makedirs(data_folder)
            data_dict = json.loads(line)

            cls_outputs = []
            if 'ori_text' in data_dict:
                text_key = 'ori_text'
            else:
                text_key = 'text'

            if isinstance(data_dict[text_key][0], list):
                chapter_texts = [[' '.join(chap)] for chap in data_dict[text_key]]
            else:
                chapter_texts = [[' '.join(data_dict[text_key])]]
            if 'qd' in dataset:     # chinese, remove the spaces between tokens
                chapter_texts = [[chap[0].replace(' ', '')] for chap in chapter_texts]

            for chapter_text in chapter_texts:
                # print(chapter_text)
                tokenized_chapter = tokenizer(chapter_text, padding='max_length', truncation=True, max_length=512, return_tensors="pt")
                tokenized_chapter.to(device)
                outputs = model(**tokenized_chapter)
                cls_outputs.append(outputs.last_hidden_state[:, 0, :].cpu().detach().numpy())
            cls_output = np.concatenate(cls_outputs, axis=0)
            assert cls_output.shape[0] == int(dataset[-1])
            np.save(os.path.join(data_folder, 'chap_features.npy'), cls_output)

        finish_time = time.time()
        print('finish data_section: %s, time %lf' % (data_section, finish_time - start_time))


def bert_base_encode_vocab(model_name, dataset):
    model_base_folder = '/share/wangyq/.cache/huggingface/transformers/' + model_name
    vocab_file = '/share/wangyq/project/HeterSumGraph/cache/' + dataset + '/vocab'
    embedding_file = '/share/wangyq/project/HeterSumGraph/cache/' + dataset + '/embedding'
    BATCH_SIZE = 32
    VOCAB_SIZE = 100000

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('device: %s' % device)

    print('loading model...')
    tokenizer = BertTokenizerFast.from_pretrained(model_base_folder)
    config = BertConfig.from_pretrained(model_base_folder)
    model = BertModel.from_pretrained(model_base_folder, config=config)
    model.to(device)
    print('finish loading model')

    vocab_fd = open(vocab_file, encoding='utf-8')
    embedding_fd = open(embedding_file, 'w', encoding='utf-8')
    embedding_fd.write(str(VOCAB_SIZE) + ' 768\n')
    sequences = []
    tic = time.time()
    for i, line in enumerate(vocab_fd):
        token, _ = line.split('\t')
        sequences.append(token)
        if i % BATCH_SIZE == BATCH_SIZE-1:
            tokenized_sequence = tokenizer(sequences, padding=True, return_tensors="pt")
            tokenized_sequence.to(device)
            outputs = model(**tokenized_sequence)
            cls_output = outputs.last_hidden_state[:, 0, :]
            for j, token in enumerate(sequences):
                embedding = cls_output[j].tolist()
                embedding_fd.write(token + ' ')
                for k, num in enumerate(embedding):
                    embedding_fd.write('{:.5f}'.format(num))
                    if k == len(embedding) - 1:
                        embedding_fd.write('\n')
                    else:
                        embedding_fd.write(' ')
            sequences = []
        if i % 1000 == 0:
            toc = time.time()
            print('embedded: %d, time: %lf' % (i, toc-tic))
            tic = time.time()
        if i > VOCAB_SIZE:
            return


def bert_base_encode_sentences(model_name, dataset):
    model_base_folder = '/share/wangyq/.cache/huggingface/transformers/' + model_name
    graph_base_folder = '/share/wangyq/project/HeterSumGraph/cache/' + dataset + '/features'
    data_base_folder = '/share/wangyq/project/HeterSumGraph/data/' + dataset

    print('bert_base_encode_sentences')
    print('data_base_folder: ', data_base_folder)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('device: %s' % device)

    print('loading model...')
    tokenizer = BertTokenizerFast.from_pretrained(model_base_folder)
    config = BertConfig.from_pretrained(model_base_folder)
    model = BertModel.from_pretrained(model_base_folder, config=config)
    model.to(device)
    print('finish loading model')

    data_sections = [f.replace('.json', '') for f in os.listdir(data_base_folder)]
    data_sections.sort()
    print('data_sections:', data_sections)

    for data_section in data_sections:
        data_fd = open(os.path.join(data_base_folder, data_section+'.json'), encoding='utf-8')
        print('start data_section: %s' % data_section)
        start_time = time.time()
        for i, line in enumerate(data_fd):
            data_folder = os.path.join(graph_base_folder, data_section, str(i))
            if not os.path.exists(data_folder):
                os.makedirs(data_folder)
            if 'sent_features.npy' in os.listdir(data_folder):
                continue
            print("missing sent_features: %s, %d" % (data_section, i))
            data_dict = json.loads(line)

            cls_outputs = []
            ori_texts = []
            if 'ori_text' in data_dict:
                text_key = 'ori_text'
            else:
                text_key = 'text'

            if isinstance(data_dict[text_key][0], list):
                for chap in data_dict[text_key]:
                    # ori_texts.extend([sent.replace(' ', '') for sent in chap])
                    ori_texts.extend([sent for sent in chap])
            else:
                ori_texts = data_dict[text_key]

            # batches = [ori_texts[:len(ori_texts)//2], ori_texts[len(ori_texts)//2:]]
            # batches = [ori_texts[:len(ori_texts)//3], ori_texts[len(ori_texts)//3:len(ori_texts)//3*2], ori_texts[len(ori_texts)//3*2:]]
            batches = []
            for i in range(6):
                if i < 6-1:
                    batches.append(ori_texts[len(ori_texts)//6*i:len(ori_texts)//6*(i+1)])
                else:
                    batches.append(ori_texts[len(ori_texts)//6*i:])
            for sequences in batches:
                # tokenized_sequence = tokenizer(sequences, padding='max_length', truncation=True, max_length=100, return_tensors="pt")
                tokenized_sequence = tokenizer(sequences, padding='max_length', truncation=True, max_length=70, return_tensors="pt")
                # print(tokenized_sequence['input_ids'].shape)
                tokenized_sequence.to(device)
                outputs = model(**tokenized_sequence)
                cls_outputs.append(outputs.last_hidden_state[:, 0, :].cpu().detach().numpy())
            cls_output = np.concatenate(cls_outputs, axis=0)
            assert cls_output.shape[0] == len(ori_texts)
            np.save(os.path.join(data_folder, 'sent_features.npy'), cls_output)

        finish_time = time.time()
        print('finish data_section: %s, time %lf' % (data_section, finish_time - start_time))


if __name__ == '__main__':
    # bert_base_encode_vocab('bert-base-uncased', 'wiki_winsize1')
    # bert_base_encode_vocab('bert-base-uncased', 'wiki_winsize3')
    # bert_base_encode_vocab('bert-base-uncased', 'wiki_winsize5')

    # bert_base_encode_sentences('bert-base-chinese', 'qd_winsize1')
    bert_base_encode_sentences('bert-base-chinese', 'qd_winsize3')
    bert_base_encode_sentences('bert-base-chinese', 'qd_winsize5')
    bert_base_encode_sentences('bert-base-uncased', 'wiki_winsize1')
    bert_base_encode_sentences('bert-base-uncased', 'wiki_winsize3')
    bert_base_encode_sentences('bert-base-uncased', 'wiki_winsize5')
    
