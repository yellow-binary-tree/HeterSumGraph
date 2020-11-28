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

import argparse
import datetime
import os
import sys
import time
import json

import torch
from rouge import Rouge

from HiGraph import HSumGraph, HSumDocGraph
from Tester import SLTester
from module.dataloader import MapDataset, graph_collate_fn
from module.embedding import Word_Embedding
from module.vocabulary import Vocab
from tools import utils
import logging
from tools.logger import logger, formatter
from myutils import result_word2id

# exp_uploader
sys.path.append('/share/wangyq/tools/')
import exp_uploader
import rouge_server


def load_test_model(model, model_name, eval_dir, save_root):
    """ choose which model will be loaded for evaluation """
    if model_name.startswith('eval'):
        bestmodel_load_path = os.path.join(eval_dir, model_name[4:])
    elif model_name.startswith('train'):
        train_dir = os.path.join(save_root, "train")
        bestmodel_load_path = os.path.join(train_dir, model_name[5:])
    elif model_name == "earlystop":
        train_dir = os.path.join(save_root, "train")
        bestmodel_load_path = os.path.join(train_dir, 'earlystop')
    else:
        logger.error("None of such model! Must be one of evalbestmodel/trainbestmodel/earlystop")
        raise ValueError("None of such model! Must be one of evalbestmodel/trainbestmodel/earlystop")
    if not os.path.exists(bestmodel_load_path):
        logger.error("[ERROR] Restoring %s for testing...The path %s does not exist!", model_name, bestmodel_load_path)
        return None
    logger.info("[INFO] Restoring %s for testing...The path is %s", model_name, bestmodel_load_path)

    model.load_state_dict(torch.load(bestmodel_load_path))

    return model


def run_test(model, dataset, loader, model_name, hps):
    test_dir = os.path.join(hps.save_root, "test")      # make a subdir of the root dir for eval data
    eval_dir = os.path.join(hps.save_root, "eval")
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    if not os.path.exists(eval_dir):
        logger.exception("[Error] eval_dir %s doesn't exist. Run in train mode to create it.", eval_dir)
        raise Exception("[Error] eval_dir %s doesn't exist. Run in train mode to create it." % (eval_dir))

    resfile = None
    if hps.save_label:
        log_dir = os.path.join(test_dir, hps.cache_dir.split("/")[-1] + '_' + model_name)
        resfile = open(log_dir, "w", encoding='utf-8')
        logger.info("[INFO] Write the decode result into %s", log_dir)

    model = load_test_model(model, model_name, eval_dir, hps.save_root)
    model.eval()

    # exp_uploader
    exp = exp_uploader.Exp(proj_name=hps.proj_name, exp_name=hps.exp_name, command=str(hps))
    exp_uploader.init_exp(exp)
    if hps.use_exp_rouge:
        test_vocab = Vocab(os.path.join(hps.cache_dir, 'test_vocab'), max_size=-1)

    iter_start_time = time.time()
    with torch.no_grad():
        logger.info("[Model] Sequence Labeling!")
        tester = SLTester(model, hps, test_dir=test_dir, limited=hps.limited)

        for i, (G, index) in enumerate(loader):
            if hps.cuda:
                G.to(torch.device("cuda"))

            pred_idxs, hypss, refers, labels = tester.evaluation(G, index, dataset, blocking=hps.blocking)

            if hps.save_label:
                for i, pred_idx, hyps, refer, label in zip(index, pred_idxs, hypss, refers, labels):
                    resfile.write(json.dumps(
                        {'index': i, 'pred_idx': pred_idx, 'label': label, 'hyps': hyps, 'refer': refer},
                        ensure_ascii=False) + '\n')

            if i % 20 == 0:
                exp_uploader.async_heart_beat(exp)

    running_avg_loss = tester.running_avg_loss

    logger.info("The number of pairs is %d", tester.rougePairNum)
    if not tester.rougePairNum:
        logger.error("During testing, no hyps is selected!")
        sys.exit(1)

    if hps.use_exp_rouge:
        exp_server_hyps, exp_server_refer = result_word2id(
            test_vocab, [chap.split('\n') for chap in tester.hyps], [chap.split('\n') for chap in tester.refer])
        rouge_server.eval_rouge(hps.proj_name, hps.exp_name, 'decode_test_ckpt-{}'.format(hps.test_model.split('_')[-1]),
                                exp_server_hyps, exp_server_refer)

    if hps.use_pyrouge:
        if isinstance(tester.refer[0], list):
            logger.info("Multi Reference summaries!")
            scores_all = utils.pyrouge_score_all_multi(tester.hyps, tester.refer)
        else:
            scores_all = utils.pyrouge_score_all(tester.hyps, tester.refer)
    else:
        rouge = Rouge()
        scores_all = rouge.get_scores(tester.hyps, tester.refer, avg=True)

    res = "Rouge1:\n\tp:%.6f, r:%.6f, f:%.6f\n" % (scores_all['rouge-1']['p'], scores_all['rouge-1']['r'], scores_all['rouge-1']['f']) \
        + "Rouge2:\n\tp:%.6f, r:%.6f, f:%.6f\n" % (scores_all['rouge-2']['p'], scores_all['rouge-2']['r'], scores_all['rouge-2']['f']) \
        + "Rougel:\n\tp:%.6f, r:%.6f, f:%.6f\n" % (scores_all['rouge-l']['p'], scores_all['rouge-l']['r'], scores_all['rouge-l']['f'])
    logger.info(res)

    tester.getMetric()
    # tester.SaveDecodeFile()
    logger.info('[INFO] End of test | time: {:5.2f}s | test loss {:5.4f} | '.format((time.time() - iter_start_time), float(running_avg_loss)))


def main():
    parser = argparse.ArgumentParser(description='HeterSumGraph Model')

    # Where to find data
    parser.add_argument('--data_dir', type=str, default='data/CNNDM', help='The dataset directory.')
    parser.add_argument('--cache_dir', type=str, default='cache/CNNDM', help='The processed dataset directory')
    parser.add_argument('--embedding_path', type=str, default='/remote-home/dqwang/Glove/glove.42B.300d.txt', help='Path expression to external word embedding.')

    # Important settings
    parser.add_argument('--model', type=str, default="HSumGraph", help="model structure[HSG|HDSG]")
    parser.add_argument('--test_model', type=str, default='evalbestmodel', help='choose different model to test [multi/evalbestmodel/trainbestmodel/earlystop]')
    parser.add_argument('--use_pyrouge', action='store_true', default=False, help='use_pyrouge')
    parser.add_argument('--num_workers', default=0, help='num_workers of the dataset')

    # Where to save output
    parser.add_argument('--save_root', type=str, default='save/', help='Root directory for all model.')
    parser.add_argument('--log_root', type=str, default='log/', help='Root directory for all logging.')

    # Hyperparameters
    parser.add_argument('--gpu', type=str, default='0', help='GPU ID to use')
    parser.add_argument('--cuda', action='store_true', default=False, help='use cuda')
    parser.add_argument('--vocab_size', type=int, default=50000, help='Size of vocabulary.')
    parser.add_argument('--batch_size', type=int, default=32, help='Mini batch size [default: 32]')
    parser.add_argument('--n_iter', type=int, default=1, help='iteration ')

    parser.add_argument('--word_embedding', action='store_true', default=True, help='whether to use Word embedding')
    parser.add_argument('--word_emb_dim', type=int, default=300, help='Word embedding size [default: 300]')
    parser.add_argument('--embed_train', action='store_true', default=False, help='whether to train Word embedding [default: False]')
    parser.add_argument('--feat_embed_size', type=int, default=50, help='feature embedding size [default: 50]')
    parser.add_argument('--n_layers', type=int, default=1, help='Number of GAT layers [default: 1]')
    parser.add_argument('--lstm_hidden_state', type=int, default=128, help='size of lstm hidden state')
    parser.add_argument('--lstm_layers', type=int, default=2, help='lstm layers')
    parser.add_argument('--bidirectional', action='store_true', default=True, help='use bidirectional LSTM')
    parser.add_argument('--n_feature_size', type=int, default=128, help='size of node feature')
    parser.add_argument('--hidden_size', type=int, default=64, help='hidden size [default: 64]')
    parser.add_argument('--gcn_hidden_size', type=int, default=128, help='hidden size [default: 64]')
    parser.add_argument('--ffn_inner_hidden_size', type=int, default=512, help='PositionwiseFeedForward inner hidden size [default: 512]')
    parser.add_argument('--n_head', type=int, default=8, help='multihead attention number [default: 8]')
    parser.add_argument('--recurrent_dropout_prob', type=float, default=0.1, help='recurrent dropout prob [default: 0.1]')
    parser.add_argument('--atten_dropout_prob', type=float, default=0.1, help='attention dropout prob [default: 0.1]')
    parser.add_argument('--ffn_dropout_prob', type=float, default=0.1, help='PositionwiseFeedForward dropout prob [default: 0.1]')
    parser.add_argument('--use_orthnormal_init', action='store_true', default=True, help='use orthnormal init for lstm [default: true]')
    parser.add_argument('--sent_max_len', type=int, default=100, help='max length of sentences (max source text sentence tokens)')
    parser.add_argument('--doc_max_timesteps', type=int, default=50, help='max length of documents (max timesteps of documents)')
    parser.add_argument('--save_label', action='store_true', default=False, help='require multihead attention')
    parser.add_argument('--limited', action='store_true', default=False, help='limited hypo length')
    parser.add_argument('--blocking', action='store_true', default=False, help='ngram blocking')

    parser.add_argument('-m', type=int, default=3, help='decode summary length')

    # exp_upload
    parser.add_argument('--proj_name', type=str, default='wyq_structural_summ', help='Project Name')
    parser.add_argument('--exp_name', type=str, default='myHeterSumGrpah', help='Experiment Name')
    parser.add_argument('--use_exp_rouge', type=bool, default=True, help='whether send decoded summ to the exp_server to get rouge')

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    torch.set_printoptions(threshold=50000)

    # File paths
    VOCAL_FILE = os.path.join(args.cache_dir, "vocab")
    LOG_PATH = args.log_root

    # train_log setting
    if not os.path.exists(LOG_PATH):
        logger.exception("[Error] Logdir %s doesn't exist. Run in train mode to create it.", LOG_PATH)
        raise Exception("[Error] Logdir %s doesn't exist. Run in train mode to create it." % (LOG_PATH))
    nowTime = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(LOG_PATH, "test_" + nowTime)
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info("Pytorch %s", torch.__version__)
    logger.info("[INFO] Create Vocab, vocab path is %s", VOCAL_FILE)
    vocab = Vocab(VOCAL_FILE, args.vocab_size)
    embed = torch.nn.Embedding(vocab.size(), args.word_emb_dim)
    if args.word_embedding:
        embed_loader = Word_Embedding(args.embedding_path, vocab)
        vectors = embed_loader.load_my_vecs(args.word_emb_dim)
        pretrained_weight = embed_loader.add_unknown_words_by_avg(vectors, args.word_emb_dim)
        embed.weight.data.copy_(torch.Tensor(pretrained_weight))
        embed.weight.requires_grad = args.embed_train

    hps = args
    logger.info(hps)

    if hps.model == "HSG":
        model = HSumGraph(hps, embed)
        logger.info("[MODEL] HeterSumGraph ")
    elif hps.model == "HDSG":
        model = HSumDocGraph(hps, embed)
        logger.info("[MODEL] HeterDocSumGraph ")
    else:
        logger.error("[ERROR] Invalid Model Type!")
        raise NotImplementedError("Model Type has not been implemented")

    # HINT: now we are trying to decode val data
    # change dataset mode to "test" if you want to decode test data.
    dataset = MapDataset(hps, mode='val')
    loader = torch.utils.data.DataLoader(dataset, batch_size=hps.batch_size, shuffle=False, collate_fn=graph_collate_fn, num_workers=args.num_workers, pin_memory=True)

    if args.cuda:
        model.to(torch.device("cuda:0"))
        logger.info("[INFO] Use cuda")

    logger.info("[INFO] Decoding...")
    if hps.test_model == "multi":
        for i in range(3):
            model_name = "evalbestmodel_%d" % i
            run_test(model, dataset, loader, model_name, hps)
    else:
        run_test(model, dataset, loader, hps.test_model, hps)


if __name__ == '__main__':
    main()
