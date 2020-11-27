import os
import sys
import json
from module.vocabulary import Vocab

sys.path.append('/share/wangyq/tools/')
import rouge_server


def result_word2id(vocab, hyps, refer):
    '''
    params:
        hyps / refer: [[sen1, sen2], [sen1, sen2], ...]
    '''
    hyps_id, refer_id = [], []
    for h, r in zip(hyps, refer):
        hyps_id_new, refer_id_new = [], []
        for sen in h:
            ids = [str(vocab.word2id(tok)) for tok in sen.split(' ') if tok]
            assert '1' not in ids, 'UNK detected in sentence'
            hyps_id_new.append(' '.join(ids))
        for sen in r:
            ids = [str(vocab.word2id(tok)) for tok in sen.split(' ') if tok]
            assert '1' not in ids, 'UNK detected in sentence'
            refer_id_new.append(' '.join(ids))
        hyps_id.append(hyps_id_new)
        refer_id.append(refer_id_new)
    return hyps_id, refer_id


if __name__ == "__main__":
    vocab_file = '/share/wangyq/project/HeterSumGraph/cache/winsize1_random_cut/test_vocab'

    # vocab_size = int(os.popen('wc -l {}'.format(vocab_file)).read().split()[0])
    vocab = Vocab(vocab_file, -1)

    hyps, refer = [], []
    decode_fd = open('./save/20201126_204639_rc1/test/winsize1_random_cut', encoding='utf-8')
    for line in decode_fd:
        line_dict = json.loads(line)
        hyps.append(line_dict['hyps'].split('\n'))
        refer.append(line_dict['refer'].split('\n'))

    hyps_id, refer_id = result_word2id(vocab, hyps, refer)
    rouge_server.eval_rouge(project_name='wyq_structural_summ', exp_name='myHeterSumGraph_test_HSG_winsize1_random_cut',
                            run_name='decode_test_ckpt-22016', decoded=hyps_id, reference=refer_id)
