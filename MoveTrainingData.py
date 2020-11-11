import os
import argparse


def move_training_data(src_folder, dest_folder):
    types = ['train', 'val', 'test']
    for t in types:
        books = os.listdir(os.path.join(src_folder, t))
        f = open(os.path.join(dest_folder, t+'.label.jsonl'), 'w')
        for book in books:
            lines = open(os.path.join(src_folder, t, book)).read().strip()
            f.write(lines + '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--src_folder', type=str, default='/share/wangyq/data/qidian_summ', help='source training data, one book per json file.')
    parser.add_argument('--dest_folder', type=str, default='./data/qidian_1106', help='dest of training data for HeterSumGraph format.')

    args = parser.parse_args()

    move_training_data(args.src_folder, args.dest_folder)