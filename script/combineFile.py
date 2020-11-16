import os
import argparse

def combine_file(input_folder, output_folder, output_filename, length):
    total_lines = 0
    fout = open(os.path.join(output_folder, output_filename), 'w')
    for i in range(length):
        if i < 10:
            input_filename = os.path.join(input_folder, output_filename + '_0' + str(i))
        else:
            input_filename = os.path.join(input_folder, output_filename + '_' + str(i))
        fin = open(input_filename, encoding='utf-8')
        for line in fin:
            total_lines += 1
            fout.write(line)
        fin.close()
    fout.close()
    print('total lines of %s: %d' % (output_filename, total_lines))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', type=str, default='cache/CNNDM/train_split', help='File to deal with')
    parser.add_argument('--output_folder', type=str, default='cache/CNNDM', help='dataset name')
    args = parser.parse_args()

    length = len(os.listdir(args.input_folder))
    print(length)
    combine_file(args.input_folder, args.output_folder, 'train.w2s.tfidf.jsonl', length // 2)
    # combine_file(args.input_folder, args.output_folder,'train.w2d.tfidf.jsonl', length // 2)

if __name__ == "__main__":
    main()