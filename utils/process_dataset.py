import os
from pathlib import Path
import csv
import argparse
import random
import numpy as np
import os.path as osp
import mxnet as mx
from tqdm import tqdm
import numbers
from _collections import defaultdict
import glob
import re
random.seed(666)
np.random.seed(666)


def gen_ms1m_total_list(root):
    path_imgrec = osp.join(root, 'train.rec')
    path_imgidx = osp.join(root, 'train.idx')
    imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')
    s = imgrec.read_idx(0)
    header, _ = mx.recordio.unpack(s)

    if header.flag > 0:
        header0 = (int(header.label[0]), int(header.label[1]))
        imgidx = np.array(range(1, int(header.label[0])))
    else:
        imgidx = np.array(list(imgrec.keys))
    with open(osp.join(root,'imlist_label.txt'),'w') as f:
        for index in tqdm(range(len(imgidx))):
            idx = imgidx[index]
            s = imgrec.read_idx(idx)
            header, img = mx.recordio.unpack(s)
            label = header.label
            if not isinstance(label, numbers.Number):
                label = label[0]
            f.write(f'{index} {int(label)}\n')



def generate_gallery_list(source_file, saved_file):
    saved_file = open(saved_file, 'w')
    csv_reader = csv.reader(open(source_file, encoding='utf-8'))
    csv_reader.__next__()
    count = 0
    for row in csv_reader:
        saved_file.write("index/%s/%s/%s/%s.jpg %d\n"%(row[0][0],row[0][1],row[0][2],row[0],count) )
        count += 1
    saved_file.close()
    csv_reader.close()
    print("Gallery indices are built.")

def generate_query_list(source_file, ref_file, saved_file_root, type='private'):
    query_list_file = open(os.path.join(saved_file_root, f'gldv2_{type}_query_list.txt'), 'w')
    query_gts_file = open(os.path.join(saved_file_root, f'gldv2_{type}_query_gt.txt'), 'w')

    gallery_dict = {}
    with open(ref_file, 'r') as f:
        for line in f.readlines():
            key = line.split(" ")[0].split("/")[-1][:-4]
            value = line.split(" ")[1].replace("\n","")
            gallery_dict[key] = value

    print("Gallery dict is built.")

    csv_reader = csv.reader(open(source_file, encoding='utf-8'))
    csv_reader.__next__()
    count = 0
    for row in csv_reader:
        if row[2].lower() == type:
            query_list_file.write("test/%s/%s/%s/%s.jpg %d\n" % (row[0][0], row[0][1], row[0][2], row[0], count))
            gts = []
            for gt in row[1].split(" "):
                gts.append(gallery_dict[gt])
            gts_str = ','.join(gts).replace('\n','')
            query_gts_file.write("test/%s/%s/%s/%s.jpg %d %s\n" % (row[0][0], row[0][1], row[0][2], row[0], count, gts_str))
            count += 1
        else:
            continue
    query_list_file.close()
    query_gts_file.close()
    csv_reader.close()
    print("Query indices are built.")
def split_ms1m_by_class(input_file, ratio=0.3, dataset='ms1m'):
    input_file = Path(input_file)
    bucket = defaultdict(list)
    random.seed(666)
    np.random.seed(666)
    with open(input_file, 'r') as f:
        for line in f.readlines():
            bucket[int(line.split(" ")[-1])].append(line)
    classes = len([*bucket])
    all_class_list = np.arange(0, classes)
    np.random.shuffle(all_class_list)
    old_class_list = all_class_list[:int(classes * ratio)]
    old_class_bool_index = np.array([False for _ in range(classes)])
    old_class_bool_index[old_class_list] = True

    old_class_file = open(
        input_file.parent / f"{dataset}_train_old_{int(ratio * 100)}percent_class.txt", "w")
    new_class_file = open(
        input_file.parent / f"{dataset}_train_new_100percent_class.txt", "w")

    old_class_num, old_img_num = 0, 0
    new_class_num, new_img_num = 0, 0
    for i in range(classes):
        if old_class_bool_index[i]:
            old_cls =  old_class_num
            for j in bucket[i]:
                path = j.split()[0]
                old_class_file.write(f'{path} {old_class_num}\n')
                old_img_num += 1
            old_class_num += 1
    old_cls_counter = 0
    for i in range(classes):
        if old_class_bool_index[i]:
            cur_cls = old_cls_counter
            old_cls_counter += 1
        else:
            cur_cls = old_class_num + new_class_num
            new_class_num += 1
        for j in bucket[i]:
            path,cls = j.split()
            new_class_file.write(f'{path} {cur_cls}\n')
            new_img_num += 1
    old_class_file.close()
    new_class_file.close()
    print("Old class count: %d, img count: %d" % (old_class_num, old_img_num))
    print("New class count: %d, img count: %d" % (new_class_num, new_img_num))
    print("Done.")

def split_gldv2_by_class(input_file, classes, ratio=0.3, dataset='gldv2'):
    input_file = Path(input_file)
    bucket = [[] for _ in range(classes)]
    random.seed(666)
    np.random.seed(666)
    all_class_list = np.arange(0, classes)
    np.random.shuffle(all_class_list)
    old_class_list = all_class_list[:int(classes * ratio)]
    old_class_bool_index = np.array([False for _ in range(classes)])
    old_class_bool_index[old_class_list] = True

    with open(input_file, 'r') as f:
        for line in f.readlines():
            bucket[int(line.split(" ")[-1])].append(line)

    old_class_file = open(
        input_file.parent / f"{dataset}_train_old_{int(ratio * 100)}percent_class.txt", "w")
    new_class_file = open(
        input_file.parent / f"{dataset}_train_new_100percent_class.txt", "w")

    old_class_num, old_img_num = 0, 0
    new_class_num, new_img_num = 0, 0

    for i in range(classes):
        if old_class_bool_index[i]:
            old_cls =  old_class_num
            for j in bucket[i]:
                path = j.split()[0]
                old_class_file.write(f'{path} {old_class_num}\n')
                old_img_num += 1
            old_class_num += 1
    old_cls_counter = 0
    for i in range(classes):
        if old_class_bool_index[i]:
            cur_cls = old_cls_counter
            old_cls_counter += 1
        else:
            cur_cls = old_class_num + new_class_num
            new_class_num += 1
        for j in bucket[i]:
            path,cls = j.split()
            new_class_file.write(f'{path} {cur_cls}\n')
            new_img_num += 1
    old_class_file.close()
    new_class_file.close()
    print("Old class count: %d, img count: %d" % (old_class_num, old_img_num))
    print("New class count: %d, img count: %d" % (new_class_num, new_img_num))
    print("Done.")

def split_ms1m_by_data(input_file, ratio=0.3, dataset='ms1m'):
    input_file = Path(input_file)
    # bucket = np.zeros([classes], dtype=int)
    bucket = defaultdict(list)
    paths = []
    with open(input_file, "r") as f:
        for line in f.readlines():
            line_splits = line.split(" ")
            current_label = int(line_splits[-1]) if line_splits[-1] != '\n' else int(line_splits[-2])
            # bucket[current_label] += 1
            bucket[current_label].append(line)
            paths.append(line)

    old_training_data_file = open(
        input_file.parent/f"{dataset}_train_old_{int(ratio * 100)}percent_data.txt" , "w")

    new_training_data_file = open(
        input_file.parent / f"{dataset}_train_new_{int((1-ratio) * 100)}percent_data.txt", "w")

    start_indexes = np.zeros([len([*bucket])], dtype=int)
    old_num = 0
    new_num = 0
    for i in range(len([*bucket])):
        if len(bucket[i]) == 0:
            continue
        if i > 0:
            start_indexes[i] = start_indexes[i - 1] + len(bucket[i - 1])
        curr_count = len(bucket[i])
        if curr_count == 1:
            old_training_data_file.write(bucket[i][0])
            new_training_data_file.write(bucket[i][0])
            old_num += 1
            new_num += 1
        elif curr_count>1 and curr_count < 4:
            old_training_data_file.write(bucket[i][0])
            old_num += 1
            for j in range(1, curr_count):
                new_training_data_file.write(bucket[i][j])
                new_num += 1
        else:
            random.seed(666 + i)
            np.random.seed(666 + i)
            all_list = np.arange(0, curr_count)
            np.random.shuffle(all_list)
            old_list = all_list[:int(curr_count * ratio)]
            # old_list = np.array(sorted(random.sample(range(0, current_count), int(current_count * ratio))), dtype=int)
            old_bool_index = np.array([False for _ in range(curr_count)])
            old_bool_index[old_list] = True
            for j, element in enumerate(bucket[i]):
                if old_bool_index[j]:
                    old_training_data_file.write(element)
                    old_num += 1
                else:
                    new_training_data_file.write(element)
                    new_num += 1

    old_training_data_file.close()
    new_training_data_file.close()
    print("Old data count: %d" % old_num)
    print("New data count: %d" % new_num)
    print("Done.")

def split_gldv2_by_data(input_file, classes, ratio=0.3, dataset='gldv2'):
    input_file = Path(input_file)
    # bucket = np.zeros([classes], dtype=int)
    from _collections import defaultdict
    bucket = defaultdict(list)
    paths = []
    with open(input_file, "r") as f:
        for line in f.readlines():
            line_splits = line.split(" ")
            current_label = int(line_splits[-1]) if line_splits[-1] != '\n' else int(line_splits[-2])
            # bucket[current_label] += 1
            bucket[current_label].append(line)
            paths.append(line)

    old_training_data_file = open(
        input_file.parent/f"{dataset}_train_old_{int(ratio * 100)}percent_data.txt" , "w")

    new_training_data_file = open(
        input_file.parent / f"{dataset}_train_new_{int((1-ratio) * 100)}percent_data.txt", "w")

    start_indexes = np.zeros([classes], dtype=int)
    old_num = 0
    new_num = 0
    for i in range(classes):
        if len(bucket[i]) == 0:
            continue
        if i > 0:
            start_indexes[i] = start_indexes[i - 1] + len(bucket[i - 1])
        curr_count = len(bucket[i])
        if curr_count == 1:
            old_training_data_file.write(bucket[i][0])
            new_training_data_file.write(bucket[i][0])
            old_num += 1
            new_num += 1
        elif curr_count>1 and curr_count < 4:
            old_training_data_file.write(bucket[i][0])
            old_num += 1
            for j in range(1, curr_count):
                new_training_data_file.write(bucket[i][j])
                new_num += 1
        else:
            random.seed(666 + i)
            np.random.seed(666 + i)
            all_list = np.arange(0, curr_count)
            np.random.shuffle(all_list)
            old_list = all_list[:int(curr_count * ratio)]
            # old_list = np.array(sorted(random.sample(range(0, current_count), int(current_count * ratio))), dtype=int)
            old_bool_index = np.array([False for _ in range(curr_count)])
            old_bool_index[old_list] = True
            for j, element in enumerate(bucket[i]):
                if old_bool_index[j]:
                    old_training_data_file.write(element)
                    old_num += 1
                else:
                    new_training_data_file.write(element)
                    new_num += 1

    old_training_data_file.close()
    new_training_data_file.close()
    print("Old data count: %d" % old_num)
    print("New data count: %d" % new_num)
    print("Done.")

def gen_reid_imlist(dir_path,pid_num=1501,camid_num=6,pid_begin=0):
    img_paths = glob.glob(osp.join(dir_path,'bounding_box_train', '*.jpg'))
    pattern = re.compile(r'([-\d]+)_c(\d)')

    pid_container = set()
    for img_path in sorted(img_paths):
        pid, _ = map(int, pattern.search(img_path).groups())
        if pid == -1: continue  # junk images are just ignored
        pid_container.add(pid)
    pid2label = {pid: label for label, pid in enumerate(pid_container)}
    dataset = []
    with open(osp.join(dir_path, 'imlist_label.txt'), 'w') as f:
        for img_path in sorted(img_paths):
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            assert 0 <= pid <= pid_num  # pid == 0 means background
            assert 1 <= camid <= camid_num
            camid -= 1  # index starts from 0
            pid = pid2label[pid]
            f.write(f'{img_path} {pid_begin + pid} {camid} {1}\n')
            dataset.append((img_path, pid_begin + pid, camid, 1))

def split_reid_by_data(input_file, ratio=0.3, dataset='market1501'):
    input_file = Path(input_file)
    # bucket = np.zeros([classes], dtype=int)
    bucket = defaultdict(list)
    paths = []
    with open(input_file, "r") as f:
        for line in f.readlines():
            line_splits = line.split(" ")
            current_label = line_splits[1]
            # bucket[current_label] += 1
            bucket[current_label].append(line)
            paths.append(line)
    old_training_data_file = open(
        input_file.parent/f"{dataset}_train_old_{int(ratio * 100)}percent_data.txt" , "w")

    new_training_data_file = open(
        input_file.parent / f"{dataset}_train_new_{int((1-ratio) * 100)}percent_data.txt", "w")

    old_num = 0
    new_num = 0
    for i in [*bucket]:
        if len(bucket[i]) == 0:
            continue
        curr_count = len(bucket[i])
        if curr_count == 1:
            old_training_data_file.write(bucket[i][0])
            new_training_data_file.write(bucket[i][0])
            old_num += 1
            new_num += 1
        elif curr_count>1 and curr_count < 4:
            old_training_data_file.write(bucket[i][0])
            old_num += 1
            for j in range(1, curr_count):
                new_training_data_file.write(bucket[i][j])
                new_num += 1
        else:
            random.seed(666 + int(i))
            np.random.seed(666 + int(i))
            all_list = np.arange(0, curr_count)
            np.random.shuffle(all_list)
            old_list = all_list[:int(curr_count * ratio)]
            # old_list = np.array(sorted(random.sample(range(0, current_count), int(current_count * ratio))), dtype=int)
            old_bool_index = np.array([False for _ in range(curr_count)])
            old_bool_index[old_list] = True
            for j, element in enumerate(bucket[i]):
                if old_bool_index[j]:
                    old_training_data_file.write(element)
                    old_num += 1
                else:
                    new_training_data_file.write(element)
                    new_num += 1

    old_training_data_file.close()
    new_training_data_file.close()
    print("Old data count: %d" % old_num)
    print("New data count: %d" % new_num)
    print("Done.")

def split_reid_by_class(input_file, ratio=0.3, dataset='ms1m'):
    input_file = Path(input_file)
    bucket = defaultdict(list)
    random.seed(666)
    np.random.seed(666)
    with open(input_file, 'r') as f:
        for line in f.readlines():
            bucket[line.split(" ")[1]].append(line)
    classes = len([*bucket])
    all_class_list = np.arange(0, classes)
    np.random.shuffle(all_class_list)
    old_class_list = all_class_list[:int(classes * ratio)]
    old_class_bool_index = np.array([False for _ in range(classes)])
    old_class_bool_index[old_class_list] = True

    old_class_file = open(
        input_file.parent / f"{dataset}_train_old_{int(ratio * 100)}percent_class.txt", "w")
    new_class_file = open(
        input_file.parent / f"{dataset}_train_new_100percent_class.txt", "w")

    old_class_num, old_img_num = 0, 0
    new_class_num, new_img_num = 0, 0

    for i in range(classes):
        if old_class_bool_index[i]:
            old_cls =  old_class_num
            for j in bucket[i]:
                path = j.split()[0]
                old_class_file.write(f'{path} {old_class_num}\n')
                old_img_num += 1
            old_class_num += 1
    old_cls_counter = 0
    for i in range(classes):
        if old_class_bool_index[i]:
            cur_cls = old_cls_counter
            old_cls_counter += 1
        else:
            cur_cls = old_class_num + new_class_num
            new_class_num += 1
        for j in bucket[i]:
            path,cls = j.split()
            new_class_file.write(f'{path} {cur_cls}\n')
            new_img_num += 1
    old_class_file.close()
    new_class_file.close()
    print("Old class count: %d, img count: %d" % (old_class_num, old_img_num))
    print("New class count: %d, img count: %d" % (new_class_num, new_img_num))
    print("Done.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--root_path', default='data/GLDv2/', type=str, help='')
    parser.add_argument('--gene_gallery', action='store_true')
    parser.add_argument('--gene_query', action='store_true')
    parser.add_argument('--split_file', default='data/GLDv2/label_81313.txt', type=str, help='')
    parser.add_argument('--dataset', default='gldv2', type=str, help='')
    parser.add_argument('--split_by_data', action='store_true')
    parser.add_argument('--split_by_class', action='store_true')
    parser.add_argument('--split_ratio', default=0.3, type=float, help='')
    args = parser.parse_args()

    root_path = args.root_path
    Path(root_path).mkdir(parents=True, exist_ok=True)
    if args.dataset == 'gldv2':
        if args.gene_gallery:
            generate_gallery_list(os.path.join(root_path, 'index.csv'),
                                  os.path.join(root_path, 'gldv2_gallery_list.txt'))

        if args.gene_query:
            assert os.path.isfile(os.path.join(root_path, 'retrieval_solution_v2.1.csv')), \
                "Please generate gallery indices first"
            generate_query_list(os.path.join(root_path, 'retrieval_solution_v2.1.csv'),
                                os.path.join(root_path, 'gldv2_gallery_list.txt'),
                                root_path, type='private')

            generate_query_list(os.path.join(root_path, 'retrieval_solution_v2.1.csv'),
                                os.path.join(root_path, 'gldv2_gallery_list.txt'),
                                root_path, type='public')

        cls_num_dic = {'gldv2': 81313, 'imagenet': 1000, 'places365': 365, 'market': 1502}
        if args.split_by_data:
            whole_training_file = os.path.join(root_path, 'label_81313.txt')
            assert os.path.isfile(whole_training_file), \
                "Please download label_81313.txt first."
            split_gldv2_by_data(whole_training_file, cls_num_dic['gldv2'], args.split_ratio, dataset='gldv2')

        if args.split_by_class:
            whole_training_file = os.path.join(root_path, 'label_81313.txt')
            assert os.path.isfile(whole_training_file), \
                "Please download label_81313.txt first."
            split_gldv2_by_class(whole_training_file, cls_num_dic['gldv2'], args.split_ratio, dataset='gldv2')
    elif args.dataset == 'ms1m':
        if not osp.exists(osp.join(args.root_path,'imlist_label.txt')):
            gen_ms1m_total_list(args.root_path)
            print('generate images list with labels of ms1m')
        whole_training_file = os.path.join(root_path, 'imlist_label.txt')
        if args.split_by_data:
            assert os.path.isfile(whole_training_file), \
                "Please generate imlist_label.txt first."
            split_ms1m_by_data(whole_training_file, args.split_ratio, dataset='ms1m')
        if args.split_by_class:
            assert os.path.isfile(whole_training_file), \
                "Please generate imlist_label.txt first."
            split_ms1m_by_class(whole_training_file, args.split_ratio, dataset='ms1m')
    elif args.dataset == 'market1501':
        if not osp.exists(osp.join(args.root_path,'imlist_label.txt')):
            gen_reid_imlist(args.root_path)
        whole_training_file = os.path.join(root_path, 'imlist_label.txt')
        if args.split_by_data:
            assert os.path.isfile(whole_training_file), \
                "Please generate imlist_label.txt first."
            split_reid_by_data(whole_training_file, args.split_ratio, dataset='market1501')
        else:
            assert os.path.isfile(whole_training_file), \
                "Please generate imlist_label.txt first."
            split_reid_by_class(whole_training_file, args.split_ratio, dataset='market1501')
