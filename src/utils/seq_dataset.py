import tqdm
import numpy as np
import torch
import os
from scipy.sparse import csr_matrix
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import random

class SeqRecDataset(Dataset):
    def __init__(self, user_seq, n_items, test_neg_items=None, data_type='train'):
        #added instead of args
        self.item_size = n_items
        self.max_len = 50
        self.model_type = 'bsarec'
        self.data_dir = './data/baby/'
        self.data_name = 'new_baby.txt'
        self.same_target_path = os.path.join(self.data_dir, self.data_name+'_same_target.npy')
        self.batch_size = 256
        self.num_workers = 4
        self.data_file = self.data_dir + self.data_name + '.txt'

        #self.user_seq = []
        #self.user_seq = {}
        self.user_to_seq = {}
        #self.max_len = args.max_seq_length
        self.user_ids = []
        self.contrastive_learning = self.model_type.lower() in ['fearec', 'duorec']
        self.data_type = data_type

        # if self.data_type=='train':
        #     for user, seq in enumerate(user_seq):
        #         input_ids = seq[-(self.max_len + 2):-2]
        #         for i in range(len(input_ids)):
        #             self.user_seq.append(input_ids[:i + 1])
        #             self.user_ids.append(user)
        # elif self.data_type=='valid':
        #     for sequence in user_seq:
        #         self.user_seq.append(sequence[:-1])
        # else:
        #     self.user_seq = user_seq
        #
        # self.test_neg_items = test_neg_items

        # if self.data_type == 'train':
        #     for user, seq in enumerate(user_seq):
        #         # Store the entire sequence for the user
        #         input_ids = seq[:-2]  # Exclude the last two items (e.g., for validation/test)
        #         self.user_seq[user] = input_ids
        #         self.user_ids.append(user)
        if self.data_type == 'train':
            for user, seq in enumerate(user_seq):
                # Exclude the last two items and limit sequence length
                input_ids = seq[-(self.max_len + 2):-2]
                # Generate all possible subsequences
                subsequences = [input_ids[:i + 1] for i in range(len(input_ids))]
                self.user_to_seq[user] = subsequences
                self.user_ids.append(user)
        elif self.data_type == 'valid':
            for user, seq in enumerate(user_seq):
                input_ids = seq[:-1]  # Exclude the last item
                self.user_to_seq[user] = [input_ids]
                self.user_ids.append(user)
        else:
            # For test or other data types
            for user, seq in enumerate(user_seq):
                self.user_to_seq[user] = [seq]
                self.user_ids.append(user)

        self.num_users = len(self.user_ids)

        if self.contrastive_learning and self.data_type=='train':
            if os.path.exists(self.same_target_path):
                self.same_target_index = np.load(self.same_target_path, allow_pickle=True)
            else:
                print("Start making same_target_index for contrastive learning")
                self.same_target_index = self.get_same_target_index()
                self.same_target_index = np.array(self.same_target_index)
                np.save(self.same_target_path, self.same_target_index)

    def get_same_target_index(self):
        num_items = max([max(v) for v in self.user_seq]) + 2
        same_target_index = [[] for _ in range(num_items)]

        user_seq = self.user_seq[:]
        tmp_user_seq = []
        for i in tqdm.tqdm(range(1, num_items)):
            for j in range(len(user_seq)):
                if user_seq[j][-1] == i:
                    same_target_index[i].append(user_seq[j])
                else:
                    tmp_user_seq.append(user_seq[j])
            user_seq = tmp_user_seq
            tmp_user_seq = []

        return same_target_index

    def __len__(self):
        return self.num_users

    def __getitem__(self, index):
        # items = self.user_seq[index]
        # input_ids = items[:-1]
        # answer = items[-1]

        user_id = self.user_ids[index]
        subsequences = self.user_to_seq[user_id]

        if self.data_type == 'train':
            # Randomly select one subsequence
            items = random.choice(subsequences)
        else:
            # For validation and testing, use the full sequence
            items = subsequences[0]

        # The last item is the answer; the rest are input_ids
        answer = items[-1]
        input_ids = items[:-1]

        # if self.data_type == 'train':
        #     # Use the last item as the answer
        #     answer = items[-1]
        #     input_ids = items[:-1]
        # elif self.data_type == 'valid':
        #     # For validation, the last item is the answer
        #     answer = items[-1]
        #     input_ids = items[:-1]
        # else:
        #     # For testing, the last item is the answer
        #     answer = items[-1]
        #     input_ids = items[:-1]

        seq_set = set(items)
        neg_answer = neg_sample(seq_set, self.item_size)

        pad_len = self.max_len - len(input_ids)
        input_ids = [0] * pad_len + input_ids
        input_ids = input_ids[-self.max_len:]
        assert len(input_ids) == self.max_len

        # pad_len = (self.max_len-1) - len(input_ids)
        # input_ids = [0] * pad_len + input_ids
        # input_ids = input_ids[-(self.max_len-1):]
        # assert len(input_ids) == (self.max_len-1)

        if self.data_type in ['valid', 'test']:
            cur_tensors = (
                torch.tensor(index, dtype=torch.long),  # user_id for testing
                torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(answer, dtype=torch.long),
                torch.zeros(0, dtype=torch.long), # not used
                torch.zeros(0, dtype=torch.long), # not used
            )

        elif self.contrastive_learning:
            sem_augs = self.same_target_index[answer]
            sem_aug = random.choice(sem_augs)
            keep_random = False
            for i in range(len(sem_augs)):
                if sem_augs[0] != sem_augs[i]:
                    keep_random = True

            while keep_random and sem_aug == items:
                sem_aug = random.choice(sem_augs)

            sem_aug = sem_aug[:-1]
            pad_len = self.max_len - len(sem_aug)
            sem_aug = [0] * pad_len + sem_aug
            sem_aug = sem_aug[-self.max_len:]
            assert len(sem_aug) == self.max_len

            cur_tensors = (
                torch.tensor(self.user_ids[index], dtype=torch.long),  # user_id for testing
                torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(answer, dtype=torch.long),
                torch.tensor(neg_answer, dtype=torch.long),
                torch.tensor(sem_aug, dtype=torch.long)
            )

        else:
            cur_tensors = (
                torch.tensor(self.user_ids[index], dtype=torch.long),  # user_id for testing
                torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(answer, dtype=torch.long),
                torch.tensor(neg_answer, dtype=torch.long),
                torch.zeros(0, dtype=torch.long), # not used
            )

        return cur_tensors


def neg_sample(item_set, item_size):
    item = random.randint(1, item_size - 1)
    while item in item_set:
        item = random.randint(1, item_size - 1)
    return item

def generate_rating_matrix_valid(user_seq, num_users, num_items):
    # three lists are used to construct sparse matrix
    row = []
    col = []
    data = []
    for user_id, item_list in enumerate(user_seq):
        for item in item_list[:-2]: #
            row.append(user_id)
            col.append(item)
            data.append(1)

    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    rating_matrix = csr_matrix((data, (row, col)), shape=(num_users, num_items))

    return rating_matrix

def generate_rating_matrix_test(user_seq, num_users, num_items):
    # three lists are used to construct sparse matrix
    row = []
    col = []
    data = []
    for user_id, item_list in enumerate(user_seq):
        for item in item_list[:-1]: #
            row.append(user_id)
            col.append(item)
            data.append(1)

    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    rating_matrix = csr_matrix((data, (row, col)), shape=(num_users, num_items))

    return rating_matrix

def get_rating_matrix(data_name, seq_dic, max_item):

    num_items = max_item + 1
    valid_rating_matrix = generate_rating_matrix_valid(seq_dic['user_seq'], seq_dic['num_users'], num_items)
    test_rating_matrix = generate_rating_matrix_test(seq_dic['user_seq'], seq_dic['num_users'], num_items)

    return valid_rating_matrix, test_rating_matrix

def get_user_seqs_and_max_item(data_file):
    lines = open(data_file).readlines()
    lines = lines[1:]
    user_seq = []
    item_set = set()
    for line in lines:
        user, items = line.strip().split('	', 1)
        items = items.split()
        items = [int(item) for item in items]
        user_seq.append(items)
        item_set = item_set | set(items)
    max_item = max(item_set)
    return user_seq, max_item

def get_user_seqs(data_file):
    lines = open(data_file).readlines()
    user_seq = []
    item_set = set()
    for line in lines:
        user, items = line.strip().split(' ', 1)
        items = items.split(' ')
        items = [int(item) for item in items]
        user_seq.append(items)
        item_set = item_set | set(items)
    max_item = max(item_set)
    num_users = len(lines)

    return user_seq, max_item, num_users

def get_seq_dic():
    data_dir = '../data/baby/'
    data_name = 'new_baby'
    data_file = data_dir + data_name + '.txt'
    user_seq, max_item, num_users = get_user_seqs(data_file)
    seq_dic = {'user_seq':user_seq, 'num_users':num_users }

    return seq_dic, max_item, num_users

def get_dataloder(seq_dic, item_size):
    batch_size = 256
    num_workers = 4

    train_dataset = SeqRecDataset(seq_dic['user_seq'], item_size, data_type='train')
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size, num_workers=num_workers)

    eval_dataset = SeqRecDataset(seq_dic['user_seq'], item_size, data_type='valid')
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=batch_size, num_workers=num_workers)

    test_dataset = SeqRecDataset(seq_dic['user_seq'], item_size, data_type='test')
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=batch_size, num_workers=num_workers)

    return train_dataloader, eval_dataloader, test_dataloader
