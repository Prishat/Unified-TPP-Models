import pandas
from tqdm import tqdm
import numpy as np
import torch
from collections import Counter
import math
import random
import pickle
import os

from .misc import get_dataloader


class ATMDataset:
    def __init__(self, params, subset):
        data = pandas.read_csv(f"data/{subset}_day.csv")
        self.subset = subset
        self.id = list(data['id'])
        self.time = list(data['time'])
        self.event = list(data['event'])
        self.params = params
        self.seq_len = params['seq_len']
        self.time_seqs, self.event_seqs = self.generate_sequence()
        self.statistic()

    def generate_sequence(self):
        MAX_INTERVAL_VARIANCE = 1
        pbar = tqdm(total=len(self.id) - self.seq_len + 1)
        time_seqs = []
        event_seqs = []
        cur_end = self.seq_len - 1
        while cur_end < len(self.id):
            pbar.update(1)
            cur_start = cur_end - self.seq_len + 1
            if self.id[cur_start] != self.id[cur_end]:
                cur_end += 1
                continue

            subseq = self.time[cur_start:cur_end + 1]
            # if max(subseq) - min(subseq) > MAX_INTERVAL_VARIANCE:
            #     if self.subset == "train":
            #         cur_end += 1
            #         continue

            time_seqs.append(self.time[cur_start:cur_end + 1])
            event_seqs.append(self.event[cur_start:cur_end + 1])
            cur_end += 1
        return time_seqs, event_seqs

    def __getitem__(self, item):
        return self.time_seqs[item], self.event_seqs[item]

    def __len__(self):
        return len(self.time_seqs)

    @staticmethod
    def to_features(batch):
        times, events = [], []
        for time, event in batch:
            time = np.array([time[0]] + time)
            time = np.diff(time)
            times.append(time)
            events.append(event)
        return torch.FloatTensor(times), torch.LongTensor(events)

    def statistic(self):
        print("TOTAL SEQs:", len(self.time_seqs))
        # for i in range(10):
        #     print(self.time_seqs[i], "\n", self.event_seqs[i])
        intervals = np.diff(np.array(self.time))
        for thr in [0.001, 0.01, 0.1, 1, 10, 100]:
            print(f"<{thr} = {np.mean(intervals < thr)}")

    def importance_weight(self):
        count = Counter(self.event)
        percentage = [count[k] / len(self.event) for k in sorted(count.keys())]
        for i, p in enumerate(percentage):
            print(f"event{i} = {p * 100}%")
        weight = [len(self.event) / count[k] for k in sorted(count.keys())]
        return weight


class stackOverflow:
    def __init__(self, params, path):
        self.params = params
        self.path = path
        self.ts, self.tspan = self.load_data(1.0 / 30.0 / 24.0 / 3600.0, 1.0, 1.0)  # Put it in the config file

    def load_data(self, scale=1.0, h_dt=0.0, t_dt=0.0):
        time_seqs = []
        with open(self.path + 'time.txt') as ftime:
            seqs = ftime.readlines()
            for seq in seqs:
                time_seqs.append([float(t) for t in seq.split()])

        tmin = min([min(seq) for seq in time_seqs])
        tmax = max([max(seq) for seq in time_seqs])

        mark_seqs = []
        with open(self.path + 'event.txt') as fmark:
            seqs = fmark.readlines()
            for seq in seqs:
                mark_seqs.append([int(k) for k in seq.split()])

        m2mid = {m: mid for mid, m in enumerate(np.unique(sum(mark_seqs, [])))}

        evnt_seqs = [[((h_dt + time - tmin) * scale, m2mid[mark]) for time, mark in zip(time_seq, mark_seq)] for
                     time_seq, mark_seq in zip(time_seqs, mark_seqs)]
        random.shuffle(evnt_seqs)

        return evnt_seqs, (0.0, ((tmax + t_dt) - (tmin - h_dt)) * scale)

class thpDataloader:
    def __init__(self, params):
        self.params = params

    def load_data(self):
        """ Load data and prepare dataloader. """

        def load_data(name, dict_name):
            with open(name, 'rb') as f:
                data = pickle.load(f, encoding='latin-1')
                num_types = data['dim_process']
                data = data[dict_name]
                return data, int(num_types)

        print('[Info] Loading train data...')
        train_data, num_types = load_data(self.params['data'] + 'train.pkl', 'train')
        print('[Info] Loading dev data...')
        dev_data, _ = load_data(self.params['data'] + 'dev.pkl', 'dev')
        print('[Info] Loading test data...')
        test_data, _ = load_data(self.params['data'] + 'test.pkl', 'test')

        trainloader = get_dataloader(train_data, self.params['batch_size'], shuffle=True)
        testloader = get_dataloader(test_data, self.params['batch_size'], shuffle=False)
        return trainloader, testloader, num_types


class nhpDatareader:
    def __init__(self, params):
        self.params = params

    def read_data(self):
        with open(os.path.join(self.params['PathData'], 'train.pkl'), 'rb') as f:
            pkl_train = pickle.load(f)
        with open(os.path.join(self.params['PathData'], 'dev.pkl'), 'rb') as f:
            pkl_dev = pickle.load(f)

        data = pkl_train['seqs']
        data_dev = pkl_dev['seqs']
        total_event_num = pkl_train['total_num']
            
        return data, data_dev, total_event_num