import pandas
from tqdm import tqdm
import numpy as np

import torch
from torch.utils.data import DataLoader

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

    @staticmethod
    def process_njsde(data):

        for i in range(len(data)):
            x = data[i][0][0]

            for j in range(len(data[i][0])):
                data[i][0][j] -= x

        ts = np.swapaxes(data, 1, 2)

        ts[:, :, 1].astype(int)
        # print(ts.shape)
        # print(ts[0][2])

        l = []
        for x in ts:
            # print(x.tolist()[2])

            x = x.tolist()
            # print(len(x))
            temp = [[x[i][0], int(x[i][1])] for i in range(len(x))]

            # print(temp)
            # break
            l.append(list(map(tuple, temp)))

        tmax = l[0][0][0]
        tmin = l[0][0][0]

        for i in range(len(l)):
            for j in range(len(l[i])):
                tmax = max(tmax, l[i][j][0])
                tmin = min(tmin, l[i][j][0])

        tspan = (0.0, (tmax - tmin) * 1.0)

        return l, tspan

    @staticmethod
    def process_thp(train_set, test_set, batch_size=4):

        for i in range(len(train_set)):
            x = train_set[i][0][0]

            for j in range(len(train_set[i][0])):
                train_set[i][0][j] -= x

        for i in range(len(test_set)):
            x = test_set[i][0][0]

            for j in range(len(test_set[i][0])):
                test_set[i][0][j] -= x

        train = []
        test = []
        types = []

        for i in range(len(train_set)):
            tot = []
            t = []
            tot.append(train_set[i][0][0])

            for j in range(1, len(train_set[i][0]), 1):
                # print(tot[j-1])
                # print(train_set[i][0][j])
                tot.append(tot[j - 1] + train_set[i][0][j])

            for x in train_set[i][1]:
                types.append(x)

            t.append(tot)
            t.append(train_set[i][0])
            t.append(train_set[i][1])
            train.append(t)

        for i in range(len(test_set)):
            tot = []
            t = []
            tot.append(test_set[i][0][0])

            for j in range(1, len(test_set[i][0]), 1):
                # print(tot[j-1])
                # print(train_set[i][0][j])
                tot.append(tot[j - 1] + test_set[i][0][j])

            for x in test_set[i][1]:
                types.append(x)

            t.append(tot)
            t.append(test_set[i][0])
            t.append(test_set[i][1])
            test.append(t)

        ntypes = len(set(types))


        def features(batch):
            total, times, events = [], [], []
            for tot, time, event in batch:
                times.append(time)
                events.append(event)
                total.append(tot)

            return torch.FloatTensor(total), torch.FloatTensor(times), torch.LongTensor(events)

        trainl = DataLoader(train, batch_size=batch_size, shuffle=True, collate_fn=features)
        testl = DataLoader(test, batch_size=batch_size, shuffle=True, collate_fn=features)

        #trainl = DataLoader(train[:500], batch_size=batch_size, shuffle=True, collate_fn=features)
        #testl = DataLoader(test[500:1000], batch_size=batch_size, shuffle=True, collate_fn=features)

        return trainl, testl, ntypes

    @staticmethod
    def process_nhp(train_set, test_set):
        for i in range(len(train_set)):
            for j in range(len(train_set[i][0]) - 1, 0, -1):
                # print(j)
                train_set[i][0][j] -= train_set[i][0][j - 1]

        for i in range(len(test_set)):
            for j in range(len(test_set[i][0]) - 1, 0, -1):
                # print(j)
                test_set[i][0][j] -= test_set[i][0][j - 1]

        train = []
        test = []
        types = []

        for i in range(len(train_set)):
            tot = []
            t = []
            tot.append(train_set[i][0][0])

            for j in range(1, len(train_set[i][0]), 1):
                # print(tot[j-1])
                # print(train_set[i][0][j])
                tot.append(tot[j - 1] + train_set[i][0][j])

            for x in train_set[i][1]:
                types.append(x)

            t.append(tot)
            t.append(train_set[i][0])
            t.append(train_set[i][1])
            train.append(t)

        for i in range(len(test_set)):
            tot = []
            t = []
            tot.append(test_set[i][0][0])

            for j in range(1, len(test_set[i][0]), 1):
                # print(tot[j-1])
                # print(train_set[i][0][j])
                tot.append(tot[j - 1] + test_set[i][0][j])

            for x in test_set[i][1]:
                types.append(x)

            t.append(tot)
            t.append(train_set[i][0])
            t.append(train_set[i][1])
            test.append(t)

        ntypes = len(set(types))

        train_d = []
        test_d = []

        for i in range(len(train)):
            seq = []
            for j in range(len(train[i][0])):
                d = {}
                d['time_since_start'] = train[i][0][j]
                d['time_since_last_event'] = train[i][1][j]
                d['type_event'] = train[i][2][j]

                seq.append(d)

            train_d.append(seq)

        for i in range(len(test)):
            seq = []
            for j in range(len(test[i][0])):
                d = {}
                d['time_since_start'] = test[i][0][j]
                d['time_since_last_event'] = test[i][1][j]
                d['type_event'] = test[i][2][j]

                seq.append(d)

            test_d.append(seq)

        return train_d, test_d, ntypes


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

    @staticmethod
    def to_features(batch):
        times, events = [], []
        for time, event in batch:
            time = list(time)
            time = np.array([time[0]] + time)
            time = np.diff(time)
            times.append(time)
            events.append(event)

        return torch.FloatTensor(times), torch.LongTensor(events)

    @staticmethod
    def process_rmtpp(train_set):
        mn = len(train_set[0])

        for i in train_set:
            mn = min(mn, len(i))

        for i in range(len(train_set)):
            train_set[i] = train_set[i][:10]

        ts = np.array(train_set)
        temp = np.swapaxes(ts, 1, 2)

        return temp

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

    def process_nhp(self):
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

        return train_data, test_data, num_types

    @staticmethod
    def process_rmtpp(trainloader, testloader, mn_size = 4):
        tset = []

        for i in trainloader:
            if i[0].shape[1] >= 4:
                tset.append(list(i[1:]))
                mn_size = min(mn_size, i[0].shape[1])

        for i in range(len(tset)):
            tset[i][0] = tset[i][0][:, :mn_size]
            tset[i][1] = tset[i][1][:, :mn_size]

        vset = []

        for i in testloader:
            if i[0].shape[1] >= 4:
                vset.append(list(i[1:]))

        for i in range(len(vset)):
            vset[i][0] = vset[i][0][:, :mn_size]
            vset[i][1] = vset[i][1][:, :mn_size]

        return tset, vset


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

        #if (to_njsde):
        #    return self.process_njsde()
        self.data = data
        self.data_dev = data_dev
        self.total_event_num = total_event_num
            
        return data, data_dev, total_event_num

    def process_njsde(self, val=False):
        train = []
        if val:
            data = self.data_dev
        else:
            data = self.data

        for i in range(len(data)):
            seq = []
            for j in range(len(data[i])):
                seq.append((data[i][j]['time_since_start'], data[i][j]['type_event']))

            train.append(seq)

        tmax = train[0][0][0]
        tmin = train[0][0][0]

        for i in range(len(train)):
            for j in range(len(train[i])):
                tmax = max(tmax, train[i][j][0])
                tmin = min(tmin, train[i][j][0])

        tspan = (0.0, (tmax-tmin)*1.0)

        return train, tspan