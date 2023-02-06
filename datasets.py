import pandas
from tqdm import tqdm
import numpy as np
import torch
from collections import Counter
import math

class ATMDataset:
    def __init__(self, config, subset):
        data = pandas.read_csv(f"data/{subset}_day.csv")
        self.subset = subset
        self.id = list(data['id'])
        self.time = list(data['time'])
        self.event = list(data['event'])
        self.config = config
        self.seq_len = config.seq_len
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
