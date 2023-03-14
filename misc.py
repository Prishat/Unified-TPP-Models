import numpy as np
from collections import Counter

def abs_error(pred, gold):
    return np.mean(np.abs(pred - gold))

def clf_metric(pred, gold, n_class):
    gold_count = Counter(gold)
    pred_count = Counter(pred)
    prec = recall = 0
    pcnt = rcnt = 0
    for i in range(n_class):
        match_count = np.logical_and(pred == gold, pred == i).sum()
        if gold_count[i] != 0:
            prec += match_count / gold_count[i]
            pcnt += 1
        if pred_count[i] != 0:
            recall += match_count / pred_count[i]
            rcnt += 1
    prec /= pcnt
    recall /= rcnt
    print(f"pcnt={pcnt}, rcnt={rcnt}")
    f1 = 2 * prec * recall / (prec + recall)
    return prec, recall, f1

import numpy as np
import torch
import torch.utils.data

from .transformer import Constants


class EventData(torch.utils.data.Dataset):
    """ Event stream dataset. """

    def __init__(self, data):
        """
        Data should be a list of event streams; each event stream is a list of dictionaries;
        each dictionary contains: time_since_start, time_since_last_event, type_event
        """
        self.time = [[elem['time_since_start'] for elem in inst] for inst in data]
        self.time_gap = [[elem['time_since_last_event'] for elem in inst] for inst in data]
        # plus 1 since there could be event type 0, but we use 0 as padding
        self.event_type = [[elem['type_event'] + 1 for elem in inst] for inst in data]

        self.length = len(data)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """ Each returned element is a list, which represents an event stream """
        return self.time[idx], self.time_gap[idx], self.event_type[idx]


def pad_time(insts):
    """ Pad the instance to the max seq length in batch. """

    max_len = max(len(inst) for inst in insts)

    batch_seq = np.array([
        inst + [Constants.PAD] * (max_len - len(inst))
        for inst in insts])

    return torch.tensor(batch_seq, dtype=torch.float32)


def pad_type(insts):
    """ Pad the instance to the max seq length in batch. """

    max_len = max(len(inst) for inst in insts)

    batch_seq = np.array([
        inst + [Constants.PAD] * (max_len - len(inst))
        for inst in insts])

    return torch.tensor(batch_seq, dtype=torch.long)


def collate_fn(insts):
    """ Collate function, as required by PyTorch. """

    time, time_gap, event_type = list(zip(*insts))
    time = pad_time(time)
    time_gap = pad_time(time_gap)
    event_type = pad_type(event_type)
    return time, time_gap, event_type


def get_dataloader(data, batch_size, shuffle=True):
    """ Prepare dataloader. """

    ds = EventData(data)
    dl = torch.utils.data.DataLoader(
        ds,
        num_workers=2,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=shuffle
    )
    return dl