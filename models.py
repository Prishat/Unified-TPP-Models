import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

import numpy as np

from .utils import rmtppNet
from .datasets import *
from .misc import *


class rmtpp():
    def __init__(self, params, data):
        # super(rmtpp, self).__init__(params, params['lossweight'])
        self.params = params
        self.data = data

        weight = np.ones(self.params['event_class'])

        self.model = rmtppNet(self.params, lossweight=weight)

    def preprocess(self, data):
        train_set = ATMDataset(self.params, subset='train')
        test_set = ATMDataset(self.params, subset='test')
        train_loader = DataLoader(train_set, batch_size=self.params['batch_size'], shuffle=True,
                                  collate_fn=ATMDataset.to_features)
        test_loader = DataLoader(test_set, batch_size=self.params['batch_size'], shuffle=False,
                                 collate_fn=ATMDataset.to_features)

        return train_loader, test_loader

    def train(self):
        train_loader, test_loader = self.preprocess(self.data)

        self.model.set_optimizer(total_step=len(train_loader) * self.params['epochs'], use_bert=False)
        self.model.cuda()

        for epc in range(self.params['epochs']):
            self.model.train()
            range_loss1 = range_loss2 = range_loss = 0
            for i, batch in enumerate(tqdm(train_loader)):
                l1, l2, l = self.model.train_batch(batch)
                range_loss1 += l1
                range_loss2 += l2
                range_loss += l

                if (i + 1) % self.params['verbose_step'] == 0:
                    print("time loss: ", range_loss1 / self.params['verbose_step'])
                    print("event loss:", range_loss2 / self.params['verbose_step'])
                    print("total loss:", range_loss / self.params['verbose_step'])
                    range_loss1 = range_loss2 = range_loss = 0

    def evaluate(self, test_loader):
        self.model.eval()

        pred_times, pred_events = [], []
        gold_times, gold_events = [], []

        for i, batch in enumerate(tqdm(test_loader)):
            gold_times.append(batch[0][:, -1].numpy())
            gold_events.append(batch[1][:, -1].numpy())
            pred_time, pred_event = self.model.predict(batch)
            pred_times.append(pred_time)
            pred_events.append(pred_event)
        pred_times = np.concatenate(pred_times).reshape(-1)
        gold_times = np.concatenate(gold_times).reshape(-1)
        pred_events = np.concatenate(pred_events).reshape(-1)
        gold_events = np.concatenate(gold_events).reshape(-1)
        time_error = abs_error(pred_times, gold_times)
        acc, recall, f1 = clf_metric(pred_events, gold_events, n_class=self.params['event_class'])
        print(f"epoch {epc}")
        print(f"time_error: {time_error}, PRECISION: {acc}, RECALL: {recall}, F1: {f1}")

    def predict(self, data):
        return self.model.predict(data)
