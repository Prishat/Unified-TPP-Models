import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np

from .utils import rmtppNet
from .datasets import *
from .misc import *


class rmtpp:
    def __init__(self, params):
        # super(rmtpp, self).__init__(params, params['lossweight'])
        self.params = params
        # self.data = data

        weight = np.ones(self.params['event_class'])

        self.model = rmtppNet(self.params, lossweight=weight)

    def train(self, train_loader, val_loader):
        # train_loader, test_loader = self.preprocess(self.data)

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

            self.evaluate(val_loader, epc, test=False)

    def evaluate(self, test_loader, epoch, test=False):
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

        if not test:
            print(f"epoch {epoch}")

        print(f"time_error: {time_error}, PRECISION: {acc}, RECALL: {recall}, F1: {f1}")

    def predict(self, data):
        return self.model.predict(data)


# Neural Jump Stochastic Differential Equation

from .utils import RunningAverageMeter, ODEJumpFunc, forward_pass


class njsde:
    def __init__(self, params):
        self.params = params

    def train(self, ts, tspan):
        self.tspan = tspan
        TS = ts
        nseqs = len(TS)
        dim_c, dim_h, dim_N, dt = 10, 10, 22, 1.0 / 30.0

        TSTR = TS[:int(nseqs * 0.2 * self.params['fold'])] + TS[int(nseqs * 0.2 * (self.params['fold'] + 1)):]
        TSVA = TS[int(nseqs * 0.2 * self.params['fold']):int(nseqs * 0.2 * self.params['fold']) + self.params[
            'batch_size']]
        TSTE = TS[int(nseqs * 0.2 * self.params['fold']) + self.params['batch_size']:int(
            nseqs * 0.2 * (self.params['fold'] + 1))]

        # initialize / load model
        self.func = ODEJumpFunc(dim_c, dim_h, dim_N, dim_N, dim_hidden=32, num_hidden=2, ortho=True,
                                jump_type=self.params['jump_type'], evnt_align=self.params['evnt_align'],
                                activation=nn.CELU())
        self.c0 = torch.randn(dim_c, requires_grad=True)
        self.h0 = torch.zeros(dim_h)
        it0 = 0
        optimizer = optim.Adam([{'params': self.func.parameters()},
                                {'params': self.c0, 'lr': 1.0e-2},
                                ], lr=1e-3, weight_decay=1e-5)

        if self.params['restart']:
            checkpoint = torch.load(self.params['paramr'])
            self.func.load_state_dict(checkpoint['func_state_dict'])
            self.c0 = checkpoint['c0']
            self.h0 = checkpoint['h0']
            it0 = checkpoint['it0']
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        loss_meter = RunningAverageMeter()

        # if read from history, then fit to maximize likelihood
        it = it0
        if self.func.jump_type == "read":
            while it < self.params['niters']:
                # clear out gradients for variables
                optimizer.zero_grad()

                # sample a mini-batch, create a grid based on that
                batch_id = np.random.choice(len(TSTR), self.params['batch_size'], replace=False)
                # print(batch_id)
                batch = [TSTR[seqid] for seqid in batch_id]
                # print(batch)

                # forward pass
                tsave, trace, lmbda, gtid, tsne, loss, mete = forward_pass(self.func,
                                                                           torch.cat((self.c0, self.h0), dim=-1), tspan,
                                                                           dt,
                                                                           batch, self.params['evnt_align'])
                loss_meter.update(loss.item() / len(batch))

                # backward prop
                self.func.backtrace.clear()
                loss.backward()
                print("iter: {}, current loss: {:10.4f}, running ave loss: {:10.4f}, type error: {}".format(it,
                                                                                                            loss.item() / len(
                                                                                                                batch),
                                                                                                            loss_meter.avg,
                                                                                                            mete),
                      flush=True)

                # step
                optimizer.step()

                it = it + 1

                # validate and visualize
                if it % self.params['nsave'] == 0:
                    self.evaluate(TSVA, it, dt)  # Make dt a class variable
                    # save
                    # torch.save({'func_state_dict': func.state_dict(), 'c0': c0, 'h0': h0, 'it0': it,
                    #            'optimizer_state_dict': optimizer.state_dict()}, outpath + '/' + args.paramw)

        # computing testing error
        self.predict(TSTE, it, dt)

    def evaluate(self, TSVA, it=0, dt=1.0 / 30.0):
        for si in range(0, len(TSVA), self.params['batch_size']):
            # use the full validation set for forward pass
            tsave, trace, lmbda, gtid, tsne, loss, mete = forward_pass(self.func, torch.cat((self.c0, self.h0), dim=-1),
                                                                       self.tspan, dt,
                                                                       TSVA[si:si + self.params['batch_size']],
                                                                       self.params['evnt_align'])

            # backward prop
            self.func.backtrace.clear()
            loss.backward()
            print("iter: {:5d}, validation loss: {:10.4f}, num_evnts: {:8d}, type error: {}".format(it,
                                                                                                    loss.item() / len(
                                                                                                        TSVA[si:si +
                                                                                                                self.params[
                                                                                                                    'batch_size']]),
                                                                                                    len(tsne),
                                                                                                    mete),
                  flush=True)

            # visualize
            tsave_ = torch.tensor([record[0] for record in reversed(self.func.backtrace)])
            trace_ = torch.stack(tuple(record[1] for record in reversed(self.func.backtrace)))
            # visualize(outpath, tsave, trace, lmbda, tsave_, trace_, None, None, tsne, range(si, si + args.batch_size), it)

    def predict(self, TSTE, it, dt):
        for si in range(0, len(TSTE), self.params['batch_size']):
            tsave, trace, lmbda, gtid, tsne, loss, mete = forward_pass(self.func, torch.cat((self.c0, self.h0), dim=-1),
                                                                       self.tspan, dt,
                                                                       TSTE[si:si + self.params['batch_size']],
                                                                       self.params['evnt_align'])
            # visualize(outpath, tsave, trace, lmbda, None, None, None, None, tsne, range(si, si + args.batch_size), it,
            #          appendix="testing")
            print("iter: {:5d}, testing loss: {:10.4f}, num_evnts: {:8d}, type error: {}".format(it, loss.item() / len(
                TSTE[si:si + self.params['batch_size']]), len(tsne), mete), flush=True)
