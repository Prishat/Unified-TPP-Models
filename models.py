import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np

from .utils import *
from .datasets import *
from .misc import *

import time
from .transformer import Constants
from .transformer.Models import Transformer

from .nhps.models import nhp
from .nhps.io import processors
from .nhps.miss import miss_mec, factorized

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

    @staticmethod
    def preprocess(ts):
        """ts: list of numpy arrays in the shape [10, 2]"""
        l = []
        for x in ts:
            l.append(list(map(tuple, x.tolist())))

        return l

    def train(self, ts, tspan):
        self.tspan = tspan
        TS = ts
        nseqs = len(TS)
        dim_c, dim_h, dim_N, dt = 10, 10, self.params['num_types'], 1.0 / 30.0

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
                print("iter: {}, current loss: {:10.4f}, running ave loss: {:10.4f}, type error: {}".format(it, loss.item() / len(batch),
                                                                                                            loss_meter.avg, mete), flush=True)

                # step
                optimizer.step()

                it = it + 1

                # validate and visualize
                if it % self.params['nsave'] == 0:
                    self.evaluate(TSVA, dt)  # Make dt a class variable
                    # save
                    # torch.save({'func_state_dict': func.state_dict(), 'c0': c0, 'h0': h0, 'it0': it,
                    #            'optimizer_state_dict': optimizer.state_dict()}, outpath + '/' + args.paramw)

        # computing testing error
        self.predict(TSTE, dt)

    def evaluate(self, TSVA, dt=1.0 / 30.0):
        for si in range(0, len(TSVA), self.params['batch_size']):
            # use the full validation set for forward pass
            tsave, trace, lmbda, gtid, tsne, loss, mete = forward_pass(self.func, torch.cat((self.c0, self.h0), dim=-1),
                                                                       self.tspan, dt,
                                                                       TSVA[si:si + self.params['batch_size']],
                                                                       self.params['evnt_align'])

            # backward prop
            self.func.backtrace.clear()
            loss.backward()
            print("validation loss: {:10.4f}, num_evnts: {:8d}, type error: {}".format(loss.item() / len(TSVA[si:si + self.params['batch_size']]),
                                                                                                    len(tsne), mete), flush=True)

            # visualize
            tsave_ = torch.tensor([record[0] for record in reversed(self.func.backtrace)])
            trace_ = torch.stack(tuple(record[1] for record in reversed(self.func.backtrace)))
            # visualize(outpath, tsave, trace, lmbda, tsave_, trace_, None, None, tsne, range(si, si + args.batch_size), it)

    def predict(self, TSTE, dt):
        for si in range(0, len(TSTE), self.params['batch_size']):
            tsave, trace, lmbda, gtid, tsne, loss, mete = forward_pass(self.func, torch.cat((self.c0, self.h0), dim=-1),
                                                                       self.tspan, dt,
                                                                       TSTE[si:si + self.params['batch_size']],
                                                                       self.params['evnt_align'])
            # visualize(outpath, tsave, trace, lmbda, None, None, None, None, tsne, range(si, si + args.batch_size), it,
            #          appendix="testing")
            print("testing loss: {:10.4f}, num_evnts: {:8d}, type error: {}".format(loss.item() / len(TSTE[si:si + self.params['batch_size']]),
                                                                                    len(tsne), mete), flush=True)


class transHP:
    def __init__(self, params):
        self.params = params

        self.model = Transformer(
            num_types=self.params['num_types'],
            d_model=self.params['d_model'],
            d_rnn=self.params['d_rnn'],
            d_inner=self.params['d_inner_hid'],
            n_layers=self.params['n_layers'],
            n_head=self.params['n_head'],
            d_k=self.params['d_k'],
            d_v=self.params['d_v'],
            dropout=self.params['dropout'],
        )
        self.model.to(self.params['device'])

    def train_epoch(self, training_data, optimizer, pred_loss_func):
        self.model.train()

        total_event_ll = 0  # cumulative event log-likelihood
        total_time_se = 0  # cumulative time prediction squared-error
        total_event_rate = 0  # cumulative number of correct prediction
        total_num_event = 0  # number of total events
        total_num_pred = 0  # number of predictions

        for batch in tqdm(training_data, mininterval=2,
                          desc='  - (Training)   ', leave=False):
            """ prepare data """
            event_time, time_gap, event_type = map(lambda x: x.to(self.params['device']), batch)

            """ forward """
            optimizer.zero_grad()

            enc_out, prediction = self.model(event_type, event_time)

            """ backward """
            # negative log-likelihood
            event_ll, non_event_ll = log_likelihood(self.model, enc_out, event_time, event_type)
            event_loss = -torch.sum(event_ll - non_event_ll)

            # type prediction
            pred_loss, pred_num_event = type_loss(prediction[0], event_type, pred_loss_func)

            # time prediction
            se = time_loss(prediction[1], event_time)

            # SE is usually large, scale it to stabilize training
            scale_time_loss = 100
            loss = event_loss + pred_loss + se / scale_time_loss
            loss.backward()

            """ update parameters """
            optimizer.step()

            """ note keeping """
            total_event_ll += -event_loss.item()
            total_time_se += se.item()
            total_event_rate += pred_num_event.item()
            total_num_event += event_type.ne(Constants.PAD).sum().item()
            # we do not predict the first event
            total_num_pred += event_type.ne(Constants.PAD).sum().item() - event_time.shape[0]

        rmse = np.sqrt(total_time_se / total_num_pred)
        return total_event_ll / total_num_event, total_event_rate / total_num_pred, rmse

    def eval_epoch(self, validation_data, pred_loss_func):
        self.model.eval()

        total_event_ll = 0  # cumulative event log-likelihood
        total_time_se = 0  # cumulative time prediction squared-error
        total_event_rate = 0  # cumulative number of correct prediction
        total_num_event = 0  # number of total events
        total_num_pred = 0  # number of predictions
        with torch.no_grad():
            for batch in tqdm(validation_data, mininterval=2,
                              desc='  - (Validation) ', leave=False):
                """ prepare data """
                event_time, time_gap, event_type = map(lambda x: x.to(self.params['device']), batch)

                """ forward """
                enc_out, prediction = self.model(event_type, event_time)

                """ compute loss """
                event_ll, non_event_ll = log_likelihood(self.model, enc_out, event_time, event_type)
                event_loss = -torch.sum(event_ll - non_event_ll)
                _, pred_num = type_loss(prediction[0], event_type, pred_loss_func)
                se = time_loss(prediction[1], event_time)

                """ note keeping """
                total_event_ll += -event_loss.item()
                total_time_se += se.item()
                total_event_rate += pred_num.item()
                total_num_event += event_type.ne(Constants.PAD).sum().item()
                total_num_pred += event_type.ne(Constants.PAD).sum().item() - event_time.shape[0]

        rmse = np.sqrt(total_time_se / total_num_pred)
        return total_event_ll / total_num_event, total_event_rate / total_num_pred, rmse

    def _train(self, training_data, validation_data, optimizer, scheduler, pred_loss_func):
        valid_event_losses = []  # validation log-likelihood
        valid_pred_losses = []  # validation event type prediction accuracy
        valid_rmse = []  # validation event time prediction RMSE
        for epoch_i in range(self.params['epoch']):
            epoch = epoch_i + 1
            print('[ Epoch', epoch, ']')

            start = time.time()
            train_event, train_type, train_time = self.train_epoch(training_data, optimizer, pred_loss_func)
            print('  - (Training)    loglikelihood: {ll: 8.5f}, '
                  'accuracy: {type: 8.5f}, RMSE: {rmse: 8.5f}, '
                  'elapse: {elapse:3.3f} min'
                  .format(ll=train_event, type=train_type, rmse=train_time, elapse=(time.time() - start) / 60))

            start = time.time()
            valid_event, valid_type, valid_time = self.eval_epoch(validation_data, pred_loss_func)
            print('  - (Testing)     loglikelihood: {ll: 8.5f}, '
                  'accuracy: {type: 8.5f}, RMSE: {rmse: 8.5f}, '
                  'elapse: {elapse:3.3f} min'
                  .format(ll=valid_event, type=valid_type, rmse=valid_time, elapse=(time.time() - start) / 60))

            valid_event_losses += [valid_event]
            valid_pred_losses += [valid_type]
            valid_rmse += [valid_time]
            print('  - [Info] Maximum ll: {event: 8.5f}, '
                  'Maximum accuracy: {pred: 8.5f}, Minimum RMSE: {rmse: 8.5f}'
                  .format(event=max(valid_event_losses), pred=max(valid_pred_losses), rmse=min(valid_rmse)))

            # logging
            with open(self.params['log'], 'a') as f:
                f.write('{epoch}, {ll: 8.5f}, {acc: 8.5f}, {rmse: 8.5f}\n'
                        .format(epoch=epoch, ll=valid_event, acc=valid_type, rmse=valid_time))

            scheduler.step()

    def train(self, trainloader, testloader, num_types):
        # setup the log file
        with open(self.params['log'], 'w') as f:
            f.write('Epoch, Log-likelihood, Accuracy, RMSE\n')

        """ optimizer and scheduler """
        optimizer = optim.Adam(filter(lambda x: x.requires_grad, self.model.parameters()),
                               self.params['lr'], betas=(0.9, 0.999), eps=1e-05)
        scheduler = optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.5)

        """ prediction loss function, either cross entropy or label smoothing """
        if self.params['smooth'] > 0:
            pred_loss_func = LabelSmoothingLoss(self.params['smooth'], num_types, ignore_index=-1)
        else:
            pred_loss_func = nn.CrossEntropyLoss(ignore_index=-1, reduction='none')

        """ number of parameters """
        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print('[Info] Number of parameters: {}'.format(num_params))

        """ train the model """
        self._train(trainloader, testloader, optimizer, scheduler, pred_loss_func)

    def evaluate(self, valloader, num_types):
        start = time.time()
        if self.params['smooth'] > 0:
            pred_loss_func = LabelSmoothingLoss(self.params['smooth'], num_types, ignore_index=-1)
        else:
            pred_loss_func = nn.CrossEntropyLoss(ignore_index=-1, reduction='none')

        valid_event, valid_type, valid_time = self.eval_epoch(valloader, pred_loss_func)
        print('  - (Testing)     loglikelihood: {ll: 8.5f}, '
              'accuracy: {type: 8.5f}, RMSE: {rmse: 8.5f}, '
              'elapse: {elapse:3.3f} min'
              .format(ll=valid_event, type=valid_type, rmse=valid_time, elapse=(time.time() - start) / 60))

    def predict(self):
        pass

class neuralHP:
    def __init__(self, params):
        self.params = params

    def train(self, train_data, val_data):
        random.seed(self.params['Seed'])
        np.random.seed(self.params['Seed'])
        torch.manual_seed(self.params['Seed'])

        hidden_dim = self.params['DimLSTM']

        agent = nhp.NeuralHawkes(
            total_num=self.params['total_event_num'], hidden_dim=hidden_dim,
            device='cuda' if self.params['UseGPU'] else 'cpu'
        )

        if self.params['UseGPU']:
            agent.cuda()

        sampling = self.params['Multiplier']

        miss_mec = factorized.FactorizedMissMec(
            device='cuda' if self.params['UseGPU'] else 'cpu',
            config_file=os.path.join(self.params['PathData'], 'censor.conf')
        )

        proc = processors.DataProcessorNeuralHawkes(
            idx_BOS=self.params['total_event_num'],
            idx_EOS=self.params['total_event_num'] + 1,
            idx_PAD=self.params['total_event_num'] + 2,
            miss_mec=miss_mec,
            sampling=sampling,
            device='cuda' if self.params['UseGPU'] else 'cpu'
        )
        #logger = processors.LogWriter(self.params['PathLog'], self.params)

        r"""
        we only update parameters that are only related to left2right machine
        """
        optimizer = optim.Adam(
            agent.parameters(), lr=self.params['learning_rate']
        )

        print("Start training ... ")
        total_logP_best = -1e6
        avg_dis_best = 1e6
        episode_best = -1
        time0 = time.time()

        episodes = []
        total_rewards = []

        max_episode = self.params['MaxEpoch'] * len(train_data)
        report_gap = self.params['TrackPeriod']

        time_sample = 0.0
        time_train_only = 0.0
        time_dev_only = 0.0
        input = []

        for episode in range(max_episode):

            idx_seq = episode % len(train_data)
            idx_epoch = episode // len(train_data)
            one_seq = train_data[idx_seq]

            # time_sample_0 = time.time()
            input.append(proc.processSeq(one_seq, n=1))
            #print(len(input))
            #print(input)
            # time_sample += (time.time() - time_sample_0)

            if len(input) >= self.params['SizeBatch']:

                batchdata_seqs = proc.processBatchSeqsWithParticles(input)
                #print('len(batchdata_seqs) := ',len(batchdata_seqs))
                #print('batchdata_seqs[0] := ', batchdata_seqs[0])
                #print('batchdata_seqs[5] := ', batchdata_seqs[5][:-2])
                #batchdata_seqs[5] = batchdata_seqs[5][:-2]

                agent.train()
                time_train_only_0 = time.time()

                objective, _ = agent(batchdata_seqs, mode=1)
                objective.backward()

                optimizer.step()
                optimizer.zero_grad()
                time_train_only += (time.time() - time_train_only_0)

                input = []

                if episode % report_gap == report_gap - 1:

                    time1 = time.time()
                    time_train = time1 - time0
                    time0 = time1

                    print("Validating at episode {} ({}-th seq of {}-th epoch)".format(
                        episode, idx_seq, idx_epoch))
                    total_logP = 0.0
                    total_num_token = 0.0

                    input_dev = []
                    agent.eval()

                    for i_dev, one_seq_dev in enumerate(val_data):

                        input_dev.append(
                            proc.processSeq(one_seq_dev, n=1))

                        if (i_dev + 1) % self.params['SizeBatch'] == 0 or \
                                (i_dev == len(val_data) - 1 and (len(input_dev) % self.params['SizeBatch']) > 0):
                            batchdata_seqs_dev = proc.processBatchSeqsWithParticles(
                                input_dev)

                            time_dev_only_0 = time.time()
                            objective_dev, num_events_dev = agent(
                                batchdata_seqs_dev, mode=1)
                            time_dev_only = time.time() - time_dev_only_0

                            total_logP -= float(objective_dev.data.sum())

                            total_num_token += float(
                                num_events_dev.data.sum() / (1 * 1.0))

                            input_dev = []

                    total_logP /= total_num_token

                    message = "Episode {} ({}-th seq of {}-th epoch), loglik is {:.4f}".format(
                        episode, idx_seq, idx_epoch, total_logP)
                    #logger.checkpoint(message)
                    print(message)

                    updated = None
                    if total_logP > total_logP_best:
                        total_logP_best = total_logP
                        updated = True
                        episode_best = episode
                    else:
                        updated = False
                    message = "Current best loglik is {:.4f} (updated at episode {})".format(
                        total_logP_best, episode_best)

                    if updated:
                        message += ", best updated at this episode"
                        torch.save(
                            agent.state_dict(), self.params['PathSave'])
                    #logger.checkpoint(message)

                    print(message)
                    episodes.append(episode)

                    time1 = time.time()
                    time_dev = time1 - time0
                    time0 = time1
                    message = "time to train {} episodes is {:.2f} and time for dev is {:.2f}".format(
                        report_gap, time_train, time_dev)

                    time_sample, time_train_only = 0.0, 0.0
                    time_dev_only = 0.0
                    #
                    #logger.checkpoint(message)
                    print(message)
        message = "training finished"
        #logger.checkpoint(message)
        print(message)

    def evaluate(self):
        pass

    def predict(self):
        pass