import torch
from torch import nn
from torch.optim import Adam
import torch.nn.functional as F
import numpy as np

import sys
import bisect
import networkx as nx
from numbers import Number
import math
from torchdiffeq import odeint_adjoint as odeint

class rmtppNet(nn.Module):
    def __init__(self, params, lossweight):
        super(rmtppNet, self).__init__()
        self.params = params
        self.n_class = params['event_class']
        self.embedding = nn.Embedding(num_embeddings=params['event_class'], embedding_dim=params['emb_dim'])
        self.emb_drop = nn.Dropout(p=params['dropout'])
        self.lstm = nn.LSTM(input_size=params['emb_dim'] + 1,
                            hidden_size=params['hid_dim'],
                            batch_first=True,
                            bidirectional=False)
        self.mlp = nn.Linear(in_features=params['hid_dim'], out_features=params['mlp_dim'])
        self.mlp_drop = nn.Dropout(p=params['dropout'])
        self.event_linear = nn.Linear(in_features=params['mlp_dim'], out_features=params['event_class'])
        self.time_linear = nn.Linear(in_features=params['mlp_dim'], out_features=1)
        self.set_criterion(lossweight)

    def set_optimizer(self, total_step, use_bert=True):
        """if use_bert:
            self.optimizer = BertAdam(params=self.parameters(),
                                      lr=self.config.lr,
                                      warmup=0.1,
                                      t_total=total_step)
        else:
            self.optimizer = Adam(self.parameters(), lr=self.config.lr)"""
        self.optimizer = Adam(self.parameters(), lr=self.params['lr'])

    def set_criterion(self, weight):
        self.event_criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(weight))
        if self.params['model'] == 'rmtpp':
            self.intensity_w = nn.Parameter(torch.tensor(0.1, dtype=torch.float, device='cuda'))
            self.intensity_b = nn.Parameter(torch.tensor(0.1, dtype=torch.float, device='cuda'))
            self.time_criterion = self.RMTPPLoss
        else:
            self.time_criterion = nn.MSELoss()

    def RMTPPLoss(self, pred, gold):
        loss = torch.mean(pred + self.intensity_w * gold + self.intensity_b +
                          (torch.exp(pred + self.intensity_b) -
                           torch.exp(pred + self.intensity_w * gold + self.intensity_b)) / self.intensity_w)
        return -1 * loss

    def forward(self, input_time, input_events):
        event_embedding = self.embedding(input_events)
        event_embedding = self.emb_drop(event_embedding)
        lstm_input = torch.cat((event_embedding, input_time.unsqueeze(-1)), dim=-1)
        hidden_state, _ = self.lstm(lstm_input)

        # hidden_state = torch.cat((hidden_state, input_time.unsqueeze(-1)), dim=-1)
        mlp_output = torch.tanh(self.mlp(hidden_state[:, -1, :]))
        mlp_output = self.mlp_drop(mlp_output)
        event_logits = self.event_linear(mlp_output)
        time_logits = self.time_linear(mlp_output)
        return time_logits, event_logits

    def dispatch(self, tensors):
        for i in range(len(tensors)):
            tensors[i] = tensors[i].cuda().contiguous()
        return tensors

    def train_batch(self, batch):
        time_tensor, event_tensor = batch
        time_input, time_target = self.dispatch([time_tensor[:, :-1], time_tensor[:, -1]])
        event_input, event_target = self.dispatch([event_tensor[:, :-1], event_tensor[:, -1]])

        time_logits, event_logits = self.forward(time_input, event_input)
        loss1 = self.time_criterion(time_logits.view(-1), time_target.view(-1))
        loss2 = self.event_criterion(event_logits.view(-1, self.n_class), event_target.view(-1))
        loss = self.params['alpha'] * loss1 + loss2
        loss.backward()

        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss1.item(), loss2.item(), loss.item()

    def predict(self, batch):
        time_tensor, event_tensor = batch
        time_input, time_target = self.dispatch([time_tensor[:, :-1], time_tensor[:, -1]])
        event_input, event_target = self.dispatch([event_tensor[:, :-1], event_tensor[:, -1]])
        time_logits, event_logits = self.forward(time_input, event_input)
        event_pred = np.argmax(event_logits.detach().cpu().numpy(), axis=-1)
        time_pred = time_logits.detach().cpu().numpy()
        return time_pred, event_pred


# Neural Jump Stochastic Differential Equation

# compute the running average
class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.vals = []
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.vals = []
        self.val = None
        self.avg = 0

    def update(self, val):
        self.vals.append(val)
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val


# SoftPlus activation function add epsilon
class SoftPlus(nn.Module):

    def __init__(self, beta=1.0, threshold=20, epsilon=1.0e-15, dim=None):
        super(SoftPlus, self).__init__()
        self.Softplus = nn.Softplus(beta, threshold)
        self.epsilon = epsilon
        self.dim = dim

    def forward(self, x):
        # apply softplus to first dim dimension
        if self.dim is None:
            result = self.Softplus(x) + self.epsilon
        else:
            result = torch.cat((self.Softplus(x[..., :self.dim])+self.epsilon, x[..., self.dim:]), dim=-1)

        return result


# multi-layer perceptron
class MLP(nn.Module):

    def __init__(self, dim_in, dim_out, dim_hidden=20, num_hidden=0, activation=nn.CELU()):
        super(MLP, self).__init__()

        if num_hidden == 0:
            self.linears = nn.ModuleList([nn.Linear(dim_in, dim_out)])
        elif num_hidden >= 1:
            self.linears = nn.ModuleList()
            self.linears.append(nn.Linear(dim_in, dim_hidden))
            self.linears.extend([nn.Linear(dim_hidden, dim_hidden) for _ in range(num_hidden-1)])
            self.linears.append(nn.Linear(dim_hidden, dim_out))
        else:
            raise Exception('number of hidden layers must be positive')

        for m in self.linears:
            nn.init.normal_(m.weight, mean=0, std=0.1)
            nn.init.uniform_(m.bias, a=-0.1, b=0.1)

        self.activation = activation

    def forward(self, x):
        for m in self.linears[:-1]:
            x = self.activation(m(x))

        return self.linears[-1](x)


# recurrent neural network
class RNN(nn.Module):

    def __init__(self, dim_in, dim_out, dim_hidden, num_hidden, activation):
        super(RNN, self).__init__()

        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_hidden = dim_hidden
        self.i2h = MLP(dim_in+dim_hidden, dim_hidden, dim_hidden, num_hidden, activation)
        self.h2o = MLP(dim_hidden, dim_out, dim_hidden, num_hidden, activation)
        self.activation = activation

    def forward(self, x, h0=None):
        assert len(x.shape) > 2,  'z need to be at least a 2 dimensional vector accessed by [tid ... dim_id]'

        if h0 is None:
            hh = [torch.zeros(x.shape[1:-1] + (self.dim_hidden,))]
        else:
            hh = [h0]

        for i in range(x.shape[0]):
            combined = torch.cat((x[i], hh[-1]), dim=-1)
            hh.append(self.activation(self.i2h(combined)))

        return self.h2o(torch.stack(tuple(hh)))


# graph convolution unit
class GCU(nn.Module):

    def __init__(self, dim_c, dim_h=0, dim_hidden=20, num_hidden=0, activation=nn.CELU(), graph=None, aggregation=None):
        super(GCU, self).__init__()

        self.cur = nn.Sequential(MLP((dim_c+dim_h),   dim_hidden, dim_hidden, num_hidden, activation), activation)
        self.nbr = nn.Sequential(MLP((dim_c+dim_h)*2, dim_hidden, dim_hidden, num_hidden, activation), activation)
        self.out = nn.Linear(dim_hidden*2, dim_c)

        nn.init.normal_(self.out.weight, mean=0, std=0.1)
        nn.init.uniform_(self.out.bias, a=-0.1, b=0.1)

        if graph is not None:
            self.graph = graph
        else:
            self.graph = nx.Graph()
            self.graph.add_node(0)

        if aggregation is None:
            self.aggregation = lambda vnbr: vnbr.sum(dim=-2)
        else:
            self.aggregation = aggregation

    def forward(self, z):
        assert len(z.shape) >= 2, 'z_ need to be >=2 dimensional vector accessed by [..., node_id, dim_id]'

        curvv = self.cur(z)

        def conv(nid):
            env = list(self.graph.neighbors(nid))
            if len(env) == 0:
                nbrv = torch.zeros(curvv[nid].shape)
            else:
                nbrv = self.aggregation(self.nbr(torch.cat((z[..., [nid]*len(env), :], z[..., env, :]), dim=-1)))
            return nbrv

        nbrvv = torch.stack([conv(nid) for nid in self.graph.nodes()], dim=-2)

        dcdt = self.out(torch.cat((curvv, nbrvv), dim=-1))

        return dcdt


# This function need to be stateless
class ODEFunc(nn.Module):

    def __init__(self, dim_c, dim_hidden=20, num_hidden=0, activation=nn.CELU(), ortho=False, graph=None, aggregation=None):
        super(ODEFunc, self).__init__()

        self.dim_c = dim_c
        self.ortho = ortho

        if graph is not None:
            self.F = GCU(dim_c, 0, dim_hidden, num_hidden, activation, aggregation, graph)
        else:
            self.F = MLP(dim_c, dim_c, dim_hidden, num_hidden, activation)

    def forward(self, t, c):
        dcdt = self.F(c)

        # orthogonalize dc w.r.t. to c
        if self.ortho:
            dcdt = dcdt - (dcdt*c).sum(dim=-1, keepdim=True) / (c*c).sum(dim=-1, keepdim=True) * c

        return dcdt

class ODEJumpFunc(nn.Module):

    def __init__(self, dim_c, dim_h, dim_N, dim_E, dim_hidden=20, num_hidden=0, activation=nn.CELU(), ortho=False,
                 jump_type="read", evnts=[], evnt_align=False, evnt_embedding="discrete",
                 graph=None, aggregation=None):
        super(ODEJumpFunc, self).__init__()

        self.dim_c = dim_c
        self.dim_h = dim_h
        self.dim_N = dim_N  # number of event type
        self.dim_E = dim_E  # dimension for encoding of event itself
        self.ortho = ortho
        self.evnt_embedding = evnt_embedding

        assert jump_type in ["simulate", "read"], "invalide jump_type, must be one of [simulate, read]"
        self.jump_type = jump_type
        assert (jump_type == "simulate" and len(evnts) == 0) or jump_type == "read"
        self.evnts = evnts
        self.evnt_align = evnt_align

        if graph is not None:
            self.F = GCU(dim_c, dim_h, dim_hidden, num_hidden, activation, aggregation, graph)
        else:
            self.F = MLP(dim_c+dim_h, dim_c, dim_hidden, num_hidden, activation)

        self.G = nn.Sequential(MLP(dim_c, dim_h, dim_hidden, num_hidden, activation), nn.Softplus())

        if evnt_embedding == "discrete":
            assert dim_E == dim_N, "if event embedding is discrete, then use one dimension for each event type"
            self.evnt_embed = lambda k: (torch.arange(0, dim_E) == k).float()
            # output is a dim_N vector, each represent conditional intensity of a type of event
            self.L = nn.Sequential(MLP(dim_c+dim_h, dim_N, dim_hidden, num_hidden, activation), SoftPlus())
        elif evnt_embedding == "continuous":
            self.evnt_embed = lambda k: torch.tensor(k)
            # output is a dim_N*(1+2*dim_E) vector, represent coefficients, mean and log variance of dim_N unit gaussian intensity function
            self.L = nn.Sequential(MLP(dim_c+dim_h, dim_N*(1+2*dim_E), dim_hidden, num_hidden, activation), SoftPlus(dim=dim_N))
        else:
            raise Exception('evnt_type must either be discrete or continuous')

        self.W = MLP(dim_c+dim_E, dim_h, dim_hidden, num_hidden, activation)

        self.backtrace = []

    def forward(self, t, z):
        c = z[..., :self.dim_c]
        h = z[..., self.dim_c:]

        dcdt = self.F(z)

        # orthogonalize dc w.r.t. to c
        if self.ortho:
            dcdt = dcdt - (dcdt*c).sum(dim=-1, keepdim=True) / (c*c).sum(dim=-1, keepdim=True) * c

        dhdt = -self.G(c) * h

        return torch.cat((dcdt, dhdt), dim=-1)

    def next_simulated_jump(self, t0, z0, t1):

        if not self.evnt_align:
            m = torch.distributions.Exponential(self.L(z0)[..., :self.dim_N].double())
            # next arrival time
            tt = t0 + m.sample()
            tt_min = tt.min()

            if tt_min <= t1:
                dN = (tt == tt_min).float()
            else:
                dN = torch.zeros(tt.shape)

            next_t = min(tt_min, t1)
        else:
            assert t0 < t1

            lmbda_dt = self.L(z0) * (t1 - t0)
            rd = torch.rand(lmbda_dt.shape)
            dN = torch.zeros(lmbda_dt.shape)
            dN[rd < lmbda_dt ** 2 / 2] += 1
            dN[rd < lmbda_dt ** 2 / 2 + lmbda_dt * torch.exp(-lmbda_dt)] += 1

            next_t = t1

        return dN, next_t

    def simulated_jump(self, dN, t, z):
        assert self.jump_type == "simulate", "simulate_jump must be called with jump_type = simulate"
        dz = torch.zeros(z.shape)
        sequence = []

        c = z[..., :self.dim_c]
        for idx in dN.nonzero():
            # find location and type of event
            loc, k = tuple(idx[:-1]), idx[-1]
            ne = int(dN[tuple(idx)])

            for _ in range(ne):
                if self.evnt_embedding == "discrete":
                    # encode of event k
                    kv = self.evnt_embed(k)
                    sequence.extend([(t,) + loc + (k,)])
                elif self.evnt_embedding == "continuous":
                    params = self.L(z[loc])
                    gsmean = params[self.dim_N*(1+self.dim_E*0):self.dim_N*(1+self.dim_E*1)]
                    logvar = params[self.dim_N*(1+self.dim_E*1):self.dim_N*(1+self.dim_E*2)]
                    gsmean_k = gsmean[self.dim_E*k:self.dim_E*(k+1)]
                    logvar_k = logvar[self.dim_E*k:self.dim_E*(k+1)]
                    kv = self.evnt_embed(torch.randn(gsmean_k.shape) * torch.exp(0.5*logvar_k) + gsmean)
                    sequence.extend([(t,) + loc + (kv,)])

                # add to jump
                dz[loc][self.dim_c:] += self.W(torch.cat((c[loc], kv), dim=-1))

        self.evnts.extend(sequence)

        return dz

    def next_read_jump(self, t0, t1):
        assert self.jump_type == "read", "next_read_jump must be called with jump_type = read"
        assert t0 != t1, "t0 can not equal t1"

        t = t1
        inf = sys.maxsize
        if t0 < t1:  # forward
            idx = bisect.bisect_right(self.evnts, (t0, inf, inf, inf))
            if idx != len(self.evnts):
                t = min(t1, torch.tensor(self.evnts[idx][0], dtype=torch.float64))
        else:  # backward
            idx = bisect.bisect_left(self.evnts, (t0, -inf, -inf, -inf))
            if idx > 0:
                t = max(t1, torch.tensor(self.evnts[idx-1][0], dtype=torch.float64))

        assert t != t0, "t can not equal t0"
        return t

    def read_jump(self, t, z):
        assert self.jump_type == "read", "read_jump must be called with jump_type = read"
        dz = torch.zeros(z.shape)

        inf = sys.maxsize
        lid = bisect.bisect_left(self.evnts, (t, -inf, -inf, -inf))
        rid = bisect.bisect_right(self.evnts, (t, inf, inf, inf))

        c = z[..., :self.dim_c]
        for evnt in self.evnts[lid:rid]:
            # find location and type of event
            loc, k = evnt[1:-1], evnt[-1]

            # encode of event k
            kv = self.evnt_embed(k)

            # add to jump
            dz[loc][self.dim_c:] += self.W(torch.cat((c[loc], kv), dim=-1))

        return dz


def logsumexp(value, dim=None, keepdim=False):
    """Numerically stable implementation of the operation
    value.exp().sum(dim, keepdim).log()
    """
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(torch.exp(value0), dim=dim, keepdim=keepdim))
    else:
        m = torch.max(value)
        sum_exp = torch.sum(torch.exp(value - m))
        if isinstance(sum_exp, Number):
            return m + math.log(sum_exp)
        else:
            return m + torch.log(sum_exp)


# this function takes in a time series and create a grid for modeling it
# it takes an array of sequences of three tuples, and extend it to four tuple
def create_tsave(tmin, tmax, dt, evnts_raw, evnt_align=False):
    """
    :param tmin: min time of sequence
    :param tmax: max time of the sequence
    :param dt: step size
    :param evnts_raw: tuple (raw_time, ...)
    :param evnt_align: whether to round the event time up to the next grid point
    :return tsave: the time to save state in ODE simulation
    :return gtid: grid time id
    :return evnts: tuple (rounded_time, ...)
    :return tse: tuple (event_time_id, ...)
    """

    if evnt_align:
        tc = lambda t: np.round(np.ceil((t-tmin) / dt) * dt + tmin, decimals=8)
    else:
        tc = lambda t: t

    evnts = [(tc(evnt[0]),) + evnt[1:] for evnt in evnts_raw if tmin < tc(evnt[0]) < tmax]

    tgrid = np.round(np.arange(tmin, tmax+dt, dt), decimals=8)
    tevnt = np.array([evnt[0] for evnt in evnts])
    tsave = np.sort(np.unique(np.concatenate((tgrid, tevnt))))
    t2tid = {t: tid for tid, t in enumerate(tsave)}

    # g(rid)tid
    # t(ime)s(equence)n(ode)e(vent)
    gtid = [t2tid[t] for t in tgrid]
    tse = [(t2tid[evnt[0]],) + evnt[1:] for evnt in evnts]

    return torch.tensor(tsave), gtid, evnts, tse


def forward_pass(func, z0, tspan, dt, batch, evnt_align, gs_info=None, type_forecast=[0.0], predict_first=True, rtol=1.0e-5, atol=1.0e-7, scale=1.0):
    # merge the sequences to create a sequence
    evnts_raw = sorted([(evnt[0],) + (sid,) + evnt[1:] for sid in range(len(batch)) for evnt in batch[sid]])

    # set up grid
    tsave, gtid, evnts, tse = create_tsave(tspan[0], tspan[1], dt, evnts_raw, evnt_align)
    func.evnts = evnts

    # convert to numpy array
    tsavenp = tsave.numpy()

    # forward pass
    trace = odeint(func, z0.repeat(len(batch), 1), tsave, method='jump_adams', rtol=rtol, atol=atol)
    params = func.L(trace)
    lmbda = params[..., :func.dim_N]

    if gs_info is not None:
        lmbda[:, :, :] = torch.tensor(gs_info[0])

    def integrate(tt, ll):
        lm = (ll[:-1, ...] + ll[1:, ...]) / 2.0
        dts = (tt[1:] - tt[:-1]).reshape((-1,)+(1,)*(len(lm.shape)-1)).float()
        return (lm * dts).sum()

    log_likelihood = -integrate(tsave, lmbda)

    # set of sequences where at least one event has happened
    seqs_happened = set(sid for sid in range(len(batch))) if predict_first else set()

    if func.evnt_embedding == "discrete":
        et_error = []
        for evnt in tse:
            log_likelihood += torch.log(lmbda[evnt])
            if evnt[1] in seqs_happened:
                type_preds = torch.zeros(len(type_forecast))
                for tid, t in enumerate(type_forecast):
                    loc = (np.searchsorted(tsavenp, tsave[evnt[0]].item()-t),) + evnt[1:-1]
                    type_preds[tid] = lmbda[loc].argmax().item()
                et_error.append((type_preds != evnt[-1]).float())
            seqs_happened.add(evnt[1])

        METE = sum(et_error)/len(et_error) if len(et_error) > 0 else -torch.ones(len(type_forecast))

    elif func.evnt_embedding == "continuous":
        gsmean = params[..., func.dim_N*(1+func.dim_E*0):func.dim_N*(1+func.dim_E*1)].view(params.shape[:-1]+(func.dim_N, func.dim_E))
        logvar = params[..., func.dim_N*(1+func.dim_E*1):func.dim_N*(1+func.dim_E*2)].view(params.shape[:-1]+(func.dim_N, func.dim_E))
        var = torch.exp(logvar)

        if gs_info is not None:
            gsmean[:, :, :] = torch.tensor(gs_info[1])
            var[:, :, :] = torch.tensor(gs_info[2])

        def log_normal_pdf(loc, k):
            const = torch.log(torch.tensor(2.0*np.pi))
            return -0.5*(const + logvar[loc] + (gsmean[loc] - func.evnt_embed(k))**2.0 / var[loc])

        et_error = []
        for evnt in tse:
            log_gs = log_normal_pdf(evnt[:-func.dim_N], evnt[-func.dim_N:]).sum(dim=-1)
            log_likelihood += logsumexp(lmbda[evnt[:-func.dim_N]].log() + log_gs, dim=-1)
            if evnt[1] in seqs_happened:
                # mean_pred embedding
                mean_preds = torch.zeros(len(type_forecast), func.dim_E)
                for tid, t in enumerate(type_forecast):
                    loc = (np.searchsorted(tsavenp, tsave[evnt[0]].item()-t),) + evnt[1:-func.dim_N]
                    mean_preds[tid] = ((lmbda[loc].view(func.dim_N, 1) * gsmean[loc]).sum(dim=0) / lmbda[loc].sum()).detach()
                et_error.append((mean_preds - func.evnt_embed(evnt[-func.dim_N:])).norm(dim=-1))
            seqs_happened.add(evnt[1])

        METE = sum(et_error)*scale/len(et_error) if len(et_error) > 0 else -torch.ones(len(type_forecast))

    if func.evnt_embedding == "discrete":
        return tsave, trace, lmbda, gtid, tse, -log_likelihood, METE
    elif func.evnt_embedding == "continuous":
        return tsave, trace, lmbda, gtid, tse, -log_likelihood, METE, gsmean, var


# Transformer Hawkes Process

from .transformer.Models import get_non_pad_mask

def softplus(x, beta):
    # hard thresholding at 20
    temp = beta * x
    temp[temp > 20] = 20
    return 1.0 / beta * torch.log(1 + torch.exp(temp))


def compute_event(event, non_pad_mask):
    """ Log-likelihood of events. """

    # add 1e-9 in case some events have 0 likelihood
    event += math.pow(10, -9)
    event.masked_fill_(~non_pad_mask.bool(), 1.0)

    result = torch.log(event)
    return result


def compute_integral_biased(all_lambda, time, non_pad_mask):
    """ Log-likelihood of non-events, using linear interpolation. """

    diff_time = (time[:, 1:] - time[:, :-1]) * non_pad_mask[:, 1:]
    diff_lambda = (all_lambda[:, 1:] + all_lambda[:, :-1]) * non_pad_mask[:, 1:]

    biased_integral = diff_lambda * diff_time
    result = 0.5 * biased_integral
    return result


def compute_integral_unbiased(model, data, time, non_pad_mask, type_mask):
    """ Log-likelihood of non-events, using Monte Carlo integration. """

    num_samples = 100

    diff_time = (time[:, 1:] - time[:, :-1]) * non_pad_mask[:, 1:]
    temp_time = diff_time.unsqueeze(2) * \
                torch.rand([*diff_time.size(), num_samples], device=data.device)
    temp_time /= (time[:, :-1] + 1).unsqueeze(2)

    temp_hid = model.linear(data)[:, 1:, :]
    temp_hid = torch.sum(temp_hid * type_mask[:, 1:, :], dim=2, keepdim=True)

    all_lambda = softplus(temp_hid + model.alpha * temp_time, model.beta)
    all_lambda = torch.sum(all_lambda, dim=2) / num_samples

    unbiased_integral = all_lambda * diff_time
    return unbiased_integral


def log_likelihood(model, data, time, types):
    """ Log-likelihood of sequence. """

    non_pad_mask = get_non_pad_mask(types).squeeze(2)

    type_mask = torch.zeros([*types.size(), model.num_types], device=data.device)
    for i in range(model.num_types):
        type_mask[:, :, i] = (types == i + 1).bool().to(data.device)

    all_hid = model.linear(data)
    all_lambda = softplus(all_hid, model.beta)
    type_lambda = torch.sum(all_lambda * type_mask, dim=2)

    # event log-likelihood
    event_ll = compute_event(type_lambda, non_pad_mask)
    event_ll = torch.sum(event_ll, dim=-1)

    # non-event log-likelihood, either numerical integration or MC integration
    # non_event_ll = compute_integral_biased(type_lambda, time, non_pad_mask)
    non_event_ll = compute_integral_unbiased(model, data, time, non_pad_mask, type_mask)
    non_event_ll = torch.sum(non_event_ll, dim=-1)

    return event_ll, non_event_ll


def type_loss(prediction, types, loss_func):
    """ Event prediction loss, cross entropy or label smoothing. """

    # convert [1,2,3] based types to [0,1,2]; also convert padding events to -1
    truth = types[:, 1:] - 1
    prediction = prediction[:, :-1, :]

    pred_type = torch.max(prediction, dim=-1)[1]
    correct_num = torch.sum(pred_type == truth)

    # compute cross entropy loss
    if isinstance(loss_func, LabelSmoothingLoss):
        loss = loss_func(prediction, truth)
    else:
        loss = loss_func(prediction.transpose(1, 2), truth)

    loss = torch.sum(loss)
    return loss, correct_num


def time_loss(prediction, event_time):
    """ Time prediction loss. """

    prediction.squeeze_(-1)

    true = event_time[:, 1:] - event_time[:, :-1]
    prediction = prediction[:, :-1]

    # event time gap prediction
    diff = prediction - true
    se = torch.sum(diff * diff)
    return se


class LabelSmoothingLoss(nn.Module):
    """
    With label smoothing,
    KL-divergence between q_{smoothed ground truth prob.}(w)
    and p_{prob. computed by model}(w) is minimized.
    """

    def __init__(self, label_smoothing, tgt_vocab_size, ignore_index=-100):
        assert 0.0 < label_smoothing <= 1.0
        super(LabelSmoothingLoss, self).__init__()

        self.eps = label_smoothing
        self.num_classes = tgt_vocab_size
        self.ignore_index = ignore_index

    def forward(self, output, target):
        """
        output (FloatTensor): (batch_size) x n_classes
        target (LongTensor): batch_size
        """

        non_pad_mask = target.ne(self.ignore_index).float()

        target[target.eq(self.ignore_index)] = 0
        one_hot = F.one_hot(target, num_classes=self.num_classes).float()
        one_hot = one_hot * (1 - self.eps) + (1 - one_hot) * self.eps / self.num_classes

        log_prb = F.log_softmax(output, dim=-1)
        loss = -(one_hot * log_prb).sum(dim=-1)
        loss = loss * non_pad_mask
        return loss