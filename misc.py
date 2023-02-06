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
