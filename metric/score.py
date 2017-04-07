import numpy as np


def precision(PredictSet, ReferenceSet):
    return len(PredictSet & ReferenceSet) / float(len(PredictSet))


def recall(PredictSet, ReferenceSet):
    return len(PredictSet & ReferenceSet) / float(len(ReferenceSet))


def F1(Precision, Recall):
    if isinstance(Precision, set) and isinstance(Recall, set):
        return 2.0 * len(Precision & Recall) / (len(Precision) + len(Recall))
    elif isinstance(Precision, float) and isinstance(Recall, float):
        return 2.0 * Precision * Recall / (Precision + Recall)
    else:
        raise ValueError("Precision's type is %s while recall's type is %s" % (type(Precision), type(Recall)))


def mse(y, y_pred):
    return np.sqrt(np.sum((y - y_pred) ** 2))


def accuracy(y, y_pred):
    _score = 0.0
    for i in range(len(y_pred)):
        _score += 1 if y[i] == y_pred[i] else 0
    return float(_score) / len(y_pred)


def metric(y, y_pred):
    return np.sum(np.abs((y_pred - y) / (y_pred + y))) / len(y)


# TianChi KDD task 2 evaluation metric
def MAPE(y, y_pred):
    return np.mean(np.abs(y_pred - y) / y)
