import numpy as np


def precision(PredictSet, ReferenceSet):
    return len(PredictSet & ReferenceSet)/float(len(PredictSet))

def recall(PredictSet, ReferenceSet):
    return len(PredictSet & ReferenceSet)/float(len(ReferenceSet))

def F1(Precision, Recall):
    if isinstance(Precision, set) and isinstance(Recall, set):
        return 2.0*len(Precision & Recall)/(len(Precision)+len(Recall))
    elif isinstance(Precision, float) and isinstance(Recall, float):
        return 2.0*Precision*Recall/(Precision+Recall)
    else:
        raise ValueError("Precision's type is %s while recall's type is %s"%(type(Precision), type(Recall)))


def mse(predict, y_val):
    y_val = y_val.reshape(predict.shape)
    return np.sqrt(np.sum((predict - y_val) ** 2))

def accuracy(predict, y_val):
    _score = 0.0
    for i in range(len(predict)):
        _score += 1 if predict[i] == y_val[i] else 0
    return float(_score) / len(predict)

def metric(predict, y_val):
    return np.sum(np.abs((predict-y_val)/(predict+y_val)))/len(predict)