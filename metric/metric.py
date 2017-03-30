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

# TianChi KDD task 2 evaluation metric
def MAPE(predict, y_val):
    if len(y_val.shape) == 2:
        mape = 0.0
        _len = 0
        for axis in range(y_val.shape[1]):
            mask = np.nonzero(y_val[:, axis] != 0)[0]
            res = np.abs(predict[mask,axis]-y_val[mask,axis])
            mape += np.sum(res/y_val[mask, axis])
            _len += len(y_val[mask])
        return mape/float(_len)
    else:
        mask = np.nonzero(y_val != 0)[0]
        res = np.abs(predict[mask]-y_val[mask])
        return np.sum(res/y_val[mask])/len(predict[mask])