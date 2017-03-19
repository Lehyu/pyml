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