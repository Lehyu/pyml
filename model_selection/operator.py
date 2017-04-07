import sys


class scorer(object):
    def __init__(self, scoring, greater_is_better=True):
        self._scorer = scoring
        self.greater_is_better = greater_is_better
        if self.greater_is_better:
            self.worst = -10
        else:
            self.worst = sys.maxsize

    def score(self, y_true, y_pred):
        return self._scorer(y_true, y_pred)

    def better(self, score, cur_score):
        if cur_score is None:
            return True
        if self.greater_is_better:
            return score > cur_score
        else:
            return score < cur_score


class post_operator(object):
    def __init__(self, operator):
        self.operator = operator

    def post_process(self, y, addition_y):
        return self.operator(y, addition_y)


def make_post_operator(operator):
    return post_operator(operator)


def make_scorer(scoring, greater_is_better=True):
    return scorer(scoring, greater_is_better)
