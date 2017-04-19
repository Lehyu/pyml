from utils import ShuffleSpliter

__all__ = ()


class Spliter(object):
    def __init__(self, n_samples, cv=3, pre_split=False):
        self.cv = cv
        self.pre_split = pre_split
        self.spliter = ShuffleSpliter(n_samples=n_samples, test_size=1.0 / self.cv)

    def split(self):
        if not self.pre_split:
            self._split_real_time()
        else:
            for ix in self.cv_ixes_:
                yield ix

    def _split_real_time(self):
        for i in range(self.cv):
            yield self.spliter.shuffle()

    def _split_before_hand(self):
        self.cv_ixes_ = []
        for i in range(self.cv):
            self.cv_ixes_.append(self.spliter.shuffle())


class RandomSpliter(Spliter):
    def __init__(self, n_samples, cv, pre_split=False):
        super().__init__(n_samples, cv, pre_split)
        self.spliter = ShuffleSpliter(n_samples=n_samples, test_size=1.0 / self.cv)
        if self.pre_split:
            self._split_before_hand()


def p_train_test_split(data, condition):
    """
    :param data: pandas.DataFrame
    :param condition: test data set condition
    :return: train, test: pandas.DataFrame
    """
    train = data.loc[~condition]
    test = data.loc[condition]
    return train, test
