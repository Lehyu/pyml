from utils import ShuffleSpliter

__all__ = ()


class CVSpliter(object):
    def __init__(self, n_samples, cv):
        self.cv = cv
        self.spliter = ShuffleSpliter(n_samples=n_samples, test_size=1.0/self.cv)

    def split(self):
        for i in range(self.cv):
            yield self.spliter.shuffle()


def p_train_test_split(data, condition):
    """
    :param data: pandas.DataFrame
    :param condition: test data set condition
    :return: train, test: pandas.DataFrame
    """
    train = data.loc[~condition]
    test = data.loc[condition]
    return train, test
