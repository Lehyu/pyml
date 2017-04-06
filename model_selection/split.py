
__all__ = ()

def p_train_test_split(data, condition):
    """
    :param data: pandas.DataFrame
    :param condition: test data set condition
    :return: train, test: pandas.DataFrame
    """
    train = data.loc[~condition]
    test = data.loc[condition]
    return train, test