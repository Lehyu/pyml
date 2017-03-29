from sklearn.model_selection import train_test_split
import numpy as np

def load_data(data, test_size=0.2, random_state=0, ohe=False):
    X, y = data.data, data.target
    if ohe:
        tmp = np.zeros((X.shape[0], len(np.unique(y))))
        tmp[np.arange(X.shape[0]), y] = 1
        y = tmp
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_val, y_train, y_val


def compare(skmodel, mymodel, data, score,test_size=0.2, random_state=0, ohe=False):
    X_train, X_val, y_train, y_val = load_data(data, test_size, random_state, ohe)
    if ohe:
        y_val = y_val.argmax(axis=1)
    mymodel.fit(X_train, y_train)
    skmodel.fit(X_train, y_train if not ohe else y_train.argmax(axis=1))
    print('sklearn model %.5f' % score(skmodel.predict(X_val), y_val))
    print('my model %.5f'%(score(mymodel.predict(X_val), y_val)))
    print('mymodel ',[v for v in mymodel.predict(X_val).astype(int)])
    print('y_val   ', [v for v in y_val.astype(int)])

    print('skmodel ', [v for v in skmodel.predict(X_val).astype(int)])