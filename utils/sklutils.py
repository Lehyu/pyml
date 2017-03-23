from sklearn.model_selection import train_test_split

def load_data(data, test_size=0.2, random_state=0):
    X, y = data.data, data.target
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_val, y_train, y_val


def compare(skmodel, mymodel, data, score,test_size=0.2, random_state=0):
    X_train, X_val, y_train, y_val = load_data(data, test_size, random_state)
    skmodel.fit(X_train, y_train)
    mymodel.fit(X_train, y_train)
    print('sklearn model %.5f'%score(skmodel.predict(X_val), y_val))
    print('my model %.5f'%(score(mymodel.predict(X_val), y_val)))