import filters
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import freqz
from sklearn.neural_network import MLPRegressor

def filterModel(x):
    # [fc, bandwidth, gain]

    w_final = None
    db_final = 0
    fs = 44100

    for fc, BW, gain in x:
        b, a = filters.bandpass_peaking(fc=fc, gain=gain, BW=BW)
        w, h = freqz(b, a, worN=np.linspace(np.pi*2/fs*20, np.pi*2/fs*20e3, 500))
        db = 20 * np.log10(abs(h))
        
        w_final = w
        db_final += db

    # plt.semilogx(w_final * fs / (2*np.pi), db_final)
    
    return w_final*fs/(2*np.pi), db_final
    
    
def genXY(n, filtersNum):
    total = n * filtersNum
    fc = np.random.uniform(20, 20e3, size=(total,1))
    bw = np.random.uniform(100, 10000, size=(total,1))
    gain = np.random.uniform(0, 20, size=(total,1))

    Y = np.concatenate((fc,bw,gain), axis=1)
    Y = Y.reshape(n, filtersNum, 3)
    
    X = []

    for paras in Y:
        f, db = filterModel(paras)
        X.append(db)

    X = np.array(X)
    Y = Y.reshape(n, filtersNum*3)
    
    return X, Y
    

if __name__ == "__main__":
    # Create a random dataset
    # [fc, bandwidth, gain]
    n = 100
    filtersNum = 1

    X, Y = genXY(n=n, filtersNum=filtersNum)

    # Fit regression model
    regr = MLPRegressor(hidden_layer_sizes=(10,), max_iter=10000)
    regr.fit(X, Y)
    print('train loss', regr.loss_)

    # Predict
    X_test, Y_test = genXY(n=n, filtersNum=filtersNum)
    print('test loss', ((Y_test - regr.predict(X_test)) ** 2).mean())
    
    # paras = [(1e4, 2500, 3), (300, 201, 10), (400, 600, 5), (600, 200, 8), 
             # (2000, 3500, 13), (6000, 4000, 3), (8500, 6000, 2.75),]
    paras = [(1e4, 2500, 3),]
    f, db = filterModel(paras)
    plt.semilogx(f, db, label="target", color='red')
    
    y_pred = regr.predict([db])    
    f, db = filterModel(y_pred.reshape(filtersNum, 3))
    plt.semilogx(f, db, label="NN")
    
    plt.legend()
    plt.show()