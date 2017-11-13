import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import freqz
from collections import Counter

import filters
import PSO

def filterModel(x):
    # [fc, bandwidth, gain]

    w_final = None
    db_final = 0
    fs = 44100

    for fc, BW, gain in x:
        b, a = filters.bandpass_peaking(fc=fc, gain=gain, BW=BW)
        w, h = freqz(b, a, worN=np.pi*2/fs*np.logspace(np.log10(20), np.log10(20e3), 1000))
        db = 20 * np.log10(abs(h))
        
        w_final = w
        db_final += db

    # plt.semilogx(w_final * fs / (2*np.pi), db_final)
    
    return w_final*fs/(2*np.pi), db_final
    
# target
paras = [(1e4, 2500, 3), (300, 201, 10), (400, 600, 5), (600, 200, 8), 
         (2000, 3500, 13), (6000, 4000, 3), (8500, 6000, 2.75),]
# paras = [(1e4, 2500, 3), (300, 201, 10), (400, 600, 5), (600, 200, 8),] 
paras = np.array(paras)
f_t, db_t = filterModel(paras)   

def fitness(x):   
    # compare
    x = np.array(x).reshape(-1, 3)
    
    x[:,0] = paras[:,0]
    f_c, db_c = filterModel(x)
    global f_t, db_t
    # 計算 mse
    mse = ((db_t - db_c) ** 2).mean()
    
    # 計算符合程度
    familiar = 1/(np.sum(db_c-db_t < 0.01)+1)*100
    
    # 計算頻率分散
    f_count = Counter(np.round(x[:,0]))
    sameCount = len([item for item, count in f_count.items() if count > 1])
    
    # print("fit {} {} {}".format(mse, familiar, sameCount))
    fit = mse + familiar + sameCount
    return fit



pso = PSO.PSO(fitness, [(20,10e3), (100,10000), (1,20)]*len(paras), swarmSize=100, w=0.5, wp=0.5, wg=0.5)
opt, optfit = pso.run(threshold=1e-2)
print(optfit)
opt = np.array(opt).reshape(-1, 3)
opt = opt.astype(int)
opt[:,0] = paras[:,0]
print(opt[np.argsort(opt[:, 0])])
f, db = filterModel(opt)
plt.semilogx(f, db, label="PSO")

plt.semilogx(f_t, db_t, linestyle='--', label="target")
plt.legend()
plt.show()
#1.31