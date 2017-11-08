import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import freqz
from pyswarm import pso

import filters
import PSO

def filterModel(x):
    # [fc, bandwidth, gain]

    w_final = None
    db_final = 0
    fs = 44100

    for fc, BW, gain in x:
        b, a = filters.bandpass_peaking(fc=fc, gain=gain, BW=BW)
        w, h = freqz(b, a, worN=np.linspace(np.pi*2/fs*20, np.pi*2/fs*20e3, 1000))
        db = 20 * np.log10(abs(h))
        
        w_final = w
        db_final += db

    # plt.semilogx(w_final * fs / (2*np.pi), db_final)
    
    return w_final*fs/(2*np.pi), db_final
    
# target
# paras = [(1e4, 2500, 3), (300, 201, 10), (400, 600, 5), (600, 200, 8), 
         # (2000, 3500, 13), (6000, 4000, 3), (8500, 6000, 2.75),]
paras = [(1e4, 2500, 3), (300, 201, 10)] 
f_t, db_t = filterModel(paras)   

def fitness(x):   
    # compare
    x = np.array(x).reshape(-1, 3)
    x[:,0] = x[:,0]**10

    f_c, db_c = filterModel(x)
    global f_t, db_t
    # 計算 mse
    mse = ((db_t - db_c) ** 2).mean()
    
    return  mse

filterNum = len(paras)
xopt, fopt = pso(fitness, (np.log10(20),100,1)*filterNum, (np.log10(20e3),10000,20)*filterNum)
print('pyswarm', fopt)
opt = np.array(xopt).reshape(-1, 3)
opt[:,0] = opt[:,0]**10
print(opt)
f, db = filterModel(opt)
plt.semilogx(f, db, label="pyswarm")

pso = PSO.PSO(fitness, [(np.log10(20),np.log10(20e3)), (100,10000), (1,20)]*filterNum)
opt, optfit = pso.run(threshold=1e-2)
print(optfit)
opt = np.array(opt).reshape(-1, 3)
opt[:,0] = opt[:,0]**10
print(opt)
f, db = filterModel(opt)
plt.semilogx(f, db, label="PSO")

plt.semilogx(f_t, db_t, label="target", color='red')

plt.legend()
plt.show()
