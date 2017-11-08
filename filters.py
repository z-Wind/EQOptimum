import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import freqz, iirnotch

def highpass_base(fc, gain, fs=44100, Q=1/np.sqrt(2)):
    '''
    https://www.dsprelated.com/showcode/170.php
    fc：截止頻率
    gain：gain(dB)
    fs：取樣率
    Q：Q factor
    '''
    
    K = np.tan((np.pi*fc)/fs)
    V0 = 10**(gain/20)
    root2 = 1/Q # sqrt(2)
    
    b0 =             (1 + root2*K + K**2) / (1 + root2*np.sqrt(V0)*K + V0*K**2);
    b1 =                (2 * (K**2 - 1) ) / (1 + root2*np.sqrt(V0)*K + V0*K**2);
    b2 =             (1 - root2*K + K**2) / (1 + root2*np.sqrt(V0)*K + V0*K**2);
    a0 = 1
    a1 =             (2 * (V0*K**2 - 1) ) / (1 + root2*np.sqrt(V0)*K + V0*K**2);
    a2 = (1 - root2*np.sqrt(V0)*K + V0*K**2) / (1 + root2*np.sqrt(V0)*K + V0*K**2);
    
    return [b0, b1, b2], [a0, a1, a2]


def highpass_Treble_Shelf(fc, gain, fs=44100, Q=1/np.sqrt(2)):
    '''
    https://www.dsprelated.com/showcode/170.php
    fc：截止頻率
    gain：gain(dB)
    fs：取樣率
    Q：Q factor
    '''
    
    K = np.tan((np.pi*fc)/fs)
    V0 = 10**(gain/20)
    root2 = 1/Q # sqrt(2)
    
    b0 = (V0 + root2*np.sqrt(V0)*K + K**2) / (1 + root2*K + K**2);
    b1 =             (2 * (K**2 - V0) ) / (1 + root2*K + K**2);
    b2 = (V0 - root2*np.sqrt(V0)*K + K**2) / (1 + root2*K + K**2);
    a0 = 1
    a1 =              (2 * (K**2 - 1) ) / (1 + root2*K + K**2);
    a2 =           (1 - root2*K + K**2) / (1 + root2*K + K**2);
    
    return [b0, b1, b2], [a0, a1, a2]


def lowpass_shelf(fc, gain, fs=44100, Q=1/np.sqrt(2)):
    '''
    https://www.dsprelated.com/showcode/170.php
    fc：截止頻率
    gain：gain(dB)
    fs：取樣率
    Q：Q factor
    '''
    
    K = np.tan((np.pi*fc)/fs)
    V0 = 10**(gain/20)
    root2 = 1/Q # sqrt(2)
    
    b0 = (1 + np.sqrt(V0)*root2*K + V0*K**2) / (1 + root2*K + K**2);
    b1 = (2 * (V0*K**2 - 1) ) / (1 + root2*K + K**2);
    b2 = (1 - np.sqrt(V0)*root2*K + V0*K**2) / (1 + root2*K + K**2);
    a0 = 1
    a1 = (2 * (K**2 - 1) ) / (1 + root2*K + K**2);
    a2 = (1 - root2*K + K**2) / (1 + root2*K + K**2);
    
    return [b0, b1, b2], [a0, a1, a2]


def bandpass_peaking(fc, gain, fs=44100, BW=None, Q=None):
    '''
    http://www.itdadao.com/articles/c15a230507p0.html
    fc：中心頻率
    gain：gain(dB)
    fs：取樣率
    BW：fc 左右 g/2 頻點 F1~F2 的距離
    Q：Q=Fc/(F2-F1)
    '''
    if BW and Q is None:
        Q = fc/BW
    elif BW and Q:
        print("warning：同時存在 Q 和 BW，只取 Q 運算")
    
    A = 10**(gain/40)
    w = 2*np.pi*fc/fs
    alpha = np.sin(w)/(2*Q)
    G = 1/(1+alpha/A)
    
    b0 = G*(1+alpha*A)
    b1 = -2*G*np.cos(w)
    b2 = G*(1-alpha*A)
    
    a0 = 1
    a1 = b1
    a2 = G*(1-alpha/A)
    
    return [b0, b1, b2], [a0, a1, a2]
    
    
def bandpass_notch(fc, fs, Q):
    '''
    https://www.dsprelated.com/showcode/170.php
    fc：截止頻率
    fs：取樣率
    Q： -3 dB bandwidth bw, Q = wc/BW.
    '''
    return iirnotch(fc/(fs/2), Q)
    
    
def plotFigure(figNum, func, **kwargs):
    plt.figure(figNum)
    b, a = func(**kwargs)
    w, h = freqz(b, a, worN=np.linspace(np.pi*2/fs*20, np.pi*2/fs*20e3, 1000))
    db = 20 * np.log10(abs(h))

    plt.semilogx(w*fs/(2*np.pi), db)

    plt.xlabel('Frequency')
    plt.ylabel('Amplitude response [dB]')
    plt.grid()

    str = "{}("
    for argName in kwargs:
        str += (argName + "={{kwargs[{}]}}, ".format(argName))
    str = str[:-2] + ")"

    plt.title(str.format(func.__name__, kwargs=kwargs))

    # plt.show()
    

if __name__ == "__main__":
    fs = 44100
    fc = 1000
    gain = 10
    BW = 1000
    Q = fc/BW
    
    plotFigure(1, highpass_base, fc=fc, gain=gain, fs=fs)
    plotFigure(2, highpass_Treble_Shelf, fc=fc, gain=gain, fs=fs)
    plotFigure(3, lowpass_shelf, fc=fc, gain=gain, fs=fs)
    plotFigure(4, bandpass_peaking, fc=fc, gain=gain, fs=fs, BW=BW)
    plotFigure(5, bandpass_notch, fc=fc, fs=fs, Q=Q)
    


    # [fc, bandwidth, gain]
    paras = [(1e4, 2500, 3), (300, 201, 10), (400, 600, 5), (600, 200, 8), 
             (2000, 3500, 13), (6000, 4000, 3), (8500, 6000, 2.75),]

    w_final = None
    db_final = 0
    fs = 44100
    plt.figure(6)


    for fc, BW, gain in paras:
        b, a = bandpass_peaking(fc=fc, gain=gain, BW=BW)
        w, h = freqz(b, a, worN=np.linspace(np.pi*2/fs*20, np.pi*2/fs*20e3, 1000))
        db = 20 * np.log10(abs(h))
        
        plt.subplot(211)
        plt.semilogx(w * fs / (2*np.pi), db)
        
        w_final = w
        db_final += db

    plt.subplot(212)
    plt.semilogx(w_final * fs / (2*np.pi), db_final)

    plt.xlabel('Frequency')
    plt.ylabel('Amplitude response [dB]')

    plt.ylim(0, max(db_final)+5)
    plt.grid()
    plt.title("Test")
    plt.show()