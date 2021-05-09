import numpy as np
import logging
from collections import Counter
from util.norm import norm_function

def get_estimated_norm(normType, stream, n, w, sRate=None, isUniSampled=True, keptDig =2 ,aveNum=1):
    norm_fn = norm_function(normType)
    normEx = get_exact_norm(norm_fn, stream,n, w)
    if keptDig is not None: normEx = np.round(normEx,keptDig+1)
    if isUniSampled:
        normUn, unStd =get_averaged_uniform_sampled_norm(aveNum, norm_fn, stream, n, w, sRate=sRate)
        errUn = abs(normEx-normUn)/normEx
        unStd= unStd/normEx
        if keptDig is not None: 
            normUn = np.round(normUn,keptDig)
            errUn =  np.round(errUn,keptDig)
            unStd =  np.round(unStd,keptDig)

    else:
        normUn, errUn, unStd = 0, 0, 0
    return normEx, normUn, errUn, unStd

def get_freqList(stream, n=None, m=None):
    # print(n)
    c = Counter(stream)   
    if n is not None and m is not None and m>n:
        freqList=[]
        for i in range(1, n+1):
            freqList.append(c[i])
        assert len(freqList) == n
    else:
        freqList = list(c.values())
    # print(c.most_common(8))
    # logging.info('Freqlist{}'.format(freqList))
    return freqList


def get_exact_norm(norm_fn, stream, n, w):
    freqList = get_freqList(stream[-w:], n=n)
    normEx = norm_fn(freqList) 
    # logging.info('normEx in w {}: {}'.format(w,normEx))
    return normEx

def get_averaged_uniform_sampled_norm(aveNum, norm_fn, stream, n, w, sRate=0.1):
    unNorms = np.array([])
    for i in range(aveNum):
        unNorm = get_uniform_sampled_norm(norm_fn, stream, n, w, sRate=sRate)
        unNorms=np.append(unNorms, unNorm)
    unNorm = np.median(unNorms)
    unNormStd = np.std(unNorms)
    return unNorm, unNormStd

def get_uniform_sampled_norm(norm_fn, stream, n, w, sRate=0.1):
    # np.random.seed(926)
    samples = np.random.choice(stream, size=int(w*sRate))
    freqList = get_freqList(samples, n=n)
    samplesNorm = norm_fn(freqList)
    # samplesNorm = np.linalg.norm(freqList, ord=2)
    uniformNorm = np.sqrt(samplesNorm**2 / sRate)
    # logging.info('{:0.2f}-normUn in w {}: {}'.format(sRate,w, uniformNorm))
    return uniformNorm