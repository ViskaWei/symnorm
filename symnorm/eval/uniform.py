import logging
import numpy as np
# import scipy.stats
import pandas as pd
from collections import Counter



class UniformTrace():
    def __init__(self):
        self.freqVecUU = None
        self.normEx = None
        self.US3 = None
        self.UU3 = None
<<<<<<< HEAD
        self.sizeUS = None
        self.rateUS = None
        self.sizeUU = None
        self.rateUU = None

    

class Uniform():
    def __init__(self, norm_fns, sSize=None, sRate=None, aveNum=1, rDigit=2, trace=None):
=======
    

class Uniform():
    def __init__(self, norm_fns, sRate, aveNum=1, rDigit=2, trace=None):
>>>>>>> 8e8331afff0bb423cea2906d4cfec613c93a0fb3
        self.trace = trace

        self.norm_fns = norm_fns
        self.sRate = sRate
<<<<<<< HEAD
        self.sSize = sSize
=======
>>>>>>> 8e8331afff0bb423cea2906d4cfec613c93a0fb3
        self.rDigit = rDigit
        self.aveNum = aveNum
        self.fill_values = np.zeros(3)

<<<<<<< HEAD
        # if self.trace is not None: np.random.seed(227)
=======
        if self.trace is not None: np.random.seed(1178)
>>>>>>> 8e8331afff0bb423cea2906d4cfec613c93a0fb3

    def get_norms_from_coord(self, coord):
        counter = Counter(coord)
        freqVec = list(counter.values())
        norms = self.get_norms_from_freqVec(freqVec)
        return norms, counter

    def get_norms_from_freqVec(self, freqVec):
        if isinstance(self.norm_fns, list):
            return np.array([norm_fn(freqVec) for norm_fn in self.norm_fns])
        else:
            return self.norm_fns(freqVec)

    def get_norms_from_sampled_universe(self, counter, n):
<<<<<<< HEAD
        size, rate = self.get_sample_size_n_rate(n)

        sampledDict = np.random.choice(np.arange(1, n + 1), size=size, replace = False)
=======
        sampleSize = int(n * self.sRate)
        sampledDict = np.random.choice(np.arange(1, n + 1), size=sampleSize, replace = False)
>>>>>>> 8e8331afff0bb423cea2906d4cfec613c93a0fb3
        
        freqVec = [counter[item] for item in sampledDict]

        if self.trace is not None:
            self.trace.cUU = counter
            self.trace.sDict = sampledDict
            self.trace.freqVecUU = freqVec
<<<<<<< HEAD
            self.trace.rateUU = rate
            self.trace.sizeUU = size


        sampleNorm = self.get_norms_from_freqVec(freqVec)
        return sampleNorm / rate 

    def get_sample_size_n_rate(self, x):
        if self.sRate is not None and self.sSize is None:
            size = int(x * self.sRate)
            rate = self.sRate
        elif self.sSize is not None and self.sRate is None:
            size = self.sSize
            rate = self.sSize / x
        else:
            raise "sSize and sRate both not None"
        return size, rate

    def get_norms_from_sampled_coord(self, coord):
        size, rate = self.get_sample_size_n_rate(len(coord))
        sampleCoord = np.random.choice(coord, size=size, replace=False)
=======

        sampleNorm = self.get_norms_from_freqVec(freqVec)
        return sampleNorm / self.sRate 

    def get_norms_from_sampled_coord(self, coord):
        sampleSize = int(len(coord) * self.sRate)
        sampleCoord = np.random.choice(coord, size=sampleSize, replace = False)
>>>>>>> 8e8331afff0bb423cea2906d4cfec613c93a0fb3
        freqVec = list(Counter(sampleCoord).values())

        if self.trace is not None:
            self.trace.freqVecUS = freqVec
<<<<<<< HEAD
            self.trace.sizeUS = size
            self.trace.rateUS = rate

        sampleNorm = self.get_norms_from_freqVec(freqVec)

        return sampleNorm / rate 
=======

        sampleNorm = self.get_norms_from_freqVec(freqVec)
        return sampleNorm / self.sRate 
>>>>>>> 8e8331afff0bb423cea2906d4cfec613c93a0fb3
    
 
    def get_stats_from_norms(self, norms, normEx):
        norm, std = norms.mean(), norms.std()
        err = abs(normEx - norm) / normEx
        std = std / normEx
        return np.around([norm, err, std], self.rDigit)
################################################# EVAL NORMS ##############################################

    def run_MLoop(self, coord_full, mList, n):
        out =[]
        for m in mList:
            coord = coord_full[:m]
            normEx, US3, UU3 = self.run(coord, n)
            out.append([normEx, *US3, *UU3])
        return np.array(out)

<<<<<<< HEAD
    def run(self, coord, n=None):
=======
    def run(self, coord, n):
>>>>>>> 8e8331afff0bb423cea2906d4cfec613c93a0fb3
        normEx, counter = self.get_norms_from_coord(coord)
        normEx = np.around(normEx, self.rDigit + 1)
        
        normUSs = np.array([self.get_norms_from_sampled_coord(coord) for i in range(self.aveNum)])
        US3 = self.get_stats_from_norms(normUSs, normEx)
<<<<<<< HEAD
        if n is not None:
            normUUs = np.array([self.get_norms_from_sampled_universe(counter, n) for i in range(self.aveNum)])
            UU3 = self.get_stats_from_norms(normUUs, normEx)
        else:
            UU3 = self.fill_values
=======

        normUUs = np.array([self.get_norms_from_sampled_universe(counter, n) for i in range(self.aveNum)])
        UU3 = self.get_stats_from_norms(normUUs, normEx)

>>>>>>> 8e8331afff0bb423cea2906d4cfec613c93a0fb3

        if self.trace is not None:
            self.trace.normEx = normEx
            self.trace.US3 = US3
            self.trace.UU3 = UU3

        return normEx, US3, UU3


    