import os
import sys
import numpy as np
import copy
import pandas as pd
import matplotlib.pyplot as plt


# def plot_symNorm(self):

def plot_mL(self, ax=None, UU=True, title=True):
    color = ['r', 'salmon']
    if ax is None: ax=plt.gca()
    try:
        df = self.dfEval
    except:
        df = self.trace.dfEval

    if len(self.wRList) == 1:
        w = self.wRList[0]
        if w == 1:
            ax.plot(df['m'], df['errCs'], 'o-', c=color[0], label = f'Sketch')
        else:
            ax.plot(df['m'], df['errCs'], 'o-', c=color[0], label = f'Sketch (w={w})')
        
    else:
        for widx, w in enumerate(self.wRList):
            df0 = df[df['w'] == w]
            ax.plot(df0['m'], df0['errCs'], 'o-', c=color[widx], label = f'Sketch (w = m/{w})')

    ax.plot(df['m'], df['errUS'], 'bo-', label = 'Uniform (Stream)')
    lbUS = np.clip(df['errUS'] - df['stdUS'], 0, None)
    ubUS = df['errUS'] + df['stdUS']
    ax.fill_between(df['m'], lbUS, ubUS, alpha=0.3, color = 'lightblue', label = "1$\sigma$-CI")
    ax.set_xscale('log', basex=2)

    if UU:
        ax.plot(df['m'], df['errUU'], 'go-', label = 'Uniform (Universe)')
        ax.autoscale(0)
        lbUU = np.clip(df['errUU'] - df['stdUU'], 0, None)
        ubUU = df['errUU'] + df['stdUU']
        ax.fill_between(df['m'], lbUU, ubUU, alpha=0.3, color = 'lightgreen', label = "1$\sigma$-CI")


    
    ax.set_xlabel('stream size m')
    ax.set_ylabel('% error')
    if title:
        ax.set_title(f"{self.normType} | {self.ftr} | rc = {df['rc'][0]} |")
    # plt.yscale('log', basey=2)
    # ax.legend(bbox_to_anchor=(1.5, 0.5), loc='center right', ncol=1)
    ax.legend()

def plot_sL(self, ax=None, title=True):
    if ax is None: ax=plt.gca()

    try:
        df = self.dfEval
    except:
        df = self.trace.dfEval
    for normType in self.normTypeList:
        df0 = df[df['norm'] == normType]
        ax.plot(df0['rc'], df0['errCs'], 'o-', label = f'{normType}')
    
    ax.set_xlabel('log2(table size)')
    ax.set_ylabel('% error')
    if title:
        ax.set_title(f"| m = {df['m'][0]} | {self.ftr} |")
    # plt.yscale('log', basey=2)
    # ax.legend(bbox_to_anchor=(1.5, 0.5), loc='center right', ncol=1)
    ax.legend()
