#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 07:58:45 2025

@author: riccardo
"""


import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy as sp
from platform import platform
from datetime import datetime


def Ver():
    SEP = "-"*90
    NOW = datetime.now()
    dt_string = NOW.strftime("%A, %B %d, %Y, %H:%M:%S")
    print(SEP)
    print(dt_string)
    print("Platform :", platform())
    print(SEP)
    print("Python", sys.version)
    try:
        import jupyterlab
        print("JupyterLab", jupyterlab.__version__)
    except:
        print("JupyterLab", "NONE")
    print(SEP)
    print("Matplotlib", mpl.__version__)
    print("Pandas", pd.__version__)
    print("NumPy", np.__version__)
    print("SciPy", sp.__version__)
    print(SEP)


def EndChart(Legend='off', Xlabel='', Ylabel='', Title='', SaveFigName=''):
    if Xlabel != '':
        plt.xlabel(Xlabel)
    if Ylabel != '':
        plt.ylabel(Ylabel)
    if Title != '':
        plt.title(Title)
    plt.grid()
    plt.grid(which='minor', linestyle=':')
    plt.minorticks_on()
    plt.tick_params(axis='x', rotation=35)
    if Legend == 'on':
        plt.legend(loc='best', fontsize='small')
    if SaveFigName != '':
        plt.savefig(SaveFigName, bbox_inches='tight')


def ChartItem(Day, Open, High, Low, Close, Style, FaceColor):
    if Style == 'candle':
        WIDTH = pd.DateOffset(hours=8)
        plt.vlines(x=Day, ymin=Low, ymax=min(Open, Close), color='b')
        plt.vlines(x=Day, ymin=max(Open, Close), ymax=High, color='b')
        X = [Day-WIDTH, Day+WIDTH, Day+WIDTH, Day-WIDTH]
        Y = [Open, Open, Close, Close]
        plt.fill(X, Y, facecolor=FaceColor, edgecolor='k', alpha=1)
    elif Style == 'bar':
        WIDTH = pd.DateOffset(hours=10)
        plt.vlines(x=Day, ymin=Low, ymax=High,
                   color=FaceColor, linewidth=2.5)
        plt.hlines(y=Open, xmin=Day-WIDTH, xmax=Day,
                   color=FaceColor, linewidth=2.5)
        plt.hlines(y=Close, xmin=Day, xmax=Day+WIDTH,
                   color=FaceColor, linewidth=2.5)
    else:
        pass


def ReadStockPrices(filename):
    T = pd.read_csv(filename).dropna() # Load the table
    T['Date'] = pd.to_datetime(T['Date'], format='%Y-%m-%d', errors='coerce') # Convert Date in datetime
    N=T.iloc[:, 1:].values
    if np.isnan(N).any(): # Check NaN presence
        raise ValueError("NaN values still present")
    if T.iloc[0, 0] > T.iloc[-1, 0]: # Flip if not in time ascending order
        T = T[::-1].reset_index(drop=True)
    D=T['Date'].values
    if any(D[:-1]>=D[1:]): # Check time ascending order
        raise ValueError("Data not in time ascending order")
    return T


def FillBetween(D, Y1, Y2, d1, d2, Color='y'):
    rows = np.where((D >= d1) & (D <= d2))[0]
    r1 = rows[0]
    r2 = rows[-1]
    Ds=D[r1:(r2+1)]
    Y1s=Y1[r1:(r2+1)]
    Y2s=Y2[r1:(r2+1)]
    plt.fill_between(Ds, Y1s, Y2s, where=Y1s > Y2s, color=Color, alpha=0.2)


def MyFindIntersections(X, Y1, Y2):
    print(X)
    if len(X) == 0:
        raise ValueError("X is empty")
    if len(X) != len(Y1):
        raise ValueError("Lengths mismatch for X and Y1")
    if len(X) != len(Y2):
        raise ValueError("Lengths mismatch for X and Y2")
    n = len(X)
    if isinstance(X[0], (int, float, np.int32, np.int64, np.float32, np.float64)):
        XI = np.full(n, np.nan)
    else:
        XI = np.full(n, pd.NaT)
    YI = np.full(n, np.nan)
    numint = 0
    Yd = Y2 - Y1
    if Yd[0] == 0:
        numint = 1
        XI[0] = X[0]
        YI[0] = Y1[0]
    for i in range(1, n):
        if Yd[i] == 0:
            numint += 1
            XI[numint] = X[i]
            YI[numint] = Y1[i]
        elif Yd[i] * Yd[i - 1] < 0:
            numint += 1
            alpha = -Yd[i - 1] / (Yd[i] - Yd[i - 1])  # 0 < alpha < 1
            XI[numint] = X[i - 1] + alpha * (X[i] - X[i - 1])
            YI[numint] = Y1[i - 1] + alpha * (Y1[i] - Y1[i - 1])
    XI = XI[:numint + 1]
    YI = YI[:numint + 1]
    return XI, YI


def MyBisection(f, x1, x2, Precision=10**(-11)):
    if not (x1 <= x2):
        raise ValueError(f"Interval [{x1},{x2}] is not valid")
    f1 = f(x1)
    f2 = f(x2)
    if f1 * f2 > 0:
        raise ValueError(f"Bisection method cannot be used: f1={f1}, f2={f2}")
    elif f1 == 0:
        return x1
    elif f2 == 0:
        return x2
    a = x1
    fa = f1
    b = x2
    fb = f2
    for h in range(10000):
        if (b - a) < Precision:
            break
        c = (a + b) / 2
        fc = f(c)
        if fc * fa < 0:
            b = c
            fb = fc
        elif fc * fb < 0:
            a = c
            fa = fc
        else:
            return c
    m = (fb - fa) / (b - a)
    return a - (fa / m)
