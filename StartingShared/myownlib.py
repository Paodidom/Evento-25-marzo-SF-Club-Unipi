#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 07:58:45 2025

@author: riccardo
"""


import matplotlib.pyplot as plt
import numpy as np
import metcomputlib as mc


def FindRows(D, d1, d2):
    rows = np.where((D >= d1) & (D <= d2))[0]
    if len(rows) == 0:
        raise ValueError("No data available for the requested period")
    return rows


def PlotVolumes(D, C, V, d1, d2):
    rows = FindRows(D, d1, d2)
    r1 = rows[0]
    r2 = rows[-1]
    for i in range(r1, r2+1):
        if i == 0:
            opt = 'b'
        elif C[i] < C[i-1]:
            opt = 'r'
        elif C[i] > C[i-1]:
            opt = 'g'
        else:
            opt = 'b'
        plt.bar(D[i], V[i], color=opt)


def PlotCandles(D, O, H, L, C, d1, d2):
    rows = FindRows(D, d1, d2)
    r1 = rows[0]
    r2 = rows[-1]
    for i in range(r1, r2+1):
        if C[i] < O[i]:
            opt = 'r'
        else:
            opt = 'g'
        mc.ChartItem(D[i], O[i], H[i], L[i], C[i], 'candle', opt)


def PlotOHLC(D, O, H, L, C, d1, d2):
    rows = FindRows(D, d1, d2)
    r1 = rows[0]
    r2 = rows[-1]
    for i in range(r1, r2+1):
        if i == 0:
            opt = 'b'
        elif C[i] < C[i-1]:
            opt = 'r'
        elif C[i] > C[i-1]:
            opt = 'g'
        else:
            opt = 'b'
        mc.ChartItem(D[i], O[i], H[i], L[i], C[i], 'bar', opt)


def PlotData(D, Y, d1, d2, Label):
    rows = FindRows(D, d1, d2)
    r1 = rows[0]
    r2 = rows[-1]
    plt.plot(D[r1:(r2+1)], Y[r1:(r2+1)], label=Label)


def PlotBars(D, Y, d1, d2, Color):
    rows = FindRows(D, d1, d2)
    r1 = rows[0]
    r2 = rows[-1]
    plt.bar(D[r1:(r2+1)], Y[r1:(r2+1)], color=Color)


def Mean(v):
    ListNotNan=np.where(np.invert(np.isnan(v)))[0]
    vok=np.array([v[i] for i in ListNotNan])
    if len(vok) == 0:
        M=np.nan
    else:
        M=np.sum(vok)/len(vok)
    return M


def Wmean(v,w):
    ListNotNan=np.where(np.invert(np.isnan(v)))[0]
    vok=np.array([v[i] for i in ListNotNan])
    wok=np.array([w[i] for i in ListNotNan])
    if len(vok) == 0:
        W=np.nan
    else:
        W=np.sum(vok * wok) / np.sum(wok)
    return W


def Var(v):
    ListNotNan=np.where(np.invert(np.isnan(v)))[0]
    vok=np.array([v[i] for i in ListNotNan])
    n = len(vok)
    if (n == 0) or (n == 1):
        V=np.nan
    else:
        M = np.sum(vok)/n
        V=np.sum((vok-M)**2)/(n-1)
    return V


def Std(v):
    return np.sqrt(Var(v))


def Sma(v, p):
    n = len(v)
    M = np.full(n, np.nan)
    for h in range(p-1, n):
        M[h] = Mean(v[(h-p+1):(h+1)])
    return M


def Wma(v, p):
    n = len(v)
    M = np.full(n, np.nan)
    pesi = np.arange(1, p + 1)
    for h in range(p-1, n):
        M[h] = Wmean(v[(h-p+1):(h+1)],pesi)
    return M


def EmaG(v, p, alpha):
    n = len(v)
    M = np.full(n, np.nan)
    for h in range(p-1, n):
        if np.isnan(M[h - 1]):
            M[h] = Mean(v[(h-p+1):(h+1)])
        elif np.isnan(v[h]):
            M[h] = M[h - 1]
        else:
            M[h] = (1 - alpha) * M[h - 1] + alpha * v[h]
    return M


def Ema(v, p):
    alpha = 2 / (p+1)
    return EmaG(v, p, alpha)


def EmaW(v, p):
    alpha = 1 / p
    return EmaG(v, p, alpha)


def Lhl(H, L, p):
    n = len(H)
    HP = np.full(n, np.nan)
    LP = np.full(n, np.nan)
    for h in range(p-1, n):
        HP[h] = max(H[(h-p+1):(h+1)])
        LP[h] = min(L[(h-p+1):(h+1)])
    return HP, LP


def Bollinger(v, p, ds):
    n = len(v)
    Bmedian = np.full(n, np.nan)
    Bwidth = np.full(n, np.nan)
    for h in range(p-1, n):
        Bmedian[h] = Mean(v[(h-p+1):(h+1)])
        Bwidth[h] = ds * Std(v[(h-p+1):(h+1)])
    Btop = Bmedian + Bwidth
    Bbottom = Bmedian - Bwidth
    return Btop, Bmedian, Bbottom


def Macd(v, ps, pl, pg):
    macd = Ema(v, ps) - Ema(v, pl)
    signal = Ema(macd, pg)
    macdhist = macd - signal
    return macd, signal, macdhist






