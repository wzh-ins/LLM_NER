# -*- coding: utf-8 -*-
import sys
import math
import numpy as np
from matplotlib.pyplot import subplots
import matplotlib.pyplot as plt
import pandas as pd
import scipy.optimize
import scipy.signal
import scipy.fftpack


def to_profile(Y):
    y_avg = np.mean(Y)
    Y_trend = np.cumsum(Y - y_avg)
    return Y_trend


def linfunc(X, Y, power):
    p = np.polyfit(X, Y, power)
    Yfunc = np.polyval(p, X)
    return Yfunc


def dividing_by_segments(X, Y, S, power):
    N = Y.size
    F = []
    for s in S:  # for each s in S
        Ns = int(np.trunc(N/s))
        Fbuf = np.zeros(2 * Ns)  # F(v, s) ** 2
        for i in range(Ns):  # calculating values of F(v, s) ** 2 from 1 to Ns and from Ns + 1 to 2Ns
            Yn = linfunc(X[i * s : (i + 1) * s], Y[i * s : (i + 1) * s], power)
            Fbuf[i] = sum((Y[i * s : (i + 1) * s] - Yn) ** 2) / s
            Yn = linfunc(X[N - (i + 1) * s : N - i * s], Y[N - (i + 1) * s : N - i * s], power)
            Fbuf[i + Ns] = sum((Y[N - (i + 1) * s : N - i * s] - Yn) ** 2) / s
        F.append(Fbuf)
    return F


def qfunc(F, q):
    Fq = np.zeros(len(F))
    N = len(F)
    if q != 0:
        for i in range(N):
            Ns = F[i].size
            f_buf = sum(F[i] ** (q / 2))
            Fq[i] = ((f_buf) / Ns) ** (1 / q)
    else:
        for i in range(N):
            Ns = F[i].size
            f_buf = sum([np.log(x) for x in F[i]])
            Fq[i] = np.exp(f_buf / (2 * Ns))
    return Fq


def Hq(Fq, S):
    Hq = np.zeros_like(Fq[:, 0])
    dHq = np.zeros_like(Fq[:, 0])
    Nf = Fq[:, 0].size
    for i in range(Nf):
        p, cov = np.polyfit(np.log(S), np.log(Fq[i]), deg=1, full=False, cov=True)
        Hq[i] = p[0]
        dHq[i] = np.sqrt(cov[0][0])
    return Hq, dHq


def Boltzman(Q, A1, A2, x0, dx):
    return (A1 - A2) / (1 + np.exp((Q - x0) / dx)) + A2


def Boltzman_der(Q, A1, A2, x0, dx):
    return - ((A1 - A2) * np.exp((Q - x0) / dx)) / (dx * (1 + np.exp((Q - x0) / dx)) ** 2)


def Boltzman_fitting(Q, H, sigma=None):
    return scipy.optimize.curve_fit(Boltzman, Q, H, sigma=sigma)


def H_der(Q, H, dH):
    Nh = H.size - 1
    Hder = np.zeros(Nh)
    dHder = np.zeros(Nh)
    for i in range(Nh):
        Hder[i] = (H[i + 1] - H[i]) / (Q[i + 1] - Q[i])
        dHder[i] = math.sqrt(dH[i + 1] ** 2 + dH[i] ** 2) / (Q[i + 1] - Q[i])
    return Hder, dHder


def PSD_trend(X, Y, power):
    return Y - linfunc(X, Y, power)


def Psd(X, Y, k=0):
    N = X.size - 1
    step = 0
    for i in range(N):
        step += X[i + 1] - X[i]
    step /= N
    if k != 0:
        Y = PSD_trend(X, Y, k)
    f, P = scipy.signal.periodogram(Y, step)
    param, cov = np.polyfit(np.log(f[1:]), np.log(P[1:]), deg=1, cov=True)
    D_error = math.sqrt(cov[0][0])
    LinP = param[0] * np.log(f[1:]) + param[1]
    D = (5 + param[0]) / 2
    return f, P, LinP, D, D_error


def Boltzman_err(Q, A1, A2, x0, dx, dA1, dA2, dx0, ddx):
    A1_err = ((1 / (1 + np.exp((Q - x0) / dx))) * dA1) ** 2
    A2_err = ((1 - 1 / (1 + np.exp((Q - x0) / dx))) * dA2) ** 2
    x0_err = (((A1 - A2) * np.exp((Q - x0) / dx)) / (dx * (1 + np.exp((Q - x0) / dx)) ** 2) * dx0) ** 2
    dx_err = (((Q - x0) * (A1 - A2) * np.exp((Q - x0) / dx)) / ((dx ** 2) * (1 + np.exp((Q - x0) / dx)) ** 2) * ddx) ** 2
    return np.sqrt(A1_err + A2_err + x0_err + dx_err)


def WMandelbrot(t, b=1.5, d=1.8, M=50):
    W = np.zeros_like(t)
    for j in range(len(t)):
        for i in range(-M, M + 1):
            W[j] += 1 / (b ** ((2 - d) * i)) * (1 - np.cos(b ** i * t[j]))
    return W


def sequence_type(test_counter, fname):
    if test_counter == 0:
        data = pd.read_csv(fname)
        a = data.columns.tolist()
        X_with_str = np.array(data[a[0]][2:])
        Y_with_str = np.array(data[a[1]][2:])

        X = np.array([float(x) for x in X_with_str])
        Y = np.array([float(y) for y in Y_with_str])
    if test_counter == 1:
        N = 4000
        t = np.linspace(0, 1, N)
        X = np.arange(0, N)
        Y = WMandelbrot(t)
    if test_counter == 2:
        n_max = 12
        Nbin = 2 ** n_max
        Y = np.zeros(Nbin)
        a = 0.8
        for i in range(Nbin):
            Y[i] = a ** (bin(i).count('1')) * (1 - a) ** (n_max - bin(i).count('1'))
        X = np.arange(0, Nbin, 1)
    return X, Y


def mfdfa(fname, m, S, Q, trend_counter, test_counter):
    X, Y = sequence_type(test_counter, fname)
    if trend_counter == 0:
        Y_trend = Y.copy()
    else:
        Y_trend = Y.copy()
        for count in range(trend_counter):
            Y_trend = to_profile(Y_trend)
    Fq = np.zeros((Q.size, S.size))
    Nq = Q.size
    F = dividing_by_segments(X, Y_trend, S, m)
    for i in range(Nq):
        Fq[i] = qfunc(F, Q[i])

    Fq2 = np.zeros((2, S.size))
    Fq2[0] = qfunc(F, 0)
    Fq2[1] = qfunc(F, 2)

    '''
    if H2 - dH2 > 1.:
        H = H - 1
        H2 = H2 - 1
    '''
    Fq1 = np.zeros((2, S.size))
    Fq1[0] = qfunc(F, 0)
    Fq1[1] = qfunc(F, 1)

    return X, Y, Fq, F


