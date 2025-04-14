#!/usr/bin/env python
"""
Statistical methods for identifying serial crimes and related offenders

Copyright (c) 2025, A.A. Bessonov (bestallv@mail.ru)

version 2.0

Routines in this module:

bayes_prob(prob, drop=0)
bayesPairs(pequal, drop=0)
cat_levels(levs)
clusterPath(crimeID, tree, n=0)
compare_categorical(C1, C2, levs=None, binary=False)
compare_numeric(C1, C2)
compare_spatial(C1, C2, longlat=False)
compareСrimes(Pairs, crimedata, varlist, binary=True, longlat=False, showpb=False, method="convolution")
comparetemporal(DT1, DT2, method="convolution", show_pb=False)
comparisonCrime(crimedata, crimeID)
crimeClust_bayes(crimeID, spatial=None, t1=None, t2=None, Xcat=None, Xnorm=None,
                      max_criminals=1000, iters=10000, burn=5000,
                      show_pb=False, update=100, seed=None,
                      use_space=True, use_time=True, use_cats=True)
crimeClust_Hier(crimedata, varlist, estimateBF,
                     linkage_method='average', **kwargs)
crimeCount(seriesdata)
crimeLink(crimedata, varlist, estimateBF, sort=True)
crimeLink_Clust_Hier(crimedata, predGB, linkage)
datapreprocess(data)
difftime(time1, time2, tz=None, units='auto')
expabsdiff(X, Y)
expabsdiff_circ(X, Y, mod=24, n=2000, method="convolution")
GBC(X, Y, start, end, step, n_splits=5, learning_rate=0.2, **kwargs)
get_d(y, X, mod=24)
getBF(x, y, weights=None, breaks=None, df=5)
getCrimes(offenderID, crimedata, offenderTable)
getCrimeSeries(offender_id, offender_table, restrict=None, show_pb=False)
getCriminals(crimeID, offender_table)
getROC(f, y)
graphDataFrame(edges, directed=True, vertices=None, use_vids=False)
haversine(x, y)
linkage_sID(BF, group, method='average')
make_breaks(x, mode='quantile', nbins=None, binwidth=None)
make_groups(X, method=1)
make_linked(X, thres=365)
make_unlinked(X, m=40, thres=365, seed=None, method=1)
makePairs(X, thres=365, m=40, seed=None, method=1)
makeSeriesData(crimedata, offender_table, time="midpoint")
naiveBayes(data, var, weights=None, df=20, nbins=30, partition='quantile')
naivebayesfit(X, y, weights=None, df=20, nbins=30, partition='quantile')
plot_bf(BF, log_scale=True, show_legend=True, xlim=None, ylim=None, 
            cols=('darkred','darkblue'), background=True, bkgcol='lightgray',
            figsize=(8, 5), ax = None)
plot_crimeClust_bayes(data, ind, legend_shrink=0.9, figsize=(10, 7),
                    step_yticks = 10, step_xticks = 100, y_ticks_revers = False,
                    cmap = 'viridis', main_title = 'Probability crimes are linked',
                    y_label = 'Unsolved Crime', x_label = 'All crime')
plot_hcc(tree, labels=None, yticks=np.arange(-2, 9, 2), figsize=(15, 7), hang=-1, font_size=10, **kwargs)
plotBF(BF, var, logscale=True, figsize=(17,28), plotstyle='ggplot', legend=True, **kwargs)
plotHCL(Z, labels, figsize=(15,8), **kwargs)
plotnaiveBayes(x, **kwargs)
plotROC(x, y, xlim, ylim, xlabel, ylabel, title, rocplot=True, plotstyle='classic')
predict_bf(BF, x, log=True)
predictGB(X, varlist, gB)
predictnaiveBayes(model, newdata, components=False, var=None, log=True)
seq(start, stop, step)
seriesCrimeID(offenderID, unsolved, solved, offenderData, varlist, estimateBF)
seriesOffenderID(crime, unsolved, solved, seriesData, varlist, estimateBF,
              linkage_method='average', group_method=3, **kwargs)

"""
from __future__ import division, absolute_import, print_function

__all__ = ['bayes_prob', 'bayesPairs', 'cat_levels', 'clusterPath', 'compare_categorical',
           'compare_numeric', 'compare_spatial', 'compareСrimes', 'comparetemporal',
           'comparisonCrime', 'crimeClust_bayes', 'crimeClust_Hier', 'crimeCount', 
           'crimeLink', 'crimeLink_Clust_Hier', 'datapreprocess', 'difftime', 'expabsdiff', 
           'expabsdiff_circ', 'GBC', 'get_d', 'getBF', 'getCrimes', 'getCrimeSeries', 
           'getCriminals', 'getROC', 'graphDataFrame', 'haversine', 'linkage_sID', 'make_breaks', 
           'make_groups', 'make_linked', 'make_unlinked', 'makePairs', 'makeSeriesData', 
           'naiveBayes', 'naivebayesfit', 'plot_bf', 'plot_crimeClust_bayes', 'plot_hcc', 
           'plotBF', 'plotHCL', 'plotnaiveBayes', 'plotROC', 'predict_bf', 'predictGB', 
           'predictnaiveBayes', 'seq', 'seriesCrimeID', 'seriesOffenderID']


import numpy as np
import pandas as pd
import math
from tqdm import tqdm
from scipy.stats import uniform
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from itertools import combinations
from igraph import Graph
from scipy.cluster.hierarchy import linkage, fcluster
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import dendrogram
from datetime import datetime, timedelta
from scipy.stats import uniform
import matplotlib.patches as mpatches
from datetime import datetime, timedelta
from sklearn.metrics import auc
from scipy.stats import norm, truncnorm, gamma
from scipy.special import gammaln
from sklearn.model_selection import train_test_split
from scipy.fft import fft, ifftn
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import KFold, GridSearchCV


def bayes_prob(prob, drop=0):
    """Extracts the crimes with the largest probability of being linked

    Extracts the crimes from result produced by crimeClust_bayes with the largest 
    probability of being linked.

    Parameters
    ----------
    prob : [dataframe] a column (or row) of the posterior probability matrix
                 produced by crimeClust_bayes.
    drop : [int] only return crimes with a posterior linkage probability
                 that exceeds drop (default: drop=0).

    Returns
    ----------
    dataframe of the indices of crimes with estimated posterior probabilities,
    ordered from largest to smallest.

    Example
    ----------
    pp=fit['p_equal']
    np.fill_diagonal(pp, np.nan)
    bp = bayes_prob(pp[A['crimeID'] == "Crime417"])
    bp['crimeID'] = np.array(A.loc[bp['index'].to_list(), 'crimeID'])
    bp['CG'] = np.array(A.loc[bp['index'].to_list(), 'CS'])
    """
    # Преобразуем prob в numpy массив
    prob.shape = -1
    # Получаем индексы сортировки в порядке убывания
    ord_indices = np.argsort(prob)[::-1] 
    # Создаем DataFrame с индексами и вероятностями
    a = pd.DataFrame({
        'index': ord_indices, 
        'prob': prob[ord_indices]
    })
    # Если drop является числом, фильтруем по вероятностям
    if isinstance(drop, (int, float)):
        a = a[a['prob'] > drop]
    return a


def bayesPairs(pequal, drop=0):
    """
    Generate pairs of indices with probabilities from a given probability matrix.
    
    Extracts the crimes from result produced by crimeClust_bayes with the largest 
    probability of being linked.
    
    Parameters
    ----------
    pequal : [dataframe] the posterior probability matrix produced by crimeClust_bayes.
    drop : [int] only return crimes with a posterior linkage probability
                 that exceeds drop (default: drop=0).

    Returns
    ----------
    dataframe of the indices of crimes with estimated posterior probabilities,
    ordered from largest to smallest.

    Example
    ----------
    """
    def lowertriuindices(m):
        """
        Returns the indices of the lower triangular part of the matrix, 
        excluding the diagonal.
        """
        return np.tril_indices(m, k=-1)
    pp = np.array(pequal)  # Убедимся, что pequal в виде numpy массива
    np.fill_diagonal(pp, np.nan)  # Заменяем диагональные элементы на NaN
    lowtriindices = lowertriuindices(pp.shape[0])  # Получаем индексы нижнего треугольника
    prob = pp[lowtriindices]  # Вероятности на нижнем треугольнике
    row, col = lowtriindices  # Получаем индексы строк и столбцов
    flip = row > col  # Проверяем, где строки больше столбцов
    a = pd.DataFrame({
        'i1': np.where(flip, col, row), 
        'i2': np.where(flip, row, col), 
        'prob': prob
    })
    a = a.sort_values(by='prob', ascending=False)  # Сортируем по вероятности в порядке убывания

    if isinstance(drop, (int, float)):  # Проверяем, если drop - это число
        a = a[a['prob'] > drop]  # Фильтруем по условию
    return a


def cat_levels(levs):
    """
    Make levels for merging category predictors
    
    Parameters
    ----------
    levs : [dataframe] levels of a catagorical variable (factor)
    
    Returns
    ----------
    levels for a new categorical variable of form f1:f2
    """
    # Удаляем дубликаты и добавляем NA, если его нет
    levs = np.unique(np.concatenate([levs, [np.nan]]))  # Добавляем NA, если его еще нет
    nlevs = len(levs)
    a = []  # Список для хранения комбинаций
    # Генерируем комбинации
    for i in range(nlevs):
        for j in range(i, nlevs):
            a.append((levs[i], levs[j]))  # Создаем кортежи для комбинаций
    # Создаем строки с объединенными значениями
    levs2 = [f"{x[0]}:{x[1]}" for x in a]
    return levs2


def clusterPath(crimeID, tree, n=0):
    """
    Follows path of linking one crime of other crimes up a dendrogram

    The sequence of crimes groups that a crime belongs to.

    Parameters
    ----------
    crimeID : [str] crime ID of under study crime.
    tree : [array-like of shape] an object produced from function crimeClust_hier.
    n : [int] bayes factor value threshold for path return (default: n=0).

    Returns
    ----------
    Dataframe of the additional crimes and the log Bayes factor at each merge.

    Example
    ----------
    clusterPath('Crime2',tree,n=1)
    """
    ll=list(tree['crimeID'])
    if crimeID not in ll:
        raise ValueError("Error in crime ID")
    bf=-tree['hc'][:,2]+tree['offset']
    df=pd.DataFrame({'i1':tree['hc'][:,0],'i2':tree['hc'][:,1]}).astype(int).rename(index = lambda x: x + 1)
    def list_flatten(data):
        nested = True
        while nested:
            new = []
            nested = False
            for i in data:
                if isinstance(i, list):
                    new.extend(i)
                    nested = True
                else:
                    new.append(i)
            data = new
        return data
    res2=[]
    for j in range(len(tree['crimeID'])-1):
        cc=list(df.iloc[j])
        res1=[]
        for i in range(len(cc)):
            if cc[i] < len(tree['crimeID']):
                res=ll[cc[i]]
            if cc[i] >= len(tree['crimeID']):
                indx=cc[i]-len(tree['crimeID'])
                res=res2[indx]
            res1.append(res)
            res1=list_flatten(res1)
        res2.append(res1)
    DF=pd.DataFrame({'logBF':bf,'crimes':res2})
    lc=[]
    for i in range(DF.shape[0]):
        if crimeID in DF['crimes'][i]:
            lc.append(i)
    DFF=DF.iloc[lc]   
    for i in range(DFF.shape[0]):
        DFF['crimes'].iloc[i].remove(crimeID)
    DFF=DFF.reset_index(drop=True)
    DFF = DFF[DFF['logBF'] > n]
    def finelDF(DFF):
        if DFF.shape[0]==1:
            return DFF
        else:
            for i in range(DFF.shape[0]-1,-1,-1):
                result = [num for num in DFF['crimes'].iloc[i] if num in DFF['crimes'].iloc[i-1]]
                for j in range(len(result)):
                    DFF['crimes'].iloc[i].remove(str(result[j]))
            return DFF
    Dff=finelDF(DFF)
    Dff.index += 1
    print(Dff)
    return Dff
    if Dff.shape[0]==0:
        print("Change the value -n-")


def compare_categorical(C1, C2, levs=None, binary=False):
    """
    Make evidence variables from categorical crime data
    
    Compares categorical crime data to check if they match.
    
    Parameters
    ----------
    C1 : [dataframe] categorical values of crime attributes.
    C2 : [dataframe] categorical values of crime attributes.
    levs : [list or array] the levels of all possible values (default: levs=None).
    binary : [bool] match/no match or all combinations (default: binary=False).
    
    Returns
    ----------
    if binary=TRUE: 1 for match, 0 for non-matches;
    if binary=FALSE: factor vector of merged values (in form of f1:f2)
    """
    if binary:
        # Считаем совпадение NA как совпадение
        C1=C1.reset_index(drop=True)
        C2=C2.reset_index(drop=True)
        match_na = (C1.isna() & C2.isna())  # Логический массив совпадающих NA
        match_value = (C1.astype(str) == C2.astype(str))  # Сравнение значений
        # Создаем массив 1 для совпадений и 0 для несоответствий
        B = np.where(match_value | match_na, 1, 0).astype(int)
        B[np.isnan(B)] = 0  # Устанавливаем NA в 0 (несовпадение)
        return pd.Series(B, dtype='category')  # Возвращаем как категориальный тип
    # Конвертируем в категориальные переменные
    C1 = pd.Categorical(C1, categories=levs)
    C2 = pd.Categorical(C2, categories=levs)
    # Проверка уровней, если не указаны
    if levs is None:
        levs = sorted(set(C1.categories).union(C2.categories))
    # Создаем DataFrame для объединенных значений
    A = pd.DataFrame({'C1': C1, 'C2': C2})
    # Меняем местами значения, если первое значение больше второго
    flip = A['C1'].isna() | (A['C1'].cat.codes > A['C2'].cat.codes)
    A.loc[flip, ['C1', 'C2']] = A.loc[flip, ['C2', 'C1']].values  # Меняем местами строки
    # Объединяем значения
    B = A['C1'].astype(str) + ':' + A['C2'].astype(str)
    # Устанавливаем уровни и возвращаем как категориальный тип
    B = pd.Categorical(B, categories=[f'{x}:{y}' for x in levs for y in levs])
    return B


def compare_numeric(C1, C2):
    """
    Make evidence variables from categorical crime data

    Compares categorical crime data to check if they match.
    
    Parameters
    ----------
    C1 : [dataframe] numerical values of crime attributes.
    C2 : [dataframe] numerical values of crime attributes.
    
    Returns
    ----------
    numeric vector of absolute differences.
    """
    C1=C1.reset_index(drop=True)
    C2=C2.reset_index(drop=True)
    return np.where(C1 == C2, 1, 0)


def compare_spatial(C1, C2, longlat=False):
    """
    Make spatial evidence variables.

    Calculates spatial distance between crimes (in km)

    Parameters
    ----------
    C1 : [DataFrame] dateframe with 2 columns of coordinates for the crimes.
    C2 : [DataFrame] dateframe with 2 columns of coordinates for the crimes.
    longlat : [bool] if false (default) the the coordinates are in (Long,Lat),
                     else assume a suitable project where euclidean distance
                     can be applied (default: longlat=False).

    Returns
    ----------
    numeric vector of distances between the crimes (in km) internal.
    """
    df=pd.concat([C1.reset_index(drop=True),C2.reset_index(drop=True)], axis=1, ignore_index=True)
    if longlat:
        # Используем great_circle из geopy для расчета большого круга
        d = df.apply(lambda x: haversine((x[1], x[0]), (x[3], x[2])), axis = 1)
    else:
        # Вычисляем евклидово расстояние
        d = np.sqrt((df.iloc[:, 0]-df.iloc[:, 2])**2+(df.iloc[:, 1]-df.iloc[:, 3])**2)/1000
    return d  # Возвращаем расстояние в километрах


def compareСrimes(Pairs, crimedata, varlist, binary=True, longlat=False, showpb=False, method="convolution"):
    """
    Creates evidence variables by calculating distance between crime pairs.

    Calculates spatial and temporal distance, difference in categorical, and absolute value of numerical crime variables

    Parameters
    ----------
    Pairs : [DataFrame] dateframe with 2 columns of crime IDs that are checked for linkage.
    crimedata : [DataFrame] dataframe of crime incident data. There must be a column named of crimes
                            that refers to the crimeIDs given in dfPairs. Other column names must correspond
                            to what is given in varlist.
    varlist : [dict] a list with elements named: crimeID, spatial, temporal and categorical. Each element
                     should be a column names of crimedata corresponding to that feature: crimeID - crime ID
                     for the crimedata that is matched to dfPairs, spatial - X,Y coordinates (in long and lat)
                     of crimes, temporal - DT.FROM, DT.TO of crimes, categorical - categorical crime variables.
    binary : [bool] match/no match or all combinations for categorical data (default: binary=True).
    longlat : [bool] are spatial coordinates (long,lat), calculated using the haversine method
                     or Euclidean distance is returned in kilometers (default: longlat=False).
    showpb : [bool] show the progress bar (default: showpb=False).
    method : [str] use convolution (default) or monte carlo integration (method='numerical').

    Returns
    ----------
    data frame of various proximity measures between the two crimes.
    """
    # Преобразуем crimeID в строковый формат
    crimeID = list(crimedata['crimeID'].astype(str))
    # Поиск индексов для пар преступлений
    i1 = np.vectorize(pd.Index(crimeID).get_loc)(Pairs.iloc[:, 0])
    i2 = np.vectorize(pd.Index(crimeID).get_loc)(Pairs.iloc[:, 1])
    # Получаем пространственные данные
    spatial = crimedata[varlist['spatial']]
    d_spat = compare_spatial(spatial.iloc[i1, 0:2], spatial.iloc[i2, 0:2], longlat=longlat)
    # Получаем временные данные
    temporal = crimedata[varlist['temporal']]
    if temporal.shape[1] == 1:
        temporal = pd.concat([temporal, temporal], axis=1)  # Дублируем столбец, если только один
    d_temp = comparetemporal(temporal.iloc[i1, 0:2], temporal.iloc[i2, 0:2], show_pb=showpb, method = method)
    Edf = pd.concat([d_spat.to_frame('spatial'),d_temp],axis=1)
    # Получаем категориальные данные
    if ('categorical' in varlist):
        catNames = varlist['categorical']
        d_cat = pd.DataFrame(index=np.arange(len(i1)), columns=catNames)
        for cat in catNames:
            levs = list(set(list(crimedata.loc[i1, cat].dropna().unique()) 
                        + list(crimedata.loc[i2, cat].dropna().unique())))
            d_cat[cat] = compare_categorical(crimedata.loc[i1, cat], crimedata.loc[i2, cat], 
                                                        levs=levs, binary=binary)
        Edf = Edf.join(d_cat)
    # Получаем числовые данные
    if ('numerical' in varlist):
        numNames = varlist['numerical']
        d_num = pd.DataFrame(index=np.arange(len(i1)), columns=numNames)
        for num in numNames:
            d_num[num] = compare_numeric(crimedata.loc[i1, num], crimedata.loc[i2, num])
        Edf = Edf.join(d_num)
    # Создаем итоговый DataFrame
    E = pd.concat([Pairs.reset_index(drop=True), Edf], axis=1)
    return E


def comparetemporal(DT1, DT2, method="convolution", show_pb=False):
    """
    Make temporal evidence variable from (possibly uncertain) temporal.

    Calculates the temporal distance between crimes

    Parameters
    ----------
    DT1 : [DataFrame] dataframe of (DT.FROM,DT.TO) for the crimes.
    DT2 : [DataFrame] dataframe of (DT.FROM,DT.TO) for the crimes.
    method : [str] use convolution (default) or monte carlo integration (method='numerical').
    show_pb : [bool] show the progress bar (default: show_pb=False)

    Returns
    ----------
    dataframe of expected absolute differences: temporal - overall difference (in days)  [0,max], 
    tod - time of day difference (in hours)  [0,12], dow - fractional day of week difference (in days) [0,3.5].
    """
    # Длина интервала (в часах)
    L1=(abs(difftime(DT1.iloc[:,1].reset_index(drop=True),DT1.iloc[:,0].reset_index(drop=True),units='hours')))
    L2=(abs(difftime(DT2.iloc[:,1].reset_index(drop=True),DT2.iloc[:,0].reset_index(drop=True),units='hours')))

    # Время (в секундах с начала эпохи)
    day1 = np.array((DT1.iloc[:,0] - pd.Timestamp('1970-01-01')).dt.total_seconds()/86400)
    day2 = np.array((DT2.iloc[:,0] - pd.Timestamp('1970-01-01')).dt.total_seconds()/86400)
    # Время суток (оригинальное)
    tod1 = np.array((DT1.iloc[:,0].sub(pd.Timestamp(0)).dt.total_seconds() / 3600) % 24)
    tod2 = np.array((DT2.iloc[:,0].sub(pd.Timestamp(0)).dt.total_seconds() / 3600) % 24)
    # День недели (оригинальное)
    dow1 = np.array((DT1.iloc[:,0].sub(pd.Timestamp(0)).dt.total_seconds() / (3600 * 24)) % 7)
    dow2 = np.array((DT2.iloc[:,0].sub(pd.Timestamp(0)).dt.total_seconds() / (3600 * 24)) % 7)
    # Вычисляем временные абсолютные разности
    n = len(L1)
    temporal = np.zeros(n)
    tod = np.zeros(n)
    dow = np.zeros(n)
    if show_pb:
        progress_bar = tqdm(total=n, desc="Processing")
    for i in range(n):
        temporal[i] = expabsdiff([day1[i], day1[i] + L1[i] / 24], [day2[i], day2[i] + L2[i] / 24])
        tod[i] = expabsdiff_circ([tod1[i], tod1[i] + L1[i]], [tod2[i], tod2[i] + L2[i]], mod=24,method=method)
        dow[i] = expabsdiff_circ([dow1[i], dow1[i] + L1[i] / 24], [dow2[i], dow2[i] + L2[i] / 24], mod=7,method=method)
        if show_pb:
            progress_bar.update(1)
    if show_pb:
        progress_bar.close()
    return pd.DataFrame({'temporal': temporal, 'tod': tod, 'dow': dow})


def comparisonCrime(crimedata, crimeID):
    """
    Selection of certain crimes from a dataframe of crime incidents.

    Parameters
    ----------
    crimedata : [DataFrame] dataframe of crime incident data.
    crimeID : [list] an crime ID that must be extracted from data of crime incidents.

    Returns
    ----------
    Dataframe of certain crimes.
    """
    res=crimedata[crimedata['crimeID'].isin(crimeID)]
    return res


def crimeClust_bayes(crimeID, spatial=None, t1=None, t2=None, Xcat=None, Xnorm=None,
                      max_criminals=1000, iters=10000, burn=5000,
                      show_pb=False, update=100, seed=None,
                      use_space=True, use_time=True, use_cats=True):
    """
    Main function for performing Bayesian clustering on crime data
    
    Bayesian model-based partially-supervised clustering for crime series identification.
    
    Parameters
    ----------
    crimeID : [column of dataframe] n-vector of criminal IDs for the n crimes in the dataset.
    spatial : [columns of dataframe] (n x 2) matrix of spatial locations, represent missing
                                     locations with NA (default: spatial=None).
    t1 : [column of dataframe] earliest possible time for crime (default: t1=None).
    t2 : [column of dataframe] latest possible time for crime (default: t2=None).
    Xcat : [column of dataframe] (n x q) matrix of categorical crime features.  Each column is a variable,
                    such as mode of weapon, modus operandi, etc.  The different factors should be coded
                    as integers 1, 2, etc. (default: Xcat=None).
    Xnorm : [column of dataframe] (n x p) matrix of continuous crime features (default: Xnorm=None).
    max_criminals : [int] maximum number of clusters in the model (default: max_criminals=1000).
    iters : [int] number of MCMC samples to generate (default: iters=10000).
    burn : [int] number of MCMC samples to discard as burn-in (default: burn=5000).
    show_pb : [bool] show the progress bar (default: show_pb=False).
    update : [int] number of MCMC iterations between graphical displays (default: update=100).
    seed : [int] seed for random number generation (default: seed=None).
    use_space : [bool] should the spatial locations be used in clustering (default: use_space=True).
    use_time : [bool] should the event times be used in clustering (default: use_time=True).
    use_cats : [bool] should the categorical crime features be used in clustering (default: use_cats=True).
    
    Returns
    ----------
    dataframe: p.equal is the column of probabilities that each pair of crimes are committed by the same criminal.
    if show_pb=True, then progress plots are produced.
    """
    def ddir(y, df, p):
        """Вычисляет логарифм плотности вероятности для распределения Дирихле."""
        alpha = df * p
        lll = (gammaln(np.sum(alpha)) +
            np.sum(alpha * np.log(y)) -
            np.sum(gammaln(alpha)))
        return lll

    def count(j, y):
        """Counts occurrences of j in the array y."""
        return np.sum(y == j)

    def get_counts(y, M):
        """Returns the counts of the unique values in y up to M."""
        counts = np.zeros(M)
        for j in range(1, M + 1):
            counts[j - 1] = count(j, y)
        return counts

    def sumbyg(y, g, M):
        """Суммирует значения y по группам g и возвращает массив длиной M."""
        # Создаем DataFrame для группировки
        df = pd.DataFrame({'y': y, 'g': g})
        # Подсчитываем сумму по группам
        sss = df.groupby('g')['y'].sum().reindex(range(M), fill_value=0)
        return sss.values  # Возвращаем значения как массив

    def count_g(g, M):
        unig = np.unique(g)
        unig.sort()
        sss = np.zeros(M, dtype=int)
        for value in unig:
            sss[value] = np.count_nonzero(g == value)
        return sss

    def rtruncnorm(n, mu, sigma, lower, upper):
        """Generate random samples from a truncated normal distribution."""
        # Calculate the cumulative distribution function (CDF) values for lower and upper bounds
        lp = norm.cdf(lower, mu, sigma)
        up = norm.cdf(upper, mu, sigma)
        # Generate uniform samples and transform to the truncated normal distribution
        y = norm.ppf(np.random.uniform(lp, up, n), mu, sigma)
        return y

    def ddir2(probs, D):
        """Calculates the log density of a Dirichlet distribution."""
        k = len(probs[0])
        return (gammaln(k * D) - k * gammaln(D) + 
                (D - 1) * np.sum(np.log(probs)))

    if seed is not None:
        np.random.seed(seed)
    # Check for time
    if t1 is None:
        use_time = False
    else:
        if t2 is None:
            t2 = np.copy(t1)
        origen = min(t1.min(),t2.min())
        if t1.apply(lambda x: isinstance(x, (pd.Timestamp, np.datetime64))).all():
            t1=(difftime(t1, origen, units='days').round(5))
        if t2.apply(lambda x: isinstance(x, (pd.Timestamp, np.datetime64))).all():
            t2=(difftime(t2, origen, units='days').round(5))
        Xnorm = np.column_stack((t1, Xnorm)) if Xnorm is not None else np.array(t1)
    # Check for spatial data
    if spatial is None:
        use_space = False
    # Check for categorical data
    if Xcat is None:
        use_cats = False
    else:
        Xcat = np.apply_along_axis(lambda x: pd.factorize(x)[0] + 1, axis=0, arr=Xcat)
    M = min(max_criminals, len(crimeID))
    g = np.copy(crimeID)
    maxg = np.nanmax(g) if np.any(~np.isnan(g)) else 0
    miss = np.isnan(g)
    g[miss] = np.random.choice(np.arange(M), size=miss.sum(), replace=True)
    g=g.astype('int')
    n = len(g)
    p = Xnorm.shape[1] if len(Xnorm.shape) > 1 else 1
    q = Xcat.shape[1] if len(Xcat.shape) >1 else 1
    miss_norm = np.isnan(Xnorm) if Xnorm.size > 0 else np.zeros((n, 0), dtype=bool)
    miss_cat = np.isnan(Xcat) if Xcat.size > 0 else np.zeros((n, 0), dtype=bool)
    Xcat[miss_cat] = 1
    if p > 1:
        for j in range(1, p):
            meanval = np.nanmean(Xnorm[~miss_norm[:, j], j])
            Xnorm[miss_norm[:, j], j] = meanval
    ncats = np.max(Xcat, axis=0)
    maxcat = np.max(ncats)
    pcat = np.zeros((maxcat, q))
    for j in range(q):
        for l in range(maxcat):
            pcat[l, j] = 1 + np.sum(Xcat[:, j] == l)
        pcat[:, j] /= np.sum(pcat[:, j])
    # Initialize mus
    mus = np.column_stack((
    np.random.uniform(np.nanmin(spatial.values[:,0]), np.nanmax(spatial.values[:,0]), size=M),
    np.random.uniform(np.nanmin(spatial.values[:,1]), np.nanmax(spatial.values[:,1]), size=M)))
    mncat = np.zeros((maxcat, q, M))
    for l in range(q):
        mncat[0:ncats[l], l, :] = 1 / ncats[l]
    mu = np.zeros((M, p))
    tau1 = np.zeros(p)
    tau2 = np.zeros(p)
    for l in range(p):
        mu[:, l] = np.nanmean(np.array([Xnorm]), axis=1)
        tau1[l] = 1 / np.nanvar(np.array([Xnorm]), axis=1, ddof=1)
        tau2[l] = 1 / np.nanvar(np.array([Xnorm]), axis=1, ddof=1)
    taus1 = 1 / np.mean(np.cov(spatial[~np.any(np.isnan(spatial), axis=1)], rowvar=False))
    taus2 = 1 / np.mean(np.cov(spatial[~np.any(np.isnan(spatial), axis=1)], rowvar=False))
    theta = np.nanmean(mu, axis=0)
    thetas = np.nanmean(mus, axis=0)
    df = np.ones(q)
    probs = np.ones(M)
    D = 100
    # Initialize storage for results
    keep_df = np.zeros((iters, q))
    keep_D = np.zeros(iters)
    keep_sd1 = np.zeros((iters, p))
    keep_sd2 = np.zeros((iters, p))
    keep_theta = np.zeros((iters, p))
    keep_sds = np.zeros((iters, 2))
    # Handle missing values
    missing = np.where(miss)[0]
    missing_s = np.where(np.isnan(spatial.values[:, 0]))[0]
    missing_t = np.where(t1 != t2)[0]
    n_missing_s = len(missing_s)
    n_missing_t = len(missing_t)
    keep_s = None
    keep_t = None
    p_equal = 0
    if show_pb:
        progress_bar = tqdm(total=iters, desc="Processing")
    # Add main loop for MCMC iterations
    for i in range(iters):
        # Missing Spatial location
        sss = 1 / np.sqrt(taus1)
        if use_space and len(missing_s) > 0:
            j = missing_s
            spatial.values[j, 0] = np.random.normal(mu[g[j], 0], sss)
            spatial.values[j, 1] = np.random.normal(mu[g[j], 1], sss)
        # Consored times    
        if use_time and len(missing_t) > 0:
            j = missing_t
            new = np.zeros(n_missing_t)
            mu_value = mu[g[j]]
            sigma = 1 / np.sqrt(tau1[0])
            t1_values = t1[j].values
            t2_values = t2[j].values
            for k in range(n_missing_t):
                new[k] = rtruncnorm(1, mu_value[k], sigma, t1_values[k], t2_values[k])
            new[np.isinf(new)] = np.nan
            Xnorm[j] = np.where(np.isnan(new), Xnorm[j], new)
        # Missing cat values
        if use_cats:
            for l in range(q):
                t = np.arange(ncats[l])
                for j in range(n):
                    if miss_cat[j, l]:
                        Xcat[j, l] = np.random.choice(t,1,replace=False,p=mncat[t,l,g[j]])  # Adjusting for zero-indexing
        if p > 1:
            for l in range(1, p):
                j = np.where(miss_norm[:, l])[0]
                if len(j) > 0:
                    Xnorm[j, l] = np.random.normal(mu[g[j], l], 1 / np.sqrt(tau1[l]), size=1)
        # Missing group labels
        oldg = g.copy()
        for j in missing:
            R = np.log(probs)
            if use_space:
                R = R - 0.5 * taus1 * ((spatial.values[j, 0] - mus[:,0])**2 + (spatial.values[j, 1] - mus[:, 1])**2)
            if use_time:
                R = R - 0.5 * tau1[0] * (Xnorm[j] - mu[:,0])**2
            ppp = np.exp(R - np.nanmax(R))  # Вычисляем ppp по формуле
            ppp[np.isnan(ppp)] = 0  # Заменяем NaN на 0
            if len(ppp.shape) > 1:
                ppp.shape = -1
            cang = g[j]   # Задаем начальное значение cang
            if np.sum(ppp) > 0:
                cang = np.random.choice(np.arange(M),1,replace=False,p=ppp/np.sum(ppp))
            R = 0
            if use_cats:
                for ccc in range(q):
                    R = R + np.log(mncat[Xcat[j,q-1],0,cang])-np.log(mncat[Xcat[j,q-1],0,g[j]])
            if np.random.uniform(0, 1) < np.max(np.exp(R)):
                g[j] = cang
        ccc = get_counts(g, M)  # Count occurrences in groups
        probs = np.random.dirichlet(D + ccc, 1)
        canD = np.exp(np.random.normal(np.log(D), 0.05,size=1))  # Proposal for D
        R = (norm.logpdf(np.log(canD), 0, 10) - norm.logpdf(np.log(D), 0, 10) + ddir2(probs, canD) - ddir2(probs, D))
        if not np.isnan(np.exp(R)):
            if np.random.uniform(0, 1) < np.exp(R):
                D = canD
        # Update spatial model
        if use_space:
            VVV = taus1 * count_g(g, M) + taus2
            MMM = taus1 * sumbyg(spatial.values[:, 0], g, M) + taus2 * thetas[0]
            mus[:, 0] = np.random.normal(MMM / VVV, 1 / np.sqrt(VVV), M)
            MMM = taus1 * sumbyg(spatial.values[:, 1], g, M) + taus2 * thetas[1]
            mus[:, 1] = np.random.normal(MMM / VVV, 1 / np.sqrt(VVV), M)
            VVV = taus2 * M + 1e-04
            MMM = taus2 * np.sum(mus, axis=0) + np.array([-76.6, 39.3]) * 1e-04
            thetas = np.random.normal(MMM / VVV, 1 / np.sqrt(VVV), 2)
            taus1 = np.random.gamma(2*n/2+0.1,1/(np.sum((spatial.values-mus[g,:])**2)/2+0.1),1)
            SS = np.sum((mus[:, 0] - thetas[0]) ** 2 + (mus[:, 1] - thetas[1]) ** 2)
            taus2 = np.random.gamma(2*M/2+0.1,1/(SS/2+0.1),1)
        # Continuous predictors
        if use_time:
            for l in range(p):
                VVV = tau1[l] * count_g(g, M) + tau2[l]
                MMM = tau1[l] * sumbyg(np.array([Xnorm])[l], g, M) + tau2[l] * theta[l]
                mu[:, l] = np.random.normal(MMM / VVV, 1 / np.sqrt(VVV), M)
                VVV = tau2[l] * M + 1e-04
                MMM = tau2[l] * np.sum(mu[:, l]) + 0 * 1e-04
                theta[l] = np.random.normal(MMM / VVV, 1 / np.sqrt(VVV), size=1)
                tau1[l] = np.random.gamma(n / 2 + 0.1, 1 / (np.sum((np.array([Xnorm])[l] - mu[g, l]) ** 2) / 2 + 0.1), size=1)
                tau2[l] = np.random.gamma(M / 2 + 0.1, 1 / (np.sum((mu[:, l] - theta[l]) ** 2) / 2 + 0.1), size=1)
        # Updates if using categorical data
        if use_cats:
            eps = 1e-04
            for l in range(q):
                t = np.arange(ncats[l])
                for j in range(M):  # For clusters
                    DDD = df[l] * pcat[t, l] + get_counts(Xcat[g == j, l], ncats[l])
                    duh = np.random.dirichlet(DDD[t], size=1)
                    duh = np.where(duh > 1 - eps, 1 - eps, duh)
                    duh = np.where(duh < eps, eps, duh)
                    mncat[t,l,j] = duh / np.sum(duh)
                candf = np.exp(np.random.normal(np.log(df[l]), 0.1,size=1))
                R = norm.pdf(np.log(candf), 0, 10) - norm.pdf(np.log(df[l]), 0, 10)
                for j in range(M):  # For clusters
                    R = ddir(mncat[t,l,j],candf,pcat[t,l])-ddir(mncat[t,l,j],df[l],pcat[t,l])
                if not np.isnan(np.exp(R)):
                    if np.random.uniform(0, 1) < np.exp(R):
                        df[l] = candf
        # Keep results
        keep_df[i] = df
        keep_sd1[i] = 1 / np.sqrt(tau1)
        keep_sd2[i] = 1 / np.sqrt(tau2)
        keep_sds[i] = (1 / np.sqrt(np.array([taus1, taus2]))).reshape(-1)
        keep_D[i] = D
        if i > burn:
            ddd = np.subtract.outer(g, g)  # Pairwise differences
            p_equal = p_equal + (ddd == 0) / (iters - burn)
        if show_pb:
            progress_bar.update(1)
    if show_pb:
        progress_bar.close()        
    result = {'p_equal': p_equal,
        'D': keep_D,
        'df': keep_df,
        'sd1': keep_sd1,
        'sd2': keep_sd2,
        'sds': keep_sds,
        'theta': keep_theta,
        's_miss': keep_s,
        't_censored': keep_t,
        'missing_s': missing_s,
        'missing_t': missing_t,
        'crimeID': crimeID.values}
    return result


def crimeClust_Hier(crimedata, varlist, estimateBF,
                     linkage_method='average', **kwargs):
    """
    Agglomerative Hierarchical Crime Series Clustering.

    Run hierarchical clustering on a set of crimes using the log Bayes Factor as the similarity metric

    Parameters
    ----------
    crimedata : [DataFrame] dataframe of crime incidents. Must contain a column named crimeID.
    varlist : [dict] a list of the variable names columns of crimedata used to create evidence variables with compareСrimes.
    estimateBF : [function] function to estimate the log bayes factor from evidence variables.
    linkage_method : [str] the type of linkage for hierarchical clustering: 'average' - uses the average bayes factor, 
                'single' - uses the largest bayes factor (most similar), 'complete' - uses the smallest bayes factor (least similar), 
                'weighted' - a balanced group average (also called WPGMA), 'centroid' - unweighted pair group method using centroids 
                (also called UPGMC), 'median' - the centroid of the new cluster is accepted as the average value of the centrides of 
                two combined clusters (WPGMC algorithm), 'ward' - uses the Ward variance minimization algorithm.
    **kwargs : arguments to pass to the function compareСrimes.

    Returns
    ----------
    The hierarchical clustering encoded as a linkage matrix based on log Bayes Factor.
    Values:
    hc : a linkage matrix.
    offsets : maximum of the log bayes factor.
    crimesID : a list of crimeID used to return a linkage matrix based on log Bayes Factor.
    """
    # Убедитесь, что метод связывания валидный
    linkage_method = linkage_method.lower()
    valid_methods = ['average', 'single', 'complete']
    if linkage_method not in valid_methods:
        raise ValueError(f"Invalid linkage method. Choose from {valid_methods}")
    crimedata = crimedata.reset_index(drop=True)
    crimeIDs = crimedata['crimeID'].unique()
    all_pairs = pd.DataFrame(
        [(crimeIDs[i], crimeIDs[j]) for i in range(len(crimeIDs)) 
         for j in range(i + 1, len(crimeIDs))],
        columns=['crimeID1', 'crimeID2']
    )
    # Сравнение преступлений
    A = compareСrimes(all_pairs, crimedata, varlist, **kwargs)
    bf = estimateBF(A)
    # Выполнение агломеративной иерархической кластеризации
    d2 = -bf  # Вычисляем расстояние  # Преобразуем в матрицу расстояний
    offset = np.ceil(bf.max(axis=0))  # Смещение для корректировки расстояний
    d2 += offset
    # Выполнение кластеризации
    hc = linkage(d2, method=linkage_method)
    HC={'hc': hc, 'offset': offset, 'crimeID': crimeIDs}
    return HC


def crimeCount(seriesdata):
    """
    Return length of each crime series and distribution of crime series length in count of offenders.
    
    Parameters
    ----------
    seriesdata : [DataFrame] crime series data, generated from makeSeriesData with offender IDs, crime IDs and the event datetime.
    
    Returns
    ----------
    DataFrame containing two columns:
    'Count_Crimes' : length of each crime series,
    'Count_Offenders' : distribution of crime series length in count of offenders.
    """
    nCrimes = seriesdata['CS'].value_counts().rename_axis('CS').reset_index(name='Count')
    nCrimes = nCrimes['Count'].value_counts(sort=False).rename_axis('Count_Crimes').reset_index(name='Count_Offenders')
    return nCrimes


def crimeLink(crimedata, varlist, estimateBF, sort=True):
    """
    Links between crime pairs based on log Bayes Factor.

    Make a dataframe of links between crime pairs based on log Bayes Factor

    Parameters
    ----------
    crimedata : [DataFrame] dataframe of crime incidents. Must contain a column named crimeID.
    varlist : [dict] a list of the variable names columns of crimedata used to create evidence variables with compareСrimes.
    estimateBF : [function] function to estimate the log bayes factor from evidence variables.
    sort : [bool, default True] sort data of columnes based on log Bayes Factor in descending (sort=True, default) or ascending order.

    Returns
    ----------
    A dataframe of links between crime pairs based on log Bayes Factor.
    """
    crimeIDs=set(crimedata['crimeID'])
    allPairs=pd.DataFrame(list(combinations(crimeIDs, 2)),columns=['i1', 'i2'])
    A=compareСrimes(allPairs,crimedata,varlist=varlist)
    bf=estimateBF(A)
    d2=-bf
    offset=math.ceil(max(bf))
    d2=d2+offset
    Hclust=pd.DataFrame({'i1':allPairs['i1'],'i2':allPairs['i2'],'dist':d2})
    Hclust=Hclust.sort_values('dist', ascending=sort)
    return Hclust


def crimeLink_Clust_Hier(crimedata, predGB, linkage):
    """
    Agglomerative Hierarchical Crime Series Clustering for crimes linkage
    
    Run hierarchical clustering on a set of crimes using the probabilities for linkage of crimes pairs as the similarity metric
    
    Parameters
    ----------
    crimedata : [DataFrame] dataframe of crime incidents. Must contain a column named crimeID.
    predGB : [DataFrame] dataframe of links between crime pairs based on probabilities produced from predictGB.
    linkage : [str] the type of linkage for hierarchical clustering: 'average' - uses the average probabilities, 'single' - uses the largest probabilities (most similar), 'complete' - uses the smallest probabilities (least similar).
    
    Returns
    ----------
    The hierarchical clustering encoded as a linkage matrix based on probabilities for linkage of crime pairs.
    """
    predGB=predGB.copy()
    df1 = predGB.set_index('i1')
    df1.index.names = [None]
    df2 = predGB.set_index('i2')
    df2.index.names = [None]
    df2.rename(columns = {'i1':'i2'}, inplace = True)
    df3=pd.concat([df1,df2])
    lC=list(crimedata['crimeID'])
    for df7 in predGB:
        df7=pd.DataFrame()
        for i in range(len(lC)):
            df4=pd.DataFrame({'crime':lC,'link':lC[i]})
            df5=df3.loc[lC[i]]
            df6 = pd.merge(df4,df5,left_on=['crime'], right_on=['i2'], how='left')
            df6.drop(['crime','link_x','i2'], axis='columns', inplace=True)
            df6.rename(columns = {'link_y':lC[i]}, inplace = True)
            df7=pd.concat([df7,df6], axis=1).fillna(value=1)
        df7.index=(lC)
    linkage_matrix = hierarchy.linkage(df7, method=linkage)
    global prob; prob = predGB['link']
    return linkage_matrix


def datapreprocess(data):
    """
    Preliminary preparation of data for crime series linkage.

    The function prepares the data on the time of the crime for the analysis of crime series linkage, combining the date and time into one column of the format datetime as well as convert categorical data into numerical data

    Parameters
    ----------
    data : [DataFrame] dataframe of crime incidents, containing the date and time of the crime in different columns and also possibly contains categorical data.

    Returns
    ----------
    Dataframe in which date and time are combined into one column of the format datetime and categorical data converted into numerical data.
    """
    series = pd.Series(data['DT.FROM']+" "+data['a.DT.FROM'], index=data.index)
    series2 = pd.Series(data['DT.TO']+" "+data['a.DT.TO'], index=data.index)
    datetime_series = pd.to_datetime(series) 
    datetime_series2 = pd.to_datetime(series2)
    data['DT.FROM']=datetime_series
    data['DT.TO']=datetime_series2
    data=data.drop(['a.DT.FROM'], axis = 1)
    data=data.drop(['a.DT.TO'], axis = 1)
    return data


def difftime(time1, time2, tz=None, units='auto'):
    """
    Time intervals creation, printing, and some arithmetic
    
    Calculates a difference of two date/time objects.
    
    Parameters
    ----------
    time1 : object of class datetime.datetime.
    time2 : object of class datetime.datetime.
    tz : [int] an optional time zone specification to be used for the conversion (default: tz=None).
    units : [str] units in which the results are desiredю Available values: 'auto' (return seconds),
                  'secs', 'mins', 'hours', 'days', 'weeks' (default: units='auto').
    
    Returns
    ----------
    an object of a difference of two date/time object with an attribute indicating the units.
    """
    # Преобразование строковых представлений дат в datetime
    if tz is None:
        time1 = pd.to_datetime(time1)
        time2 = pd.to_datetime(time2)
    else:
        time1 = pd.todatetime(time1).tzlocalize(tz)
        time2 = pd.todatetime(time2).tzlocalize(tz)

    # Вычисляем разницу
    z = (time1 - time2).dt.total_seconds()  # Разница в секундах
    # Определение единиц
    if units == 'auto':
        if np.isnan(z):
            units = 'secs'
        else:
            zz = abs(z)
            if not np.isfinite(zz) or zz < 60:
                units = 'secs'
            elif zz < 3600:
                units = 'mins'
            elif zz < 86400:
                units = 'hours'
            else:
                units = 'days'
    # Возвращаем разницу в выбранных единицах
    if units == 'secs':
        return z
    elif units == 'mins':
        return z / 60
    elif units == 'hours':
        return z / 3600
    elif units == 'days':
        return z / 86400
    elif units == 'weeks':
        return z / (86400 * 7)
    else:
        raise ValueError("Unsupported units: choose from 'secs', 'mins', 'hours', 'days', or 'weeks'.")


def expabsdiff(X, Y):
    """
    Expected absolute difference between the two dates of the crime pairs, 
    expressed in days of the week.

    Calculates the expected absolute difference of two uniform two dates 
    of the crime pairs, expressed in days of the week

    Parameters
    ----------
    X : [float] a list of two values - Julian minimum day of the week and 
                Julian maximum day of the week of the first crime in the pair.
    Y : [float] a list of two values - Julian minimum day of the week and 
                Julian maximum day of the week of the second crime in the pair.

    Returns
    ----------
    The expected absolute difference between the two dates of the crime pairs, 
    expressed in days of the week.
    """
    if X[1] < X[0]:
        raise ValueError("X[1] < X[0]")
    if Y[1] < Y[0]:
        raise ValueError("Y[1] < Y[0]")
    if X[0] <= Y[0]:
        # set Sx to have minimum lower bound
        Sx = X
        Sy = Y
    else:
        Sx = Y
        Sy = X
    # Scenario 1 (no overlap)
    if Sx[1] <= Sy[0]:
        return np.mean(Sy) - np.mean(Sx)
    bks = np.sort(Sx + Sy)
    sz = np.diff(bks)
    mids = bks[1:] - sz / 2
    # Scenario 2 (partial overlap)
    if Sx[1] <= Sy[1]:
        px = sz * np.array([1, 1, 0]) / np.diff(Sx)
        py = sz * np.array([0, 1, 1]) / np.diff(Sy)
        return ((mids[1] - mids[0]) * px[0] * py[1] +
            (mids[2] - mids[0]) * px[0] * py[2] +
            (sz[1] / 3) * px[1] * py[1] +
            (mids[2] - mids[1]) * px[1] * py[2])
    # Scenario 3 (Y completely overlaps X)
    if Sx[1] > Sy[1]:
        px = sz * np.array([1, 1, 1]) / np.diff(Sx)
        return ((mids[1] - mids[0]) * px[0] +
            (sz[1] / 3) * px[1] +
            (mids[2] - mids[1]) * px[2])


def expabsdiff_circ(X, Y, mod=24, n=2000, method="convolution"):
    """
    Expected absolute difference between the two dates of the crime pairs, 
    expressed in time of day (in hours) or day of week (in days).

    Estimates the expected circular temporal distance between crimes using 
    discrete FFT or numerical integration

    Parameters
    ----------
    X : [float] a list of two values - min and min+length the two dates of the first crime of the pair, 
                expressed in time of day (in hours) or day of week (in days). X[0] must be >= 0 and X[1] >= X[1]. 
                It is possible that X[1] can be > mod.
    Y : [float] a list of two values - min and min+length the two dates of the second crime of the pair, 
                expressed in time of day (in hours) or day of week (in days). X[0] must be >= 0 and X[1] >= X[1]. 
                It is possible that X[1] can be > mod.
    mod : [int] the period of time. E.g., mod=24 for time of day (in hours), mod=7 for day of week (in days).
    n : [int] number of bins for discretization of continuous time domain. E.g., there is 1440 min/day, 
                so n = 2000 should give close to minute resolution.
    method : [str] use convolution (method='convolution', default) or monte carlo integration (method='numerical').

    Returns
    ----------
    The expected absolute difference between the two dates of the crime pairs, expressed in time of day (in hours) 
    or day of week (in days).
    """
    def next_n(n):
        """Helper function to find the next even number."""
        return n + 1 if n % 2 else n
    if X[0] < 0 or Y[0] < 0:
        raise ValueError("X and Y must be > 0")
    if np.diff(X) >= mod or np.diff(Y) >= mod:
        return mod / 4  # uniform over mod
    if np.diff(X) == 0:
        return get_d(X[0], Y, mod=mod)
    if np.diff(Y) == 0:
        return get_d(Y[0], X, mod=mod)
    if method == "convolution":
        while((n % 2) !=0):
            n=n = next_n(n)  # Ensure n is even
        theta = np.linspace(0, mod, n + 1)
        delta = np.diff(theta[:2])[0]
        x=np.diff(uniform.cdf(theta,X[0],X[1]-X[0]))+np.diff(uniform.cdf(np.array(theta)+mod,X[0],X[1]-X[0]))
        y=np.diff(uniform.cdf(theta,Y[0],Y[1]-Y[0]))+np.diff(uniform.cdf(np.array(theta)+mod,Y[0],Y[1]-Y[0]))
        conv = np.fft.ifft(np.fft.fft(x) * np.conjugate(np.fft.fft(y)))
        conv = np.round(conv.real, decimals=10)
        tt = np.delete(np.where(theta <= mod / 2, theta, mod - theta),n)
        d = np.sum(tt * conv)
        return d
    if method == "numerical":
        if np.diff(Y) < np.diff(X):
            tt = np.linspace(Y[0], Y[1], n)
            d = np.mean(get_d(tt, X, mod=mod))
        else:
            tt = np.linspace(X[0], X[1], n)
            d = np.mean(get_d(tt, Y, mod=mod))
        return d


def GBC(X, Y, start, end, step, n_splits=5, learning_rate=0.2, **kwargs):
    """
    Gradient Boosting for classification of linked and unlinked crime pairs.

    GBC builds an additive model with most optimization of arbitrary differentiable loss functions for classification of linked and unlinked crime pairs

    Parameters
    ----------
    X : [array-like, sparse matrix of shape] training dataframe of crime incidents with predictors.
    Y : [array-like of shape] target values. Labels must correspond to training dataframe.
    start : [int] the minimum number of boosting stages to performance.
    end : [int] the maximum number of boosting stages to performance.
    step : [int] step to select the most optimal number of boosting stages to performance.
    n_splits : [int] number of folds, but must be at least 2 (default=5).
    learning_rate : [float] learning rate shrinks the contribution of each tree by learning_rate (default=0.2).
    **kwargs : arguments to pass to the function GradientBoostingClassifier sklearn.

    Returns
    ----------
    model of Gradient Boosting for classification for linkage crimes pairs.
    """
    def gb_Gridsearch(data_features: pd.DataFrame,
                       data_target: pd.DataFrame,
                       n_estimators: List[int]) -> GridSearchCV:
        classifier = GradientBoostingClassifier()
        cross_validation = KFold(n_splits=n_splits, shuffle=True)
        grid_params = {'n_estimators': n_estimators}
        gs = GridSearchCV(classifier, grid_params, scoring='roc_auc', cv=cross_validation)
        gs.fit(data_features, data_target)
        return gs
    gb_gs = gb_Gridsearch(X, Y, list(range(start, end, step)))
    n=list(gb_gs.best_params_.values())
    gb = GradientBoostingClassifier(n_estimators=n[0], learning_rate=learning_rate, verbose=False, random_state=241, **kwargs).fit(X, Y)
    return gb


def get_d(y, X, mod=24):
    """
    Expected absolute distance between the two dates of the crime pairs, expressed 
    in time of day (in hours) or day of week (in days).

    Parameters
    ----------
    y : [tuple: float] a vector of times in [0, mod)
    X : [tuple: float] a list of two values - min and min+length the two dates of the 
                       crime of the pair, expressed in time of day (in hours) or day of week (in days). 
                       X[0] must be >= 0 and X[1] >= X[0]. It is possible that X[1] can be > mod. I.e., do not do X%mod.
    mod : [int] the period of time. E.g., mod=24 for time of day (in hours), mod=7 for day of week (in days).

    Returns
    ----------
    The expected absolute difference between the two dates of the crime pairs, 
    expressed in time of day (in hours) or day of week (in days).
    """
    if X[0] > mod or X[0] < 0:
        raise ValueError("Minimum X[1] not within limits [0, mod)")
    if X[1] < X[0]:
        raise ValueError("X[2] must be >= X[1]")
    y = (y - X[0]) % mod
    B = X[1] - X[0]  # length of interval
    if B == 0:
        return mod / 2 - abs(mod / 2 - abs(y))  # For |X| = 0
    if B >= mod:
        return np.full(len(y), mod / 4)  # For long intervals
    D = np.zeros(1)
    if np.diff(X)[0] >= mod / 2:
        K = mod - B / 2 - (mod / 2) ** 2 / B
        u = y - mod / 2
        if (y <= B - mod / 2):
            D[0] = y * (1 - mod / B) + K
        elif (y > B - mod / 2) & (y <= mod / 2):
            D[0] = (y - B / 2) ** 2 / B + B / 4
        elif (y > mod / 2) & (y <= B):
            D[0] = (B - y) * (1 - mod / B) + K
        elif (y > B):
            D[0] = mod / 2 - B / 4 - ((u - B/2)) ** 2 / B
    else:
        u = y - B / 2
        if (y < B):
            D[0] = u ** 2 / B + B / 4
        elif (y >= B) & (y <= mod / 2):
            D[0] = u
        elif (y > mod / 2) & (y <= B + mod / 2):
            D[0] = mod / 2 - ((y - mod / 2) ** 2 + (B - y + mod / 2) ** 2) / (2 * B)
        elif (y > B + mod / 2):
            D[0] = mod - u
    return D


def getBF(x, y, weights=None, breaks=None, df=5):
    """
    Estimates the bayes factor for continous and categorical predictors.

    Continous predictors are first binned, then estimates shrunk towards zero

    Parameters
    ----------
    x : [array-like of shape] predictor vector (continuous or categorical/factors).
    y : [array-like of shape] binary vector indicating linkage (1 = linked, 0 = unlinked).
    weights : [array-like of shape] vector of observation weights or the column name in data 
            that corresponds to the weights (default weights=None).
    breaks : [int, default None] set of break point for continuous predictors or NULL for categorical or discrete.
    df : [int, default 5] the effective degrees of freedom for the cetegorical density estimates.

    Returns
    ----------
    The set containing: dataframe the levels/categories with estimated Bayes factor, 
    'breaks' - set of break point for continuous predictors, 'a' - list of markers of the linked and 
    unlinked crime pairs, 'df' - the effective degrees of freedom, 'df2' - the effective degrees of freedom 
    for linked and unlinked crime pairs.
    
    Notes
    ----------
    This adds pseudo counts to each bin count to give df effective degrees of freedom. 
    Must have all possible factor levels and must be of factor class.
    Give linked and unlinked a different prior according to sample size.
    """
    def replace_na(x, r=0):
        return np.where(np.isnan(x), r, x)  # Замена NA на значение r

    if isinstance(x, pd.DataFrame):
        raise ValueError("x must be a vector, not a dataframe")
    linked = (y.astype(int) == 1)  # Преобразование y в логическое значение (1 - приведенное к True)

    # Установка весов
    if weights is None:
        weights = np.ones_like(linked, dtype=float)
    var_type = x.dtype  # Получаем тип переменной
    if var_type == 'float64':  # Проверка на числовые типы
        n_bks = len(breaks)
        xbins = pd.cut(x, bins=breaks, duplicates='drop')  # Разделяем x на интервалы
        # Преобразуем в категориальный формат
        # Накопление весов для связанных и не связанных значений
        tlinked = pd.DataFrame(weights.loc[linked.values]).groupby(xbins.loc[linked.values]).sum().reindex(xbins.cat.categories, fill_value=0)
        tunlinked = pd.DataFrame(weights.loc[~linked.values]).groupby(xbins.loc[~linked.values]).sum().reindex(xbins.cat.categories, fill_value=0)
        from_to = pd.DataFrame({
            'from': breaks[:-1],
            'to': breaks[1:]})
        from_to = from_to.drop(from_to[from_to['from'] == from_to['to']].index).reset_index(drop=True)
        # Создаем итоговый DataFrame
        E = pd.DataFrame({
            'value': tlinked.index,
            'N.linked': replace_na(tlinked.reset_index()['wt']),
            'N.unlinked': replace_na(tunlinked.reset_index()['wt'])})
        E = pd.concat([from_to,E],axis=1)
    elif var_type == 'category':  # Для факторных и категориальных переменных
        x = pd.Categorical(x)
        tlinked = pd.DataFrame(weights.loc[linked.values]).groupby(x[np.where(y==1)[0]]).sum().reindex(x.categories, fill_value=0)
        tunlinked = pd.DataFrame(weights.loc[~linked.values]).groupby(x[np.where(y!=1)[0]]).sum().reindex(x.categories, fill_value=0)
        # Создаем итоговый DataFrame
        E = pd.DataFrame({
            'value': tlinked.index,
            'N.linked': replace_na(tlinked.reset_index()['wt']),
            'N.unlinked': replace_na(tunlinked.reset_index()['wt'])})
    else:
        raise ValueError("Unsupported variable type")
    def df2a(df, k, N):
        return (N / k) * ((k - df) / (df - 1)) # Формула для расчета a linked
    def a2df(a, k, N):
        return k * (N + a) / (N + k * a)  
    def get_p(N, a):
        return (N + a) / np.sum(N + a)  # Формула для расчета вероятностей
    nlevs = len(E)
    df = min(np.append(df, nlevs - 1e-8))  # Корректировка df
    a_linked = df2a(df, k=nlevs, N=np.sum(E['N.linked']))
    a_unlinked = df2a(df, k=nlevs, N=np.sum(E['N.unlinked']))
    E['p.linked'] = get_p(E['N.linked'].values, a_linked)
    E['p.unlinked'] = get_p(E['N.unlinked'].values, a_unlinked)
    E['BF'] = E['p.linked'] / E['p.unlinked']
    E['BF'].replace({np.nan: 1}, inplace=True) # Замена NaN на 1
    return E


def getCrimes(offenderID, crimedata, offenderTable):
    """
    Generate a list of crimes for a specific offender.

    Parameters
    ----------
    offenderID : [list: str] an offender ID that is in offenderTable.
    crimedata : [DataFrame] dataframe of crime incidents. Must contain a column named crimeID.
    offenderTable : [DataFrame] offender table that indicates the offender(s) responsible for solved crimes.
                    offenderTable must have columns named - offenderID and crimeID.

    Returns
    ----------
    The subset of crimes in crimedata that are attributable to the offender named offenderID.

    Example
    ----------
    getCrimes(['Prodan'], Crimes, Offenders)
    """
    # Получаем crimeID для заданных offenderID
    cid = offenderTable['crimeID'][offenderTable['offenderID'].isin(offenderID)]
    # Фильтруем crimedata для получения тех преступлений, которые соответствуют crimeID
    crimes = crimedata[crimedata['crimeID'].isin(cid)]
    return crimes


def getCrimeSeries(offender_id, offender_table, restrict=None, show_pb=False):
    """
    Generate a list of offenders and their associated crime series.

    Parameters
    ----------
    offender_id : [list] a list of offender
    offenderTable : [DataFrame] offender table that indicates the offender(s) responsible for solved crimes. 
                    offenderTable must have columns named - offenderID and crimeID.
    restrict : [list] if vector of 'crimeID', then only include those crimeIDs in 'offender_table'. 
                    If 'None', then return all crimes for offender (default: restrict=None).
    show_pb : [bool] show the progress bar (default: show_pb=False).

    Returns
    ----------
    List of offenders with their associated crime series.

    Example
    ----------
    get_crime_series([''Prodan','Popkov''], Offenders)
    """
    offender_id = list(set(offender_id))
    n = len(offender_id)
    cs = []
    valid_crime_ids = True
    if restrict is not None:  # Предварительно вычислить допустимые идентификаторы преступлений
        valid_crime_ids = (offender_table['crimeID'].isin(restrict))

    valid_offender_ids = (offender_table['offenderID'].isin(offender_id))  # Предварительно вычислить действительные идентификаторы нарушителей
    valid = valid_crime_ids & valid_offender_ids
    offender_table = offender_table[valid].drop_duplicates()  # Использовать только уникальные допустимые записи из таблицы нарушителей для ускорения расчётов
    if show_pb:
        pb = tqdm(total=n)
    for i in range(n):
        oid = offender_id[i]
        ind = offender_table.index[offender_table['offenderID'].isin([oid])]
        cid = [str(x) for x in offender_table.loc[ind, 'crimeID']]
        cs.append({'offenderID': oid, 'crimeID': cid})
        if show_pb: pb.update(1)
    if show_pb:
        pb.close()
    return cs if n > 1 else cs[0]


def getCriminals(crimeID, offender_table):
    """
    List of the offenders responsible for a set of solved crimes.

    Generates the IDs of criminals responsible for a set of solved crimes using the information in offenderTable

    Parameters
    ----------
    crimeID : [list: str] crime IDs of solved crimes.
    offenderTable : [DataFrame] offender table that indicates the offender(s) responsible for solved crimes. 
                                offenderTable must have columns named - offenderID and crimeID.

    Returns
    ----------
    List of offenderIDs responsible for crimes labeled crimeID.

    Example
    ----------
    getCriminals(['Crime100'], Offenders)
    """
    crimeID = [str(x) for x in crimeID]
    ind = offender_table['crimeID'].isin(crimeID)
    offenderID = sorted(set([str(x) for x in offender_table.loc[ind, 'offenderID']]))
    return offenderID


def getROC(f, y):
    """
    Cacluate ROC metrics for interpret the results of classification.

    Orders scores from largest to smallest and evaluates performance for each value. 
    This assumes an analyst will order the predicted scores and start investigating the linkage claim in this order

    Parameters
    ----------
    f : [array-like of shape] predicted score for linkage cases.
    y : [array-like of shape] target scores: linked=1, unlinked=0.

    Returns
    ----------
    Dataframe of evaluation metrics:
    'FPR' - false positive rate - proportion of unlinked pairs that are incorrectly assessed as linked,
    'TPR' - true positive rate; recall; hit rate - proportion of all linked pairs that are correctly assessed as linked,
    'PPV' - positive predictive value; precision - proportion of all pairs that are predicted linked and truely are linked,
    'Total' - the number of cases predicted to be linked,
    'TotalRate' - the proportion of cases predicted to be linked,
    'threshold' - the score threshold that produces the results.

    Examples
    ----------
    nb=predictnaiveBayes(NB,test[test.columns[3:-1]],var)
    v=getROC(nb,test['Y'])
    """
    f = pd.Series(f)  # Преобразуем в pandas Series
    y = pd.Series(y)  # Преобразуем в pandas Series
    # Индексы для сортировки по убыванию вероятностей
    ord = f.argsort()[::-1]  # Сортировка с помощью индексов
    f = f.iloc[ord].reset_index(drop=True)
    y = y.iloc[ord].reset_index(drop=True)
    # Определяем уникальные значения
    uniq = ~f.duplicated(keep='last')  # Получаем уникальные значения с учетом дубликатов
    n_pos = np.sum(y == 1)
    n_neg = np.sum(y == 0)
    # Считаем TP и FP
    TP = np.cumsum(y == 1)[uniq]
    FP = np.cumsum(y == 0)[uniq]
    # Общее количество проверяемых случаев
    Total = np.arange(len(f))[uniq]

    # Вычисляем TPR, FPR и PPV
    TPR = TP / n_pos if n_pos > 0 else np.zeros_like(TP)
    FPR = FP / n_neg if n_neg > 0 else np.zeros_like(FP)
    PPV = TP / Total if Total.size > 0 else np.zeros_like(TP)
    # Создаем датафрейм с результатами
    result_df = pd.DataFrame({
        'FPR': FPR,
        'TPR': TPR,
        'PPV': PPV,
        'Total': Total,
        'TotalRate': Total / len(f),
        'threshold': f[uniq].values  # Только уникальные пороги
    })
    return result_df


def graphDataFrame(edges, directed=True, vertices=None, use_vids=False):
    """
    Generates a graph from one or two dataframes.

    Parameters
    ----------
    edges : [DataFrame] pandas DataFrame containing edges and metadata. The first
      two columns of this DataFrame contain the source and target vertices
      for each edge. These indicate the vertex *names* rather than ids
      unless 'use_vids' is True and these are nonnegative integers.
    directed : [bool] setting whether the graph is directed
    vertices : [DataFrame] None (default) or pandas DataFrame containing vertex
      metadata. The first column must contain the unique ids of the
      vertices and will be set as attribute 'name'. Although vertex names
      are usually strings, they can be any hashable object. All other
      columns will be added as vertex attributes by column name.
    use_vids : [DataFrame] whether to interpret the first two columns of the 'edges'
      argument as vertex ids (0-based integers) instead of vertex names.
      If this argument is set to True and the first two columns of 'edges'
      are not integers, an error is thrown.

    Returns
    ----------
    The graph

    Notes
    ----------
    Vertex names in either the 'edges' or 'vertices' arguments that are set
    to NaN (not a number) will be set to the string "NA". That might lead
    to unexpected behaviour: fill your NaNs with values before calling this
    function to mitigate.
    """
    if edges.shape[1] < 2:
        raise ValueError("the data frame should contain at least two columns")
    if use_vids:
        if str(edges.dtypes[0]).startswith("int") and str(
            edges.dtypes[1]
        ).startswith("int"):
            names_edges = None
        else:
            raise TypeError("vertex ids must be 0-based integers")
    else:
        if edges.iloc[:, :2].isna().values.any():
            warn("In 'edges' NA elements were replaced with string \"NA\"")
            edges = edges.copy()
            edges.iloc[:, :2].fillna("NA", inplace=True)
        names_edges = np.unique(edges.values[:, :2])
    if (vertices is not None) and vertices.iloc[:, 0].isna().values.any():
        warn(
            "In the first column of 'vertices' NA elements were replaced "
            + 'with string "NA"'
        )
        vertices = vertices.copy()
        vertices.iloc[:, 0].fillna("NA", inplace=True)
    if vertices is None:
        names = names_edges
    else:
        if vertices.shape[1] < 1:
            raise ValueError("vertices has no columns")
        names_vertices = vertices.iloc[:, 0]
        if names_vertices.duplicated().any():
            raise ValueError("Vertex names must be unique")
        names_vertices = names_vertices.values
        if (names_edges is not None) and len(
            np.setdiff1d(names_edges, names_vertices)
        ):
            raise ValueError(
                "Some vertices in the edge DataFrame are missing from "
                + "vertices DataFrame"
            )
        names = names_vertices
    if names is not None:
        nv = len(names)
    else:
        nv = edges.iloc[:, :2].values.max() + 1
    g = Graph(n=nv, directed=directed)
    if names is not None:
        for v, name in zip(g.vs, names):
            v["name"] = name
    if (vertices is not None) and (vertices.shape[1] > 1):
        cols = vertices.columns
        for v, (_, attr) in zip(g.vs, vertices.iterrows()):
            for an in cols[1:]:
                v[an] = attr[an]
    if names is not None:
        names_idx = pd.Series(index=names, data=np.arange(len(names)))
        e0 = names_idx[edges.values[:, 0]]
        e1 = names_idx[edges.values[:, 1]]
    else:
        e0 = edges.values[:, 0]
        e1 = edges.values[:, 1]
    g.add_edges(list(zip(e0, e1)))
    if edges.shape[1] > 2:
        for e, (_, attr) in zip(g.es, edges.iloc[:, 2:].iterrows()):
            for a_name, a_value in list(attr.items()):
                e[a_name] = a_value
    return g


def haversine(x, y):
    """
    Calculate the distance (in km) between two points on Earth using their longitude and latitude.
    
    Parameters
    ----------
    x : [float] longitude.
    y : [float] latitude.
    
    Returns
    ----------
    numeric vector of distances between the crimes (in km).
    
    Notes
    ----------
    As the Earth is nearly spherical, the haversine formula provides a good approximation of the distance between two points of the Earth surface, with a less than 1% error on average.
    """
    R = 6372.8
    dLat = radians(y[0] - x[0])
    dLon = radians(y[1] - x[1])
    lat1 = radians(x[0])
    lat2 = radians(y[0])
    a = sin(dLat/2)**2 + cos(lat1)*cos(lat2)*sin(dLon/2)**2
    c = 2*asin(sqrt(a))
    s=R * c
    return s


def linkage_sID(BF, group, method='average'):
    """
    Performs hierarchical linkage for Bayes Factors based on specified groups.

    Parameters
    ----------
    BF : [array] array-like of Bayes Factor values.
    group : [int] array-like of group identifiers corresponding to each Bayes Factor.
    method : [str] method of linkage; options are 'average', 'single', 'complete'.
    
    Returns
    ----------
    DataFrame containing grouped Bayes Factors and their linkage values,
    sorted in descending order.
    """
    # Определяем функцию для связи
    def link_fun(x):
        return {
            'single': np.max(x),
            'complete': np.min(x),
            'average': np.mean(x)
        }
    # Группируем по 'group' и применяем link_fun
    df = pd.DataFrame({'BF': BF, 'group': group})
    # Применение groupby и apply для вычисления связей
    grouped = df.groupby('group')['BF'].agg(link_fun).reset_index()
    # Плоская таблица с результатами
    grouped[['single', 'complete', 'average']] = grouped['BF'].apply(lambda x: pd.Series(x))
    grouped.drop('BF', axis=1, inplace=True)
    grouped_columns = ['group', 'single', 'complete', 'average']
    # Сортировка по выбранному методу
    if method not in grouped_columns:
        raise ValueError(f"Method '{method}' not recognized. Choose from 'average', 'single', or 'complete'.")
    Y_sorted = grouped.sort_values(by=method, ascending=False).reset_index(drop=True)
    return Y_sorted


def make_breaks(x, mode='quantile', nbins=None, binwidth=None):
    """
    Make break points for binning continuous predictors.

    Parameters
    ----------
    x : [array-like of shape] observed sample.
    mode : [str] one of 'width' (fixed width) or 'quantile' (default) binning.
    nbins : [int] number of bins.
    binwidth : [int] bin width; corresponds to quantiles if mode='quantile'.

    Returns
    ----------
    Set of unique break points for binning.
    """
    def seq(start,stop,step):
        r=list(np.arange(start,stop,step))
        r.append(stop)
        return np.array(r)
    # Проверка корректности параметров
    if (nbins is not None and binwidth is not None) or (nbins is None and binwidth is None):
        raise ValueError("Specify exactly one of nbins or width")
    # Установка диапазона значений
    if mode == 'width':
        rng = (np.nanmin(x), np.nanmax(x))
        if binwidth is not None:
            # Создаем разбиения на основе ширины
            bks = np.unique(np.concatenate(([rng[0]], seq(rng[0], rng[1] + binwidth, binwidth), [rng[1]])))
        else:
            # Создаем разбиения по количеству интервалов
            bks = np.linspace(rng[0], rng[1], nbins + 1)
    elif mode == 'quantile':
        if binwidth is not None:
            # Создаем разбиения на основе квантилей
            probs = seq(0, 1, binwidth)
        else:
            # Если не задана ширина, создаем разбиения по количеству интервалов
            probs = list(np.linspace(0, 1, nbins + 1))
        bks = list(x.quantile(probs))
    return np.sort((bks))  # Возврат отсортированных уникальных


def make_groups(X, method=1):
    """
    Generates crime groups from crime series data.

    This function generates crime groups that are useful for making unlinked pairs and for agglomerative linkage

    Parameters
    ----------
    X : [DataFrame] crime series data, generated from makeSeriesData with offender IDs, crime IDs and the event datetime.
    method : [int, default 1] method forms crimes groups: Method=1 forms groups by finding the maximal 
                        connected offender subgraph. Method=2 forms groups from the unique group of co-offenders. 
                        Method=3 forms from groups from offenderIDs.

    Returns
    ----------
    Vector of crime group labels.

    Notes
    ----------
    Method=1 forms groups by finding the maximal connected offender subgraph. So if two offenders have ever co-offended, 
    then all of their crimes are assigned to the same group. Method=2 forms groups from the unique group of co-offenders. 
    So for two offenders who co-offended, all the co-offending crimes are in one group and any crimes committed individually 
    or with other offenders are assigned to another group. Method=3 forms groups from the offender(s) responsible. So a crime 
    that is committed by multiple people will be assigned to multiple groups.
    """
    if method == 1:
        pairwise = lambda A: np.array(list(combinations(A, 2))) if len(A) > 1 else None

        Y = X.groupby('crimeID')['offenderID'].unique().apply(list)
        EL = np.concatenate([pairwise(y) for y in Y if len(y) > 1])
        EL=pd.DataFrame(EL)

        G=set(X['offenderID'])
        Gm=graphDataFrame(EL, directed=False, vertices=pd.DataFrame(G)).simplify()
        Gcl=Gm.connected_components().membership
        CG = pd.DataFrame(list(zip(Gcl, G)), columns =['cl', 'offenderID'])
        CGdata = pd.merge(X['offenderID'],CG,left_on=['offenderID'], right_on=['offenderID'], how='left')
    elif method == 2:
        ID = X.groupby('crimeID')['offenderID'].apply(lambda x: ', '.join(sorted(set(x)))).reset_index()
        ID.columns = ['crimeID', 'group']
        g = ID
        g['cl'] = g['group'].factorize()[0]
        CG = g.set_index('crimeID')['cl'].reindex(X['crimeID']).astype(int).reset_index()
        CGdata = pd.merge(X[['crimeID','offenderID']],CG,left_on=['crimeID'], right_on=['crimeID'], how='left')
        CGdata = CGdata.drop(columns='crimeID')
    elif method == 3:
        CG = X['offenderID'].factorize()[0]
        CGdata=pd.DataFrame({'offenderID':X['offenderID'], 'cl':CG})
    return CGdata


def make_linked(X, thres=365):
    """
    Generates unique indices for linked crime pairs (with weights).

    Parameters
    ----------
    X : crime series data, generated from makeSeriesData with offender IDs, crime IDs and the event datetime.
    thres : the threshold (in days) of allowable time distance (default: thres=365).

    Returns
    ----------
    Dataframe of all linked pairs (with weights).

    Notes
    ----------
    For linked crime pairs, the weights are such that each crime series contributes a total weight of no greater than 1. 
    Specifically, the weights are Wij = min(1/Nm: Vi,Vj) in Cm, where Cm is the crime series for offender m and Nm is 
    the number of crime pairs in their series (assuming Vi and Vj are together in at least one crime series). Such that 
    each crime series contributes a total weight of 1. Due to co-offending, the sum of weights will be smaller than the 
    number of series with at least two crimes.
    """
    pairwise = lambda A: np.array(list(combinations(A, 2))) if len(A) > 1 else np.nan
    W = X.groupby('offenderID').apply(lambda x: pairwise(x.index)).dropna()
    LP = np.concatenate(W, axis=0)
    i1 = np.reshape(LP[:,[0]],len(LP)); i2 = np.reshape(LP[:,[1]],len(LP))
    D = pd.DataFrame({'i1': X['crimeID'][i1].reset_index(drop=True),
                    'i2': X['crimeID'][i2].reset_index(drop=True),
                    'offenderID': X['offenderID'][i1].reset_index(drop=True)})
    val=abs(X['TIME'][i1].reset_index(drop=True)-X['TIME'][i2].reset_index(drop=True)).dt.days
    D=D.loc[(val<=thres)]
    flip = D[['i1', 'i2']].apply(lambda row: row[0] > row[1], axis=1)
    D.loc[flip, ['i1', 'i2']] = D.loc[flip, ['i2', 'i1']].values
    D = D.drop(D[D.i1 == D.i2].index)                  
    D = D.drop_duplicates()
    tab=D.groupby(D['offenderID'],as_index=True).size().reset_index()
    tab=tab.rename(columns={tab.columns[1]: "wt"})
    EL= pd.merge(D,tab,left_on=['offenderID'], right_on=['offenderID'], how='left')
    EL.drop(['offenderID'], axis='columns', inplace=True)
    EL['wt']=1/EL['wt']
    unique_names = EL['i1'] + ':' + EL['i2']
    minweights = EL.groupby(unique_names)['wt'].min()
    EL2 = EL.set_index(unique_names).loc[minweights[unique_names].index].reset_index(drop=True)
    EL2=EL2.drop_duplicates(keep='first').reset_index(drop=True)
    return EL2


def make_unlinked(X, m=40, thres=365, seed=None, method=1):
    """
    Generates a sample of indices of unlinked crime pairs.

    This function generates a set of crimeIDs of unlinked crime pairs. It selects first, crime groups 
    are identifyed as the maximal connected offender subgraphs. Then indices are drawn from each crime 
    group and paired with crimes from other crime groups according to weights to ensure that large groups 
    don't give the most events

    Parameters
    ----------
    X : [DataFrame] crime series data, generated from makeSeriesData with offender IDs, crime IDs and the event datetime.
    m : [int] the number of samples from each crime group for unlinked pairs (default: m=40).
    thres : [int] the threshold (in days) of allowable time distance (default: thres=365).
    seed : [int] seed for random number generation (default: seed=None).
    method : [int] method forms crimes groups: Method=1 (default) forms groups by finding the maximal connected offender subgraph. 
    Method=2 forms groups from the unique group of co-offenders. Method=3 forms from groups from offenderIDs.
    
    samplepairs : [str] the method of forming the unlinked crime pairs: 'random' randomly (default) or 'full' a complete set.
    

    Returns
    ----------
    Dataframe of all unlinked pairs (with weights).

    Notes
    ----------
    To form the unlinked crime pairs, crime groups are identified as the maximal connected offender subgraphs. 
    Then indices are drawn from each crime group (with replacment) and paired with crimes from other crime groups 
    according to weights that ensure that large groups don't give the most events.
    """
    if seed is not None:
        np.random.seed(seed)
    xCG=make_groups(X, method=method)
    nCG=xCG['cl'].nunique()
    nCrimes = xCG['cl'].value_counts().rename_axis('cl').reset_index(name='Count')
    Y = pd.concat([X[['crimeID', 'TIME']].reset_index(drop=True), xCG['cl']], axis=1)
    Y = pd.merge(Y, nCrimes, left_on=['cl'], right_on=['cl'], how='left')
    Y['Count']=(1/(Y['Count']*nCG))
    Y=Y.rename(columns={"Count": "wt"})
    Y=Y.drop_duplicates(keep = 'first').reset_index(drop = True)
    Y=Y.drop_duplicates(subset=['cl'], keep = 'first').reset_index(drop = True)
    Y=Y.rename(columns={"cl": "CG"})
    EL=pd.DataFrame(columns=['i1','i2','val'])
    I = np.array(range(len(Y)))
    for i in range(nCG):
        ind = Y.index[Y['CG'] == i].tolist()
        i1 = np.random.choice(ind, m, replace=True)
        ind = Y['CG'] == i
        i2 = np.random.choice(I[~ind], m, p=Y.loc[~ind, 'wt']/Y.loc[~ind, 'wt'].sum())
        val =abs(Y.loc[i1, 'TIME'].reset_index(drop=True)-Y.loc[i2, 'TIME'].reset_index(drop=True))
        val=pd.to_timedelta(val, errors='coerce').dt.days
        el = pd.DataFrame({
            'i1': np.array(Y.loc[i1, 'crimeID']),
            'i2': np.array(Y.loc[i2, 'crimeID']),
            'val': val})
        EL = pd.concat([EL,el]) if EL is not None else el
    EL = EL[(EL['val'] <= thres)]
    flip = EL[['i1', 'i2']].apply(lambda row: row[0] > row[1], axis=1)
    EL.loc[flip, ['i1', 'i2']] = EL.loc[flip, ['i2', 'i1']].values
    EL = EL.drop_duplicates(['i1', 'i2'])
    EL['val']=pd.Series([1] * len(EL))
    EL=EL.rename(columns={'val': "wt"})
    EL = EL.drop(EL[EL.i1 == EL.i2].index)
    EL=EL.drop_duplicates(keep='first').reset_index(drop=True)
    return EL


def makePairs(X, thres=365, m=40, seed=None, method=1):
    """
    Generates indices of linked and unlinked crime pairs (with weights).

    These functions generate a set of crimeIDs for linked and unlinked crime pairs. 
    Linked pairs are assigned a weight according to how many crimes are in the crime series. 
    For unlinked pairs, crimes are selected from each crime group and pairs them with crimes in other crime groups.

    Parameters
    ----------
    X : [DataFrame] crime series data, generated from makeSeriesData with offender IDs, crime IDs and the event datetime.
    thres : [int] the threshold (in days) of allowable time distance (default: thres=365).
    m : [int] the number of samples from each crime group for unlinked pairs (default: m=40).
    seed : [int] seed for random number generation.
    method : [int, default 1] method forms crimes groups: Method=1 forms groups by finding the maximal connected offender subgraph. 
                        Method=2 forms groups from the unique group of co-offenders. Method=3 forms from groups from offenderIDs.
    samplepairs : [str] the method of forming the unlinked crime pairs: 'random' randomly (default) or 'full' a complete set.

    Returns
    ----------
    Dataframe of indices of crime pairs with weights. The last column 'type' indicates if the crime pair is linked or unlinked.

    Notes
    ----------
    Method=1 forms groups by finding the maximal connected offender subgraph. So if two offenders have ever co-offended, then all 
    of their crimes are assigned to the same group. Method=2 forms groups from the unique group of co-offenders. So for two offenders 
    who co-offended, all the co-offending crimes are in one group and any crimes committed individually or with other offenders are assigned 
    to another group. Method=3 forms groups from the offender(s) responsible. So a crime that is committed by multiple people will be 
    assigned to multiple groups.
    makePairs is a Convenience function that calls makeLinked and makeUnlinked and combines the results.
    """
    linked_pairs = make_linked(X, thres)
    unlinked_pairs = make_unlinked(X, m, thres, seed, method=method)
    linked_pairs['type'] = 'linked'
    unlinked_pairs['type'] = 'unlinked'
    all_pairs = pd.concat([linked_pairs, unlinked_pairs],ignore_index=True)
    all_pairs = all_pairs[~all_pairs[['i1', 'i2']].apply(lambda x: x[0] == x[1], axis=1)]
    return all_pairs


def makeSeriesData(crimedata, offender_table, time="midpoint"):
    """
    Make dataframe of crime series data.

    Creates a dataframe with index to crimedata and offender information. It is used to generate the linkage data

    Parameters
    ----------
    crimedata : [DataFrame] dataframe of crime incidents. crimedata must have columns named: crimeID, DT.FROM and DT.TO.
    offenderTable : [DataFrame] offender table that indicates the offender(s) responsible for solved crimes. 
                                offenderTable must have columns named - offenderID and crimeID.
    time : [str] the event time to be returned: 'average', 'early' (default) or 'later'.

    Returns
    ----------
    Dataframe representation of the crime series present in the crimedata. It includes the crime IDs ('crimeID'), 
    index of that crimeID in the original crimedata ('Index'), the crime series ID ('CS') corresponding to each offenderID 
    and the event time ('TIME').

    Notes
    ----------
    The creates a crimeseries data object that is required for creating linkage data. It creates a crime series ID ('CS') 
    for every offender. Because of co-offending, a single crime ('crimeID') can belong to multiple crime series.
    """
    cid = np.array(crimedata["crimeID"].unique())
    oid = np.array(getCriminals(cid, offender_table))
    CS = getCrimeSeries(oid, offender_table, restrict=cid, show_pb=False)
    nCS = len(CS)
    nCrimes = list(map(len, [item['crimeID'] for item in CS]))
    a = np.array(sum(list(map(lambda x: x["crimeID"], CS)), []))
    b = np.array([crimedata.index[crimedata['crimeID'] == value][0] for value in a])
    if crimedata["DT.TO"].isnull().all():
        crimedata["DT.TO"] = crimedata["DT.FROM"]
    if 'DT.TO' not in crimedata:
        crimedata['DT.TO'] = crimedata['DT.FROM']
    series_data = pd.DataFrame({
        "crimeID": a,
        "Index": b,
        "CS": np.repeat(range(nCS), nCrimes)})
    series_data = series_data.merge(offender_table[['crimeID','offenderID']],on='crimeID')
    series_data = series_data.drop_duplicates(subset=['crimeID','Index','offenderID'])
    if time == "midpoint":
        series_data = series_data.merge(crimedata[['crimeID','DT.FROM','DT.TO']],on='crimeID')
        series_data['TIME'] = series_data['DT.FROM'] + (series_data['DT.TO'] - series_data['DT.FROM'])/2
        series_data = series_data.drop(['DT.FROM','DT.TO'], axis=1)
    elif time == "earliest":
        series_data = series_data.merge(crimedata[['crimeID','DT.FROM']],on='crimeID')
        series_data = series_data.rename(columns={"DT.FROM": "TIME"})
    elif time == "latest":
        series_data = series_data.merge(crimedata[['crimeID','DT.TO']],on='crimeID')
        series_data = series_data.rename(columns={"DT.TO": "TIME"})
    return series_data


def naiveBayes(data, var, weights=None, df=20, nbins=30, partition='quantile'):
    """
    Naive bayes classifier using histograms and shrinkage.

    Fits a naive bayes model to continous and categorical/factor predictors

    Parameters
    ----------
    data : [DataFrame] dataframe of the evidence variables of the crimes incident data and 
            including columne of binary vector indicating linkage of crime pairs (1 = linked, 0 = unlinked).
    var : [list, str] list of the names or column numbers of specific predictors.
    weights : [DataFrame] a column in dataframe of observation weights or the column name in data that corresponds to the weights.
    df : [int] the effective degrees of freedom for the variables density estimates (default: df=20).
    nbins : [int] number of bins (default: nbins=30).
    partition : [str] one of 'width' (fixed width) or 'quantile' (default) binning.

    Returns
    ----------
    BF a bayes factor object representing list of component bayes factors.

    Notes
    ----------
    After binning, this adds pseudo counts to each bin count to give df approximate degrees of freedom. 
    If partition=quantile, this does not assume a continuous uniform prior over support, but rather a discrete uniform over 
    all (unlabeled) observations points.

    Example
    ----------
    X=compareСrimes(allPairs,Crimes,varlist=varlist)
    Y=pd.DataFrame(np.where(allPairs['type']=='linked',1,0),columns=['Y'])
    D=pd.concat([X,Y],axis=1)
    train,test = train_test_split(D,test_size=0.3)
    var=['spatial','temporal','tod','dow','Location','MO','Weapon','AgeV']
    NB=naiveBayes(train,var,df=10,nbins=15)
    """
    X=data.iloc[:,:-1]
    y=data.iloc[:,-1:]
    # Проверяем веса
    if weights is not None:
        if not isinstance(weights, (pd.Series, np.ndarray)) or weights.dtype.kind != 'f':
            raise ValueError("Weights must be a numeric vector (float or integer).")
        if any(weights < 0):
            raise ValueError("Negative weights not allowed.")
    # Применяем веса, если они есть
    if weights is not None:
        weights = weights
        X=X[var]
        # Обучение модели
        BF=naivebayesfit(X, y, weights=weights, df=df, nbins=nbins, partition=partition)
    else:
        weights=X['wt']
        X=X[var]
        # Обучение модели
        BF=naivebayesfit(X, y, weights=weights, df=df, nbins=nbins, partition=partition)
    return BF


def naivebayesfit(X, y, weights=None, df=20, nbins=30, partition='quantile'):
    """
    Direct call to naive bayes classifier.

    Parameters
    ----------
    X : [DataFrame] dataframe of the evidence variables of the crimes incident data.
    y : [array-like of shape] binary vector indicating linkage of crime pairs (1 = linked, 0 = unlinked).
    weights : vector of observation weights or the column name in data that corresponds to the weights (default: weights=None).
    df : [int] the effective degrees of freedom for the variables density estimates (default: df=20).
    nbins : [int] number of bins.
    partition : [str] one of 'width' (fixed width) or 'quantile' (default) binning.

    Returns
    ----------
    Dictionary of component bayes factors.
    """
    # Установим режим разбиения
    assert partition in ['quantile', 'width'], "Partition must be either 'quantile' or 'width'"
    # Преобразуем X в DataFrame
    X = pd.DataFrame(X)
    # Убедимся, что y является целочисленным массивом
    y = y.astype(int)
    # Получаем названия переменных
    vars = X.columns
    nvars = len(vars)
    # Установим степень свободы
    df = np.full(shape=nvars, fill_value=df)
    # Список для хранения байесовских факторов
    BF = [None] * nvars
    # Если веса не указаны, устанавливаем их равными единицам
    if weights is None:
        weights = np.ones_like(y, dtype=float)
    # Основной цикл для каждой переменной
    for j in range(nvars):
        var = vars[j]
        x = X[var]
        # Если x является числовым, генерируем разбиения
        if x.dtype == 'float64':
            bks = make_breaks(x, partition, nbins)
        else:
            bks = None
        # Получаем байесовский фактор для данной переменной
        BF[j] = getBF(x, y, weights, breaks=bks, df=df[j])
    # Устанавливаем имена и класс
    BF = dict(zip(vars, BF))
    BF['class'] = 'naiveBayes'
    return BF


def plot_bf(BF, log_scale=True, show_legend=True, xlim=None, ylim=None, 
            cols=('darkred','darkblue'), background=True, bkgcol='lightgray',
            figsize=(8, 5), ax = None):
    """
    Plots 1D for predictors of Naive Bayes Model.
    
    Parameters
    ----------
    BF : [object] model produced by function naiveBayes.
    log_scale : [bool] if logscale=True calculates the natural logarithm of the elements used to plot (default: log_scale=True).
    show_legend : [bool] if show_legend=True (default) the legend is placed on the plot.
    xlim : [int] set the x limits of the current x-axes (default: xlim=None).
    ylim : [int] set the x limits of the current y-axes (default: ylim=None).
    cols : [tuple: str] the colors of the bar faces for positive and negative values ​​of the Bayes factor 
                        (default: cols=('darkred','darkblue')).
    background : [bool] if background=True (default) set the face color of the Figure rectangle.
    bkgcol : [str] set the face color of the Figure rectangle (default: bkgcol='lightgray').
    figsize : [tuple: int] width, height in inches of plot (float, float), default: figsize=(8, 5)
    ax : axes are added using subplots (default: ax = None).
    
    Returns
    ----------
    Plots 1D of Bayes factor.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    if background:
        ax.set_facecolor(bkgcol)
    red_patch = mpatches.Patch(color='darkred', label='Favors H₁')
    blue_patch = mpatches.Patch(color='darkblue', label='Favors H₀')
    if 'from' and 'to' in BF.columns.to_list():
        BF = BF.dropna(subset=['from','to'])
        # Создаем массивы для оси X и оси Y
        n = len(BF)
        xx = np.concatenate(([BF['from'].iloc[0]], BF['to'].values))
        yy = np.concatenate((BF['BF'].values, [BF['BF'].iloc[-1]]))
        # Установка пределов по оси Y
        if ylim is None:
            ylim = [np.nanmin(BF['BF']), np.nanmax(BF['BF'])]
            if log_scale:
                ylim = [x * y for x, y in zip([-1, 1], [min(abs(np.log(ylim))), max(abs(np.log(ylim)))])]
        # Установка пределов по оси X
        if xlim is None:
            xlim = [np.nanmin(xx)-.5, np.nanmax(xx)+.5]
        # Логарифмическое преобразование (если необходимо)
        if log_scale:
            yy = np.log(yy)
        # Рисуем график
        baseline = 0 if log_scale else 1
        for i in range(n):
            ax.fill_between(xx[i:i + 2], yy[i], baseline,
                            color=cols[0] if yy[i] > baseline else cols[1],
                            alpha=0.75)
        ax.grid(which = "major", linewidth = 1)
        ax.grid(which = "minor", linewidth = 0.3)
        ax.minorticks_on()
        ax.set_ylabel('log(BF)' if log_scale else 'BF')
        # Добавление легенды
        if show_legend:
            ax.legend(handles=[red_patch, blue_patch], loc='best', labelcolor=cols)     
    else:
        # Столбчатый график для логарифмов BF
        baseline = 0 if log_scale else 1
        if log_scale:
            BF['logBF'] = np.log(BF['BF'])
        else:
            BF['logBF'] = BF['BF']
        mp = np.arange(len(BF))  # Индексы для барплота
        baseline = 0 if log_scale else 1
        ax.axhline(y=baseline, color='black', linestyle='--')
        ax.grid(linewidth = 1)
        ax.bar(mp, BF['logBF'], color=[cols[0] if val > 0 else cols[1] for val in BF['logBF']],
                    alpha=0.75)
        ax.set_xticks(mp, BF['value'])
        ax.set_ylabel('log(BF)' if log_scale else 'BF')
        # Добавление легенды
        if show_legend:
            ax.legend(handles=[red_patch, blue_patch], loc='best', labelcolor=cols)


def plot_crimeClust_bayes(data, ind, legend_shrink=0.9, figsize=(10, 7),
                    step_yticks = 10, step_xticks = 100, y_ticks_revers = False,
                    cmap = 'viridis', main_title = 'Probability crimes are linked',
                    y_label = 'Unsolved Crime', x_label = 'All crime'):
    """
    Image plot of linkage probabilities

    Image plot  how strongly the unsolved crimes are linked to the existing (solved) crime series.

    Parameters
    ----------
    data : [dict] Bayesian model-based clustering approach produced by crimeClust_bayes.
    ind : [dict] array-like with indexes of unsolved crimes.
    legend_shrink : [float] fraction by which to multiply the size of the colorbar (default: legend_shrink=0.9).
    figsize : [tuple: int] width, height in inches of plot (float, float), default: figsize=(10, 7).
    step_yticks : [int] display step of ytick location on the y-axis (default: step_yticks=10).
    step_xticks : [int] display step of ytick location on the x-axis (default: step_xticks=100).
    y_ticks_revers : [bool] whether to display the yticks on the y-axis in the reverse order (default: False).
    cmap : [str] the Colormap instance or registered colormap name used to map scalar data to colors (default: cmap='viridis').
    main_title : [str] set a title for the plot (default: 'Probability crimes are linked').
    y_label set a title for the y-axis (default: 'Unsolved Crime').
    x_label set a title for the y-axis (default: 'All crime').
    """
    data = pd.DataFrame(data[ind]).set_index(ind)
    if y_ticks_revers:
        data = data.sort_index(ascending=False)
    plt.figure(figsize=figsize, facecolor='white')  # Установите размер фигуры
    # Используйте imshow для отображения данных DataFrame как изображения
    img = plt.imshow(data, cmap=cmap, aspect='auto')  # 'viridis' - это цветовая схема
    # Добавьте цветочную легенду
    cbar = plt.colorbar(img, orientation='vertical', shrink=legend_shrink)
    # Настройте оси
    ticks_x = range(0, len(data.columns), step_xticks)  # Отображение меток через каждые десять столбцов
    plt.xticks(ticks_x, data.columns[ticks_x])
    ticks_y = range(0, len(data.index), step_yticks)  # Отображение меток через каждые десять индексов
    plt.yticks(ticks_y, data.index[ticks_y])
    plt.title(main_title)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.show()


def plot_hcc(tree, labels=None, yticks=np.arange(-2, 9, 2), figsize=(15, 7), hang=-1, font_size=10, **kwargs):
    """
    Plot a hierarchical crime clustering object

    This function creates a dendrogram object and then plots it used log Bayes factor.

    Parameters
    ----------
    tree : [array-like of shape] an object produced from function crimeClust_hier.
    labels : [array-like of shape] Crime IDs used to plot hierarchical crime clustering.
    yticks: [tuple: int] set the y-limits and step yticks of the y-axes (default: np.arange(-2, 9, 2)).
    figsize : [tuple: int] a method used to change the dimension of plot window, width, height in inches (default: figsize=(15,7)).
    hang : [int] displacement for display (default: hang=-1).
    font_size : [int] Tick label font size in points (default: font_size=10).
    **kwargs: arguments of the plotting functions scipy.cluster.hierarchy.dendrogram.
    
    Returns
    ----------
    A dendrogram.
    """
    # Сохраняем старые параметры
    old_par = plt.rcParams.copy()
    if labels is None:
        labels = tree.get('crimeID')
    offset = tree['offset']
    tree = tree['hc']
    # Устанавливаем параметры оформления
    plt.figure(figsize=figsize, facecolor='white')
    plt.ylabel('log Bayes factor')
    plt.gca().set_ylim([hang - 1, max(yticks) + hang])
    annotate_above = kwargs.pop('annotate_above', 0)
    # Преобразуем дерево в дендрограмму и рисуем его
    Dd = hierarchy.dendrogram(tree, orientation='top', labels=labels, **kwargs)
    # Установка расстояний для узлов
    for i, d, c in zip(Dd['icoord'], Dd['dcoord'], Dd['color_list']):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            if y > annotate_above:
                plt.plot(x, y, 'o', c=c)
                plt.annotate("%.3g" % y, (x, y), xytext=(0, -5), fontsize=font_size, textcoords='offset points', va='top', ha='center')
    # Установка цвета для меток
    for leaf, leaf_color in zip(plt.gca().get_xticklabels(), Dd["leaves_color_list"]):
        leaf.set_color(leaf_color)
    # Настройка осей Y
    labs = -yticks
    for lab in labs:
        plt.axhline(y=lab + offset, color="grey", linestyle='--', linewidth=0.5)
    # Устанавливаем деления для левой оси
    plt.yticks(labs + offset, labels=-labs, fontsize=font_size)
    # Установка пределов оси Y
    plt.gca().set_ylim(bottom=math.floor(min(tree[:,2])), top=math.ceil(max(tree[:,2])))
    # Подсчет количества кластеров и их отображение на правой оси
    nClusters = [np.sum(tree[:,2] < threshold) for threshold in (labs + offset)]
    # Устанавливаем деления для правой оси
    ax2 = plt.gca().twinx()
    ax2.set_yticks(labs + offset, labels=nClusters, fontsize=font_size)
    # Синхронизация пределов оси Y
    ax2.set_ybound(lower=math.floor(min(tree[:,2])), upper=math.ceil(max(tree[:,2])))
    # Название осей
    ax2.set_ylabel('number of clusters')
    plt.title('Dendrogram with Cluster Information')
    plt.show()
    return Dd


def plotBF(BF, var, logscale=True, figsize=(17,28), plotstyle='ggplot', legend=True, **kwargs):
    """
    Plots for predictors of Naive Bayes Model.

    Makes plots of components bayes factors from naiveBayes. This function attempts to plot all of the component plots in one window or individual Bayes factors

    Parameters
    ----------
    BF : [array-like of shape] Bayes Factor.
    var : [list, str] list of the names or column numbers of specific predictors.
    logscale : [bool, default True] if logscale=True calculates the natural logarithm of the elements used to plot.
    figsize : [tuple: int] width, height in inches of plot (float, float), default: figsize=(17,28).
    plotstyle : [str] stylesheets from Matplotlib for plot.
    legend : [bool, default True] if legend=True the legend is placed on the plot.
    **kwargs : arguments of the plotting functions from a collection matplotlib.pyplot.

    Returns
    ----------
    Plot of Bayes factor.
    """
    varls=var[0:4]
    nnn=len(list(BF.keys())[:-1])
    d_keys = list(BF.keys())[:-1]
    nlen = int([nnn/2 if (nnn%2 ==0) else (nnn+1)/2][0])
    fig = plt.figure(figsize=figsize, facecolor='white')
    plt.style.use(plotstyle)
    for i in range(nnn):
        ax = fig.add_subplot(nlen, 2, i+1)
        BFi=BF[d_keys[i]]
        varl=var[i]
        red_patch = mpatches.Patch(color='orangered', label='$A_{L}$')
        red_patch_2 = mpatches.Patch(color='mediumblue', label='$A_{U}$')
        if varl in varls:
            ylim=((min(BFi['BF'])-0.2),max(BFi['BF'])+0.2)
            title='BF'
            if logscale is True:
                ylim = np.array([-1,1])*np.array([min(12,max(abs(np.log(ylim))))])
                title='log(BF)'
            BFt=BFi['to']
            n=len(BFt)
            BFto=list(BFi['to'].dropna())
            BFto.reverse()
            BFto.insert(0,BFi['from'][i])
            xx=BFto
            yy = list(BFi['BF'])
            if logscale is True:
                yy = np.log(yy)
            x = range(n)
            for i in range(n):
                clrs = ['mediumblue' if (x < 0) else 'orangered' for x in yy]
            x_=np.linspace(min(BFi['from']),max(BFi['to']),len(BFi['from']))
            x_=x_.round(0).astype('int') if max(x_)>=15 else x_.round(2)
            ax = plt.gca()
            ax.bar(x, yy, align='center', color = clrs, **kwargs)
            if legend is True:
                ax.legend(handles=[red_patch,red_patch_2],loc='best')
            ax.set_xticks(x,labels=x_,rotation='vertical')
            ax.set_ylim(ylim)
            ax.set_ylabel(title)
            ax.set_title(varl)
        else:
            BFi['logBF']=np.log(BFi['BF'])
            x = BFi['value']
            yy=BFi['logBF']
            ax = plt.gca()
            for i in range(n):
                clrs = ['mediumblue' if (x < 0) else 'orangered' for x in yy]
            ax.bar(x, yy, align='center', color = clrs, **kwargs)
            if legend is True:
                ax.legend(handles=[red_patch,red_patch_2],loc='best')
            ax.set_xticks(x)
            ax.set_ylabel(title)
            ax.set_title(varl)
    ax.plot()
    return plt.show()


def plotHCL(Z, labels, figsize=(15,8), **kwargs):
    """
    Plot a hierarchical crime clustering object of crime linkage based on probabilities

    This function creates a dendrogram object and then plots it used probabilities for linkage of crimes pairs.

    Parameters
    ----------
    Z : [array-like of shape] an object produced from crimeLink_Clust_Hier.
    labels : [array-like of shape] Crime IDs used to plot hierarchical crime clustering.
    figsize : [tuple: int] a method used to change the dimension of plot window, width, height in inches (default: figsize=(15,8)).
    **kwargs : arguments of the plotting functions from a collection matplotlib.pyplot.

    Returns
    ----------
    A dendrogram.
    """
    plt.figure(figsize=figsize,facecolor='w')
    plt.style.use('classic')
    plt.ylabel('distance between crimes',fontsize=14)
    annotate_above = kwargs.pop('annotate_above', 0)
    max_d = kwargs.pop('max_d', None)
    if max_d and 'color_threshold' not in kwargs:
        kwargs['color_threshold'] = max_d
    Dd=dendrogram(Z=Z, labels=list(labels), **kwargs)
    for i, d, c in zip(Dd['icoord'], Dd['dcoord'], Dd['color_list']):
        x = 0.5 * sum(i[1:3])
        y = d[1]
        if y > annotate_above:
            plt.plot(x, y, 'o', c=c)
            plt.annotate("%.3g" % y, (x, y), xytext=(0, -5), textcoords='offset points', va='top', ha='center')
    if max_d:
        plt.axhline(y=max_d, c='k')
    plt.show()


def plotnaiveBayes(x, **kwargs):
    """
    Plots for Naive Bayes Model
    
    Parameters
    ----------
    x : [object] model produced by function naiveBayes.
    **kwargs : arguments of the plotting functions plot_bf.
    
    Returns
    ----------
    plots of Bayes factor from a naive Bayes model
    """
    varls = list(x.keys())[:-1]  # Получаем все ключи, кроме последнего
    n = len(varls)  # Количество графиков
    ncols = 2  # Количество столбцов
    nrows = (n + ncols - 1) // ncols  # Количество строк (округление вверх)
    # Создаем подграфики
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, nrows * 4), facecolor='white')
    axs = axs.flatten()  # Получаем одномерный массив подграфиков для легкости доступа
    # Обход графиков
    for i, key in enumerate(varls):
        df = x[key]  # Получаем DataFrame по ключу
        plot_bf(df, ax=axs[i], **kwargs)  # Вызов функции plotbf для построения графика
        axs[i].set_title(f'Plot for {key}')  # Установка заголовка для каждого графика
    # Удаляем пустые подграфики
    for j in range(i + 1, len(axs)):
        fig.delaxes(axs[j])  # Удаляем неиспользуемые подграфики
    plt.tight_layout()  # Установка аккуратного расположения между подграфиками
    plt.show()


def plotROC(x, y, xlim, ylim, xlabel, ylabel, title, rocplot=True, plotstyle='classic'):
    """
    Plot of ROC curves and other metrics for classifier.

    Returns of plot of the Receiver Operating Characteristic (ROC) metric and other metrics to evaluate classifier output quality for crime series linkage

    Parameters
    ----------
    x : [array-like of shape] input values, e.g., false positive rate.
    y : [array-like of shape] input values, e.g., true positive rate.
    xlim : [list: int] get or set the x limits of the current axes.
    ylim : [list: int] get or set the y limits of the current axes.
    xlabel : [str] set the label for the x-axis.
    ylabel : [str] set the label for the y-axis.
    title : [str] set a title for the plot.
    rocplot : [bool, default rocplot=True] If is True, the ROC curve will be plotted, if is False, the other metrics of for classifier not will be plotted.
    plotstyle : [str] style sheets for main plot (default plotstyle='classic').

    Returns
    ----------
    Plot display.

    Examples
    ----------
    nb=predictnaiveBayes(NB,test[test.columns[3:-1]],var)
    v=getROC(nb,test['Y'])
    plotROC(v['FPR'],v['TPR'],xlim=[-0.01, 1.0], ylim=[0.0, 1.03], xlabel='False Positive Rate', ylabel='True Positive Rate', title='ROC NB')
    """
    plt.figure(facecolor='white')
    plt.style.use(plotstyle)
    if rocplot == True:
        AUC=round(auc(x,y),3)
        plt.plot(sorted(list(x)), sorted(list(y)), color='red', lw=2, label = 'model')
        plt.plot(sorted(list(x)), sorted(list(y)), color='red', lw=2, label = 'AUC: %.3f'%AUC)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='dashdot', label='random')
        plt.plot([0,0,1,1],[0,1,1,1],'green', lw=2, linestyle='--', label='perfect')
        plt.legend(loc=4)
    else:
        plt.plot(sorted(list(x)), sorted(list(y)), color='red', lw=2)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()


def predict_bf(BF, x, log=True):
    """
    Generate prediction of a component bayes factor.

    Parameters
    ----------
    BF : [array-like of shape] Bayes Factor.
    x : [array-like of shape] vector of new predictor values.
    log : [bool, default True] if log=True, return the log bayes factor estimate.

    Returns
    ----------
    Estimated (log) bayes factor from a single predictor.
    """
    if 'from' and 'to' in BF: # Извлечение границ из атрибута BF
        breaks = np.unique(BF[['from','to']].values)
        # Если границы заданы, выполняем разбиение значений
        if breaks is not None:
            x = pd.cut(x, bins=breaks, duplicates='drop')   
        # Получаем индексы для соответствия
        ind = x.map(lambda x: BF['value'][BF['value'].isin([x])].index[0] if x in BF['value'].values else None)    
        dfBF=BF.get('BF')
        # Устанавливаем значение байесовского фактора в 1 для отсутствующих данных
        dfBF=pd.concat([dfBF,pd.Series([1],index=[len(dfBF)])])
        ind = np.where(np.isnan(np.array(ind)), max(dfBF.index), np.array(ind))
        bf = dfBF.iloc[ind].values  # Получаем байесовские факторы
    else:
        bf = BF['BF'].iloc[np.array(x)].values
    # Если log = True, применяем логарифм
    if log:
        bf = np.log(bf)
    return bf


def predictGB(X, varlist, gB):
    """
    Predict class probabilities for crimes groups.
    
    Parameters
    ----------
    X : [DataFrame] training dataframe of crime incidents with predictors.
    varlist : [dict] list of the names of specific predictors.
    gB : [object] model of Gradient Boosting for classification produced from GBC.
    
    Returns
    ----------
    dataframe of links between crime pairs based on probabilities.
    """
    gb=gB
    crimeIDs=set(X['crimeID'])
    allPairs=pd.DataFrame(list(combinations(crimeIDs, 2)),columns=['i1', 'i2'])
    A=compareСrimes(allPairs,X,varlist=varlist)
    A1=A[A.columns[3:]]
    res2=gb.predict_proba(A1)
    Result=A[A.columns[0:2]]
    res3=pd.DataFrame({'link':res2[:, 1]})
    Result=pd.concat([Result, res3],axis=1).sort_values(by=['link'],ascending=False)
    return Result


def predictnaiveBayes(model, newdata, components=False, var=None, log=True):
    """
    Generate prediction (sum of log bayes factors) from a naiveBayes object.

    Parameters
    ----------
    model : [object] a naive bayes object from naiveBayes.
    newdata : [DataFrame] a dataframe of new predictors, column names must match NB names and var.
    components : [bool, default False] return the log bayes factors from each component (components=True) 
                                      or return the sum of log bayes factors (components=False).
    var : [list, str] a list of the names or column numbers of specific predictors (default: var=None).
    log : [bool, default True] if log=True, return the log bayes factor estimate.

    Returns
    ----------
    Estimated (log) bayes factor from a single predictor.

    Notes
    ----------
    This does not include the log prior odds, so will be off by a constant.
    """
    if newdata is None:
        raise ValueError("newdata must be provided")
    X = pd.DataFrame(newdata)  # Преобразуем новые данные в DataFrame
    NB = model  # Наивный байесовский объект
    # Если переменные не указаны, используем имена модели
    if var is None:
        var = list(NB.keys())[:-1]
    # Проверка на наличие переменных в новых данных
    missing_vars = [var_ for var_ in var if var_ not in X.columns]
    if missing_vars:
        print(f"Warning: The columns: {', '.join(missing_vars)} are missing from newdata")

    # Удаляем отсутствующие переменные
    var = [var_ for var_ in var if var_ not in missing_vars]
    nvars = len(var)
    BF = np.full((X.shape[0], nvars), np.nan)  # Инициализируем матрицу
    for j in range(nvars):
        var_ = var[j]
        if var_ in X.columns:
            BF[:, j] = predict_bf(NB[var_], X[var_], log=log)
    # Устанавливаем названия столбцов
    colnames = var
    # Возвращаем компоненты или суммы
    if components:
        return pd.DataFrame(BF, columns=colnames)
    BF_sum = np.nansum(BF, axis=1)  # Суммируем по строкам
    return BF_sum


def seq(start, stop, step):
    """
    Generate regular sequences.
    
    Parameters
    ----------
    start : [int] the starting values of the sequence.
    stop : [int] the end (maximal) of the sequence.
    step : [int] increment of the sequence.
    
    Returns
    ----------
    array of regular sequences.
    """
    r=list(np.arange(start,stop,step))
    r.append(stop)
    return np.array(r)


def seriesCrimeID(offenderID, unsolved, solved, offenderData, varlist, estimateBF):
    """
    Identification of offender related with unsolved crimes.

    Performs crime series identification by finding the crime series that are most closely related (as measured by Bayes Factor) to an offender.

    Parameters
    ----------
    offenderID : [str] an offender ID that is in offenderTable.
    unsolved : [DataFrame] incident data for the unsolved crimes. Must have a column named 'crimeID'.
    solved : [DataFrame] incident data for the solved crimes. Must have a column named 'crimeID'.
    offenderTable : [DataFrame] offender table that indicates the offender(s) responsible for solved crimes. offenderTable must have columns named - offenderID and crimeID.
    varlist : [dict] a list with elements named: crimeID, spatial, temporal and categorical. Each element should be a column names of crimedata corresponding to that feature: crimeID - crime ID for the crimedata that is matched to unsolved and solved, spatial - X,Y coordinates (in long and lat) of crimes, temporal - DT.FROM, DT.TO of crimes, categorical - categorical crime variables.
    estimateBF : [function] function to estimate the log bayes factor from evidence variables.

    Returns
    ----------
    Dataframe with two columnes: 'crimeID' - ID's of unsolved crimes, 'BF' - Bayes Factor; or print "This offender is not related to crimes".
    """
    offID=getCrimes(offenderID, solved, offenderData)
    offID=offID.copy().reset_index(drop=True)
    offID['crimeID'] = offID['crimeID'].str.replace(" ".join(re.findall("[a-zA-Z]+", offID['crimeID'][0])), offenderID)
    crimeIDs=set(unsolved['crimeID']) | set(offID['crimeID'])
    allPairsM=pd.DataFrame(list(combinations(crimeIDs, 2)), columns=['i1', 'i2'])
    allPairsM1=allPairsM[allPairsM['i1'].str.contains(offenderID)]
    allPairsM2=allPairsM[allPairsM['i2'].str.contains(offenderID)]
    allPairsM=pd.concat([allPairsM1, allPairsM2], ignore_index=True)
    allPairsM=allPairsM.loc[allPairsM['i1'].str.contains(offenderID) != allPairsM['i2'].str.contains(offenderID)]
    crimeData=pd.concat([offID, unsolved], ignore_index=True)
    EvVar=compareСrimes(allPairsM, crimeData, varlist=varlist)
    def replaces(df):
        for dfres in df:
            dfres=[]
            for i in range(0,df.shape[0]):
                dfres.append(str(df.i1[i] if (offenderID in df.crimeID[i]) else df.crimeID[i]))
            return dfres
    EvVar['crimeID']=replaces(EvVar)
    bf=pd.DataFrame({'crimeID':EvVar['crimeID'],'BF':estimateBF(EvVar)})
    bf2=bf.sort_values(by=['BF'], ascending=False).drop_duplicates(subset=['crimeID']).reset_index(drop=True)
    DF = bf2[~(bf2['BF'] < 0)]
    DF.index += 1
    if len(DF) > 0:
        return DF
    else:
        print('This offender is not related to crimes')


def seriesOffenderID(crime, unsolved, solved, seriesData, varlist, estimateBF,
              linkage_method='average', group_method=3, **kwargs):
    """
    Crime series identification.

    Performs crime series identification by finding the crime series that are most closely 
    related (as measured by Bayes Factor) to an unsolved crime

    Parameters
    ----------
    crime : [str] an crime ID that is in unsolved.
    unsolved : [DataFrame] incident data for the unsolved crimes. Must have a column named 'crimeID'.
    solved : [DataFrame] incident data for the solved crimes. Must have a column named 'crimeID'.
    seriesData : [DataFrame] crime series data, generated from makeSeriesData.
    varlist : [dict] a list with elements named: crimeID, spatial, temporal and categorical. 
                     Each element should be a column names of crimedata corresponding to that feature: 
                     crimeID - crime ID for the crimedata that is matched to unsolved and solved, 
                     spatial - X,Y coordinates (in long and lat) of crimes, temporal - DT.FROM, DT.TO of crimes, 
                     categorical - categorical crime variables.
    estimateBF : [function] function to estimate the log bayes factor from evidence variables.
    linkage_method : [str] method of linkage; options are 'average' (default), 'single', 'complete'.
    group_method : [int, default 3] method forms crimes groups: groupmethod=1 forms groups by finding the maximal 
                    connected offender subgraph. groupmethod=2 forms groups from the unique group of co-offenders. 
                    groupmethod=3 forms from groups from offenderIDs.
    **kwargs : other arguments for function compareСrimes.

    Returns
    ----------
    DataFrame of series with their respective Bayes factors.

    Notes
    ----------
    Method=1 forms groups by finding the maximal connected offender subgraph. So if two offenders have ever co-offended, 
    then all of their crimes are assigned to the same group. Method=2 forms groups from the unique group of co-offenders. 
    So for two offenders who co-offended, all the co-offending crimes are in one group and any crimes committed individually 
    or with other offenders are assigned to another group. Method=3 forms groups from the offender(s) responsible. So a crime 
    that is committed by multiple people will be assigned to multiple groups.

    Example
    ----------
    seriesOffenderID('Crime3',UnsolvedData,Crimes,seriesData,varlist,estimateBF)
    """
    # Make crime data
    if crime in list(solved['crimeID']):
        raise ValueError("Error in unsolved crime ID")
    # Combine crime and solved data
    crimedata = unsolved[unsolved['crimeID'].isin([crime])]
    crimedata=pd.concat([crimedata,solved],ignore_index=True)
    # Compare crime pairs
    pairs = pd.DataFrame({
        'i1': crimedata['crimeID'][0],
        'i2': solved['crimeID'].unique()
    })
    # Assuming compareСrimes is a predefined function that takes pairs and crimedata
    X = compareСrimes(pairs, crimedata, varlist, **kwargs)
    # Estimate the Bayes Factor
    bf = estimateBF(X)
    # Find 'nearest' series to new.crime
    CG = make_groups(seriesData, method=group_method)  # Assuming this is a predefined function
    SD = pd.DataFrame({'crimeID': seriesData['crimeID'], 'CG': CG['cl']})
    # Match Bayes Factor with crimeID
    SD['BF'] = bf[np.array([pairs['i2'].tolist().index(id) for id in SD['crimeID']])]  # Assuming bf is a list or array
    SD = SD.dropna()  # Remove NA values
    # Perform hierarchical clustering
    Z = linkage_sID(SD['BF'], group=SD['CG'], method=linkage_method)
    seriesData = seriesData.merge(SD[['crimeID', 'CG']], on='crimeID', how='left')
    seriesData=seriesData.rename(columns={"CG": "group"})
    return {'score': Z, 'groups': seriesData}  # Returning Series Data and linkage matrix