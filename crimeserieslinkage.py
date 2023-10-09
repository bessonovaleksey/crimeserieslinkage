#!/usr/bin/env python
"""
Statistical methods for identifying serial crimes and related offenders

Copyright (c) 2022, A.A. Bessonov (bestallv@mail.ru)

Routines in this module:

clusterPath(crimeID,tree,n=0)
compareCrimes(dfPairs, crimedata, varlist, longlat=False, method='convolution')
compareSpatial(C1, C2, pairsdata, longlat='False')
compareTemporal(DT1, DT2, pairsdata, method='convolution')
comparisonCrime(crimedata, crimeID)
conv_circ(x, y)
crimeClust_Hier(crimedata, varlist, estimateBF, linkage)
crimeCount(seriesdata)
crimeLink(crimedata, varlist, estimateBF, sort=True)
crimeLink_Clust_Hier(crimedata, predGB, linkage)
datapreprocess(data)
difftime(t1, t2)
euclid_distance(x, y)
expAbsDiff(day_x, day_y, L1, L2, pairsdata)
expAbsDiffcirc(X, Y, mod = 24, n = 2000, method='convolution')
GBC(X, Y, start, end, step, n_splits=5, learning_rate=0.2, **kwargs)
getBF(x, y, weights=None, breaks=None, df=5)
getCrimes(offenderID, crimedata, offenderTable)
getCrimeSeries(offenderTable, *arg)
getCriminals(crimeID, offenderTable)
getD(y, X, mod=24)
getROC(f, y)
graphDataFrame(edges, directed=True, vertices=None, use_vids=False)
haversine(x, y)
julian_date(date)
makebreaks(x, mode='quantile', nbins='NULL', binwidth='NULL')
makeGroups(X, method=1)
makeLinked(X, crimedata, valtime=365)
makePairs(X, crimedata, method=1, valtime=365)
makeSeriesData(crimedata, offenderTable, time='early')
makeUnlinked(X, crimedata, method=1, valtime=365)
naiveBayes(data, var, partition='quantile', df=20, nbins=30)
naiveBayesfit(X, y, weights=None, partition='quantile', df=20, nbins=30)
plot_hcc(Z, labels, **kwargs)
plotBF(BF, var, logscale=True, figsize=(15,15), plotstyle='ggplot', legend=True, **kwargs)
plotHCL(Z, labels, **kwargs)
plotROC(x, y, xlim, ylim, xlabel, ylabel, title, rocplot=True, plotstyle='classic')
predictBF(BF, x, log=True)
predictGB(X, varlist, gB)
predictnaiveBayes(result, newdata, var, components=True, log=True)
predictNB_classes(result, newdata, var, components=True, log=True)
seriesCrimeID(offenderID, unsolved, solved, offenderData, varlist, estimateBF)
seriesOffenderID(crime, unsolved, solved, seriesdata, varlist, estimateBF, n=10, groupmethod=3)

"""
from __future__ import division, absolute_import, print_function

__all__ = ['clusterPath', 'compareCrimes', 'compareSpatial', 'compareTemporal', 'comparisonCrime',
           'conv_circ', 'crimeClust_Hier', 'crimeCount', 'crimeLink', 'crimeLink_Clust_Hier',
           'datapreprocess','difftime', 'euclid_distance', 'expAbsDiff',
           'expAbsDiffcirc', 'GBC', 'getBF', 'getCrimes', 'getCrimeSeries',
           'getCriminals', 'getD', 'getROC', 'graphDataFrame', 'haversine', 'julian_date',
           'makebreaks', 'makeGroups','makeLinked', 'makePairs',
           'makeSeriesData', 'makeUnlinked', 'naiveBayes', 'naiveBayesfit',
           'plot_hcc', 'plotBF', 'plotHCL', 'plotROC', 'predictBF',
           'predictGB', 'predictnaiveBayes', 'predictNB_classes',
           'seriesCrimeID', 'seriesOffenderID']

import pandas as pd
import numpy as np
import datetime
import itertools
import igraph
import math
import re
import scipy.stats as sps
from scipy.stats import reciprocal
from scipy import signal
from math import radians, cos, sin, asin, sqrt
from sklearn.model_selection import train_test_split
from heapq import nlargest
from more_itertools import distinct_combinations
from itertools import combinations
from itertools import combinations_with_replacement
from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.fft import fft, ifftn
from sklearn.metrics import auc
from typing import List
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import KFold, GridSearchCV
from igraph import Graph


def clusterPath(crimeID,tree,n=0):
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
    ll=list(tree[2])
    if crimeID not in ll:
        raise ValueError("Error in crime ID")
    bf=-tree[0][:,2]+tree[3]
    df=pd.DataFrame({'i1':tree[0][:,0],'i2':tree[0][:,1]}).astype(int).rename(index = lambda x: x + 1)
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
    for j in range(len(tree[2])-1):
        cc=list(df.iloc[j])
        res1=[]
        for i in range(len(cc)):
            if cc[i] < len(tree[2]):
                res=ll[cc[i]]
            if cc[i] >= len(tree[2]):
                indx=cc[i]-len(tree[2])
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
    if Dff.shape[0]==0:
        print("Change the value -n-")


def compareCrimes(dfPairs, crimedata, varlist, longlat=False, method='convolution'):
    """
    Creates evidence variables by calculating distance between crime pairs.

    Calculates spatial and temporal distance, difference in categorical, and absolute value of numerical crime variables

    Parameters
    ----------
    dfPairs : [DataFrame] dateframe with 2 columns of crime IDs that are checked for linkage.
    crimedata : [DataFrame] dataframe of crime incident data. There must be a column named of crimes that refers to the crimeIDs given in dfPairs. Other column names must correspond to what is given in varlist.
    varlist : [dict] a list with elements named: crimeID, spatial, temporal and categorical. Each element should be a column names of crimedata corresponding to that feature: crimeID - crime ID for the crimedata that is matched to dfPairs, spatial - X,Y coordinates (in long and lat) of crimes, temporal - DT.FROM, DT.TO of crimes, categorical - categorical crime variables.
    longlat : [bool, default False] are spatial coordinates (long,lat), calculated using the haversine method (default False) or Euclidean distance is returned in kilometers.
    method : [str] use convolution (default) or monte carlo integration (method='numerical').

    Returns
    ----------
    data frame of various proximity measures between the two crimes.
    """ 
    if all(x in list(crimedata) for x in sum(list(varlist.values()), [])):
        cat=varlist['categorical']
        temp=varlist['temporal']
        spat=varlist['spatial']
        Pairs = pd.merge(dfPairs, crimedata, left_on=['i1'], right_on=['crimeID'], how='left')
        i1spat=Pairs[spat]
        i1temp=Pairs[temp]
        i1cat=Pairs[cat]
        Pairs.drop(Pairs.iloc[:,3:], axis=1, inplace=True)
        Pairs = pd.merge(dfPairs, crimedata, left_on=['i2'], right_on=['crimeID'], how='left')
        i2spat=Pairs[spat]
        i2temp=Pairs[temp]
        i2cat=Pairs[cat]
        Pairs.drop(Pairs.iloc[:,3:], axis=1, inplace=True)
        Pairs=Pairs
        TTD=compareTemporal((i1temp['DT.FROM'],i1temp['DT.TO']),(i2temp['DT.FROM'],i2temp['DT.TO']),Pairs,method=method)
        TTS=compareSpatial((i1spat['X'],i1spat['Y']),(i2spat['X'],i2spat['Y']),Pairs,longlat=longlat)
        TTC=pd.DataFrame(np.where(i1cat == i2cat, 1, 0),columns=cat)
        Edf=pd.concat([Pairs,TTS,TTD,TTC],axis=1)
        Edf=Edf.fillna(0)
        return Edf
    else:
        print ('There are names in varlist that do not match column names in crimedata')


def compareSpatial(C1, C2, pairsdata, longlat='False'):
    """
    Make spatial evidence variables.
    
    Calculates spatial distance between crimes (in km)
    
    Parameters
    ----------
    C1 : [DataFrame] dateframe with 2 columns of coordinates for the crimes.
    C2 : [DataFrame] dateframe with 2 columns of coordinates for the crimes.
    pairsdata : [DataFrame] dataframe of crime groups from crime series data.
    longlat : [bool, default False] if false (default) the the coordinates are in (Long,Lat), else assume a suitable project where euclidean distance can be applied.
    
    Returns
    ----------
    numeric vector of distances between the crimes (in km) internal.
    """    
    for spatial in pairsdata:
        spatial=[]
        for i in range(max(pairsdata.index)+1):
            if longlat:
                d=haversine((C1[0][i],C1[1][i]),(C2[0][i],C2[1][i]))
                spatial.append(d)
            else:
                d=euclid_distance((C1[0][i],C1[1][i]),(C2[0][i],C2[1][i]))
                spatial.append(d)
    TableTTS=pd.DataFrame({'spatial':spatial})
    return TableTTS


def compareTemporal(DT1, DT2, pairsdata, method='convolution'):
    """
    Make temporal evidence variable from (possibly uncertain) temporal.
    
    Calculates the temporal distance between crimes
    
    Parameters
    ----------
    DT1 : [DataFrame] dataframe of (DT.FROM,DT.TO) for the crimes.
    DT2 : [DataFrame] dataframe of (DT.FROM,DT.TO) for the crimes.
    pairsdata : [DataFrame] dataframe of crime groups from crime series data.
    method : [str] use convolution (default) or monte carlo integration (method='numerical').
    
    Returns
    ----------
    dataframe of expected absolute differences: temporal - overall difference (in days)  [0,max], tod - time of day difference (in hours)  [0,12], dow - fractional day of week difference (in days) [0,3.5].
    """
    def TOD(x):
        tod=x.timestamp()/3600%24
        return tod
    def DOW(x):
        dow=(x.timestamp()/(3600*24))%7
        return dow
    vectdifftime = np.vectorize(difftime,otypes=[np.float],cache=False)
    L1 = vectdifftime(DT1[1],DT1[0])
    L2 = vectdifftime(DT2[1],DT2[0])
    vectjulian_date = np.vectorize(julian_date,otypes=[np.float],cache=False)
    day1=vectjulian_date(DT1[0])
    day2=vectjulian_date(DT2[0])
    vectTOD=np.vectorize(TOD,otypes=[np.float],cache=False)
    tod1=vectTOD(DT1[0])
    tod2=vectTOD(DT2[0])
    vectDOW=np.vectorize(DOW,otypes=[np.float],cache=False)
    dow1=vectDOW(DT1[0])
    dow2=vectDOW(DT2[0])
    temporal=expAbsDiff(day1, day2, L1, L2,pairsdata)
    for tod in pairsdata:
        tod=[]
    for i in list(range(max(pairsdata.index)+1)):
        todi=expAbsDiffcirc((tod1[i],tod1[i]+L1[i]), (tod2[i],tod2[i]+L2[i]),method=method)
        tod.append(todi)
    for dow in pairsdata:
        dow=[]
    for i in list(range(max(pairsdata.index)+1)):
        dowi=expAbsDiffcirc((dow1[i],dow1[i]+L1[i]/24), (dow2[i],dow2[i]+L2[i]/24),mod=7,method=method)
        dow.append(dowi)
    TableTTD=pd.DataFrame({'temporal':temporal,'tod':tod,'dow':dow})
    return TableTTD


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


def conv_circ(x, y):
    """
    Make the Fast Fourier Transform to compute the several kinds of convolutions of two sequences are treated as circular, i.e., periodic.
    
    Parameters
    ----------
    x : [float] numeric sequences to be convolved.
    y : [float] numeric sequences to be convolved.
    The input sequences x and y must have the same length.

    Returns
    ----------
    Discrete, linear convolution of x and y.
    """
    return np.real(np.fft.ifft(np.fft.fft(x)*np.fft.fft(y)))


def crimeClust_Hier(crimedata, varlist, estimateBF, linkage):
    """
    Agglomerative Hierarchical Crime Series Clustering.
    
    Run hierarchical clustering on a set of crimes using the log Bayes Factor as the similarity metric
    
    Parameters
    ----------
    crimedata : [DataFrame] dataframe of crime incidents. Must contain a column named crimeID.
    varlist : [dict] a list of the variable names columns of crimedata used to create evidence variables with compareCrimes.
    estimateBF : [function] function to estimate the log bayes factor from evidence variables.
    linkage : [str] the type of linkage for hierarchical clustering: 'average' - uses the average bayes factor, 'single' - uses the largest bayes factor (most similar), 'complete' - uses the smallest bayes factor (least similar), 'weighted' - a balanced group average (also called WPGMA), 'centroid' - unweighted pair group method using centroids (also called UPGMC), 'median' - the centroid of the new cluster is accepted as the average value of the centrides of two combined clusters (WPGMC algorithm), 'ward' - uses the Ward variance minimization algorithm.
    
    Returns
    ----------
    The hierarchical clustering encoded as a linkage matrix based on log Bayes Factor.
    Values:
    hc : a linkage matrix.
    dhc : a vector of log Bayes Factor.
    crimesID : a list of crimeID used to return a linkage matrix based on log Bayes Factor.
    offsets : maximum of the log bayes factor.
    """
    crimeIDs=set(crimedata['crimeID'])
    allPairs=pd.DataFrame(list(combinations(crimeIDs, 2)),columns=['i1', 'i2'])
    A=compareCrimes(allPairs,crimedata,varlist=varlist)
    bf=estimateBF(A)
    d2=-bf
    offset=math.ceil(max(bf))
    d2=d2+offset
    hc = hierarchy.linkage(d2, method=linkage)
    global dhc; dhc = d2
    global crimesID; crimesID=crimeIDs
    global offsets; offsets=offset
    return hc, dhc, crimesID, offsets


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
    varlist : [dict] a list of the variable names columns of crimedata used to create evidence variables with compareCrimes.
    estimateBF : [function] function to estimate the log bayes factor from evidence variables.
    sort : [bool, default True] sort data of columnes based on log Bayes Factor in descending (sort=True, default) or ascending order.
    
    Returns
    ----------
    A dataframe of links between crime pairs based on log Bayes Factor.
    """
    crimeIDs=set(crimedata['crimeID'])
    allPairs=pd.DataFrame(list(combinations(crimeIDs, 2)),columns=['i1', 'i2'])
    A=compareCrimes(allPairs,crimedata,varlist=varlist)
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
    varc=data.columns[1:]
    for i in range(len(varc)):
        if type(data[varc[i]][0]) is str:
            data[varc[i]] = data[varc[i]].astype('category').cat.codes
    return data


def difftime(t1, t2):
    """
    Calculates time between two vectors of datetimes.
    
    Parameters
    ----------
    t1 : [datetime64, datetime, Timestamp] first set of times.
    t2 : [datetime64, datetime, Timestamp] second set of times.
    
    Returns
    ----------
    Numeric vector of times between the datetime objects.
    """
    times=abs(t1-t2)
    hours=(times.total_seconds())//3600
    return hours


def euclid_distance(x, y):
    """
    Сalculates the Euclidean distance between two geographic coordinates
    
    Parameters
    ----------
    x : [float] longitude.
    y : [float] latitude.
    
    Returns
    ----------
    numeric vector of distances between the crimes (in km).
    """
    s = (sqrt((y[0]-x[0])**2 + (y[1]-x[1])**2))/1000
    return s


def expAbsDiff(day_x, day_y, L1, L2, pairsdata):
    """
    Expected absolute difference between the two dates of the crime pairs, expressed in days of the week.
    
    Calculates the expected absolute difference of two uniform two dates of the crime pairs, expressed in days of the week
    
    Parameters
    ----------
    day_x : [tuple: float, int] a list of two values - Julian minimum day of the week and Julian maximum day of the week of the first crime in the pair.
    day_y : [tuple: float, int] a list of two values - Julian minimum day of the week and Julian maximum day of the week of the second crime in the pair.
    L1 : [float, int] difference in hours between two dates of the first crime (DT.FROM and DT.TO).
    L2 : [float, int] difference in hours between two dates of the second crime (DT.FROM and DT.TO).
    pairsdata : dataframe of crime groups from crime series data.
    
    Returns
    ----------
    The expected absolute difference between the two dates of the crime pairs, expressed in days of the week.
    """
    L1=L1
    L2=L2
    for temporal in pairsdata:
        temporal=[]
    def expAbsDiff2(day_1, day_2):
        d_1,d1_1=day_1,day_1+L1[i]/24
        d_2,d1_2=day_2,day_2+L2[i]/24
        if d1_1<d_1:
            raise ValueError('d1_1<d_1')
        if d1_2<d_2:
            raise ValueError('d1_2<d_2')
        if d_1<=d_2:
            Sx,Sx1=d_1,d1_1
            Sy,Sy1=d_2,d1_2
        else:
            Sx,Sx1=d_2,d1_2
            Sy,Sy1=d_1,d1_1
        if Sx1<=Sy:
            return (((Sy+Sy1)/2)-((Sx+Sx1)/2))
        ks=Sx,Sx1,Sy,Sy1
        bks=list(ks)
        bks=sorted(bks)
        sz=np.diff(bks)
        mids=bks[1:]-sz/2
        Sxx=Sx,Sx1
        Sxx=np.diff(Sxx)
        Syy=Sy,Sy1
        Syy=np.diff(Syy)
        if Sx1<=Sy1:
            px=sz*[1,1,0]/Sxx
            py=sz*[0,1,1]/Syy
            return ((mids[1]-mids[0])*px[0]*py[1]+(mids[2]-mids[0])*px[0]*py[2]+(sz[1]/3)*px[1]*py[1]+(mids[2]-mids[1])*px[1]*py[2])
        if Sx1>Sy1:
            px=sz*[1,1,1]/Sxx
            return ((mids[1]-mids[0])*px[0]+(sz[1]/3)*px[1]+(mids[2]-mids[1])*px[2])
    for i in list(range(max(pairsdata.index)+1)):
        temporali=expAbsDiff2(day_x[i],day_y[i])
        temporal.append(temporali)
    return temporal


def expAbsDiffcirc(X, Y, mod = 24, n = 2000, method='convolution'):
    """
    Expected absolute difference between the two dates of the crime pairs, expressed in time of day (in hours) or day of week (in days).
    
    Estimates the expected circular temporal distance between crimes using discrete FFT or numerical integration
    
    Parameters
    ----------
    X : [tuple float, int] a list of two values - min and min+length the two dates of the first crime of the pair, expressed in time of day (in hours) or day of week (in days). X[0] must be >= 0 and X[1] >= X[1]. It is possible that X[1] can be > mod.
    Y : [tuple float, int] a list of two values - min and min+length the two dates of the second crime of the pair, expressed in time of day (in hours) or day of week (in days). X[0] must be >= 0 and X[1] >= X[1]. It is possible that X[1] can be > mod.
    mod : [int] the period of time. E.g., mod=24 for time of day (in hours), mod=7 for day of week (in days).
    n : [int] number of bins for discretization of continuous time domain. E.g., there is 1440 min/day, so n = 2000 should give close to minute resolution.
    method : [str] use convolution (method='convolution', default) or monte carlo integration (method='numerical').
    
    Returns
    ----------
    The expected absolute difference between the two dates of the crime pairs, expressed in time of day (in hours) or day of week (in days).
    """
    if X[0]<0 or Y[0]<0:
        raise ValueError('X and Y must be >0')
    if np.diff(X)>=mod or np.diff(Y) >= mod:
        d=mod/4
        return d
    else:
        if np.diff(X) == 0:
            d=getD(X[0],Y,mod=mod)
            return d
        else:
            if np.diff(Y) == 0:
                d=getD(Y[0],X,mod=mod)
                return d
            else:
                if method =='convolution':
                    f=lambda n,l,x=1:min(f(n,l,x*e)for e in l)if x<n else x
                    if n%2 !=0:
                        n=f(n+1,[2,3,5])
                    theta=np.linspace (0, mod, num = n)
                    x=np.diff(reciprocal.cdf(theta,X[0],X[1]))+np.diff(reciprocal.cdf(np.array(theta)+mod,X[0],X[1]))
                    y=np.diff(reciprocal.cdf(theta,Y[0],Y[1]))+np.diff(reciprocal.cdf(np.array(theta)+mod,Y[0],Y[1]))
                    conv = conv_circ(x, y)
                    tt=np.where(np.delete(theta, 0)<=mod/2, np.delete(theta, 0), np.sort(np.delete(theta, 0))[::-1])
                    d=sum(conv*tt)
                    return d
                if method =='numerical':
                    if np.diff(Y) < np.diff(X):
                        tt = np.array(np.linspace (Y[0], Y[1], num = n))
                        d=np.mean(getD(tt,X,mod=mod))
                    else:
                        tt = np.array(np.linspace (X[0], X[1], num = n))
                        d=np.mean(getD(tt,Y,mod=mod))
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


def getBF(x, y, weights=None, breaks=None, df=5):
    """
    Estimates the bayes factor for continous and categorical predictors.
    
    Continous predictors are first binned, then estimates shrunk towards zero
    
    Parameters
    ----------
    x : [array-like of shape] predictor vector (continuous or categorical/factors).
    y : [array-like of shape] binary vector indicating linkage (1 = linked, 0 = unlinked).
    weights : [array-like of shape] vector of observation weights or the column name in data that corresponds to the weights (default weights=None).
    breaks : [int, default None] set of break point for continuous predictors or NULL for categorical or discrete.
    df : [int, default 5] the effective degrees of freedom for the cetegorical density estimates.
    
    Returns
    ----------
    The set containing: dataframe the levels/categories with estimated Bayes factor, 'breaks' - set of break point for continuous predictors, 'a' - list of markers of the linked and unlinked crime pairs, 'df' - the effective degrees of freedom, 'df2' - the effective degrees of freedom for linked and unlinked crime pairs.
    Notes
    ----------
    This adds pseudo counts to each bin count to give df effective degrees of freedom. Must have all possible factor levels and must be of factor class.
    Give linked and unlinked a different prior according to sample size.
    """
    def replaceNA(x):
        x=x.fillna(0)
        return x
    if weights is None:
        weights=pd.DataFrame({'wt':np.repeat(1, len(y), axis=0)})
    if x.dtypes == 'float64':
        nbks=len(breaks)
        x.fillna(max(x)+1)
        x=pd.cut(x, bins=breaks, duplicates='drop')
        xd=pd.DataFrame({'value':x.array}).drop_duplicates().dropna().sort_values(by=['value']).reset_index(drop=True)
        if all(x.isna()):
            x=pd.DataFrame(pd.qcut(range(len(x)),bks))
        tlinked=pd.concat([weights.iloc[np.where(y==1)],x.iloc[np.where(y==1)]],axis=1)
        tlinked=tlinked.pivot_table(columns=tlinked.iloc[:,1],values='wt',aggfunc='sum')
        tunlinked=pd.concat([weights.iloc[np.where(y==0)],x.iloc[np.where(y==0)]],axis=1)
        tunlinked=tunlinked.pivot_table(columns=tunlinked.iloc[:,1],values='wt',aggfunc='sum')
        fromto=pd.DataFrame({'from':breaks[:-1],'to':breaks[1:]})
        if fromto.shape[0] < tlinked.size:
            new_row = pd.Series({"from": 'NaN', "to": 'NaN'})
            for i in range(fromto.shape[0],tlinked.size):
                fromto = fromto.append(new_row, ignore_index=True)
        Nunlinked = replaceNA(pd.DataFrame({'N.unlinked':tunlinked.loc['wt']}).reset_index(drop=True))
        Nlinked = replaceNA(pd.DataFrame({'N.linked':tlinked.loc['wt']}).reset_index(drop=True))
        E=pd.concat([fromto,xd,Nlinked,Nunlinked],axis=1)
        if all(E['value'].isna()):
            E['value']=list(range(1,len(breaks)))
    else:
        x.fillna(max(x)+1)
        tlinked=pd.concat([weights.iloc[np.where(y==1)],x.iloc[np.where(y==1)]],axis=1)
        tlinked=tlinked.pivot_table(columns=tlinked.iloc[:,1],values='wt',aggfunc='sum')
        tunlinked=pd.concat([weights.iloc[np.where(y==0)],x.iloc[np.where(y==0)]],axis=1)
        tunlinked=tunlinked.pivot_table(columns=tunlinked.iloc[:,1],values='wt',aggfunc='sum')
        Nunlinked = replaceNA(pd.DataFrame({'N.unlinked':tunlinked.loc['wt']}).reset_index(drop=True))
        Nlinked = replaceNA(pd.DataFrame({'N.linked':tlinked.loc['wt']}).reset_index(drop=True))
        value=pd.DataFrame({'value':list(tlinked.columns)})
        E=pd.concat([value,Nlinked,Nunlinked],axis=1)
    def df2a(df, k, N):
        df=(N/k)*((k-df)/(df-1))
        return df
    def a2df(a, k, N):
        df=k*(N+a)/(N+k*a)
        return df
    nlevs=E.shape[0]
    df=min(df,nlevs-1e-8)
    alinked = df2a(df,k=nlevs,N=sum(E['N.linked']))
    aunlinked = df2a(df,k=nlevs,N=sum(E['N.unlinked']))
    def getP(N, a):
        gP=(N+a)/sum(N+a)
        return gP
    E['p.linked']=getP(E['N.linked'],alinked)
    E['p.unlinked']=getP(E['N.unlinked'],aunlinked)
    E['BF']=E['p.linked']/E['p.unlinked']
    E['BF']=E['BF'].fillna(1)
    a = (alinked,aunlinked)
    df2 = (a2df(alinked,k=nlevs,N=sum(E['N.linked'])),a2df(aunlinked,k=nlevs,N=sum(E['N.linked'])))
    return E, breaks, a, df, df2


def getCrimes(offenderID, crimedata, offenderTable):
    """
    Generate a list of crimes for a specific offender.

    Parameters
    ----------
    offenderID : [str] an offender ID that is in offenderTable.
    crimedata : [DataFrame] dataframe of crime incidents. Must contain a column named crimeID.
    offenderTable : [DataFrame] offender table that indicates the offender(s) responsible for solved crimes. offenderTable must have columns named - offenderID and crimeID.

    Returns
    ----------
    The subset of crimes in crimedata that are attributable to the offender named offenderID.

    Example
    ----------
    getCrimes('Prodan', Crimes, Offenders)
    """
    if offenderID in list(offenderTable['offenderID']):
        ID4 = offenderTable.loc[offenderID==offenderTable['offenderID'], 'crimeID']
        lstCrime = crimedata.loc[crimedata['crimeID'].isin(ID4)]
        return lstCrime
    else:
         print("Error in offender ID")


def getCrimeSeries(offenderTable, *arg):
    """
    Generate a list of offenders and their associated crime series.
    
    Parameters
    ----------
    offenderTable : [DataFrame] offender table that indicates the offender(s) responsible for solved crimes. offenderTable must have columns named - offenderID and crimeID.
    *arg : [str] vector of one or more of offender IDs.
    
    Returns
    ----------
    List of offenders with their associated crime series.
    
    Example
    ----------
    getCrimeSeries(Offenders, 'Prodan','Popkov')
    """
    for x in arg:
        ID3 = offenderTable.loc[x==offenderTable['offenderID'], 'crimeID']
        print("offenderID",x, "crimeID", np.array(ID3), sep='\n', end='\n\n')


def getCriminals(crimeID, offenderTable):
    """
    List of the offenders responsible for a set of solved crimes.

    Generates the IDs of criminals responsible for a set of solved crimes using the information in offenderTable

    Parameters
    ----------
    crimeID : [str] crime IDs of solved crimes.
    offenderTable : [DataFrame] offender table that indicates the offender(s) responsible for solved crimes. offenderTable must have columns named - offenderID and crimeID.

    Returns
    ----------
    List of offenderIDs responsible for crimes labeled crimeID.

    Example
    ----------
    getCriminals('Crime100', Offenders)
    """
    if crimeID in list(offenderTable['crimeID']):
        crimeID2=offenderTable['crimeID']
        which = lambda lst:list(np.where(lst)[0])
        ind = which(list(map(lambda crimeID2: crimeID2 == crimeID, crimeID2)))
        offenderID = set(offenderTable.loc[ind, 'offenderID'])
        return list(offenderID)
    else:
        print("Crimes with such ID there is no in data of offenders")


def getD(y, X, mod=24):
    """
    Expected absolute distance between the two dates of the crime pairs, expressed in time of day (in hours) or day of week (in days).
    
    Parameters
    ----------
    y : [tuple: float] a vector of times in [0, mod)
    X : [tuple: float] a list of two values - min and min+length the two dates of the crime of the pair, expressed in time of day (in hours) or day of week (in days). X[0] must be >= 0 and X[1] >= X[0]. It is possible that X[1] can be > mod. I.e., do not do X%mod.
    mod : [int] the period of time. E.g., mod=24 for time of day (in hours), mod=7 for day of week (in days).
    
    Returns
    ----------
    The expected absolute difference between the two dates of the crime pairs, expressed in time of day (in hours) or day of week (in days).
    """
    if X[0]>mod or X[0]<0:
        raise ValueError('Minimum X[0] not within limits [0,mod)')
    if X[1]<X[0]:
        raise ValueError('X[1] must be >= X[0]')
    y=(np.array(y)-X[0])%mod
    B=X[1]-X[0]
    if B == 0:
        D=mod/2-abs(mod/2-abs(np.array(y)))
        return D 
    elif B>=mod:
        D=(np.repeat(np.array(mod/4), [1 if type(y) == float or int else len (y)]))
        return D
    elif (np.diff(X)>=mod/2) and (all(np.array(y)) <= B-mod/2 if type(y) == np.ndarray else np.array(y) <= B-mod/2):
        K = mod - B/2 - (mod/2)**2/B
        D = np.array(y)*(1-mod/B) + K
        return D
    elif (np.diff(X)>=mod/2) and (all(np.array(y)) > B-mod/2 if type(y) == np.ndarray else np.array(y) > B-mod/2) and (all(np.array(y)) <= mod/2 if type(y) == np.ndarray else np.array(y) <= mod/2):
        D = (np.array(y)-B/2)**2/B+B/4
        return D
    elif (np.diff(X)>=mod/2) and (all(np.array(y)) > mod/2 if type(y) == np.ndarray else np.array(y) > mod/2) and (all(np.array(y)) <= B if type(y) == np.ndarray else np.array(y) <= B):
        K = mod - B/2 - (mod/2)**2/B
        D =  (B-np.array(y))*(1-mod/B) + K
        return D
    elif (np.diff(X)>=mod/2) and (all(np.array(y)) > B if type(y) == np.ndarray else np.array(y) > B):
        u = np.array(y)-mod/2
        D = mod/2 - B/4 - ((u-B/2))**2/B
        return D
    elif (all(np.array(y)) <B if type(y) == np.ndarray else np.array(y) <B):
        u = np.array(y)-B/2
        D = u**2/B + B/4
        return D
    elif (all(np.array(y)) >=B if type(y) == np.ndarray else np.array(y) >=B) and (all(np.array(y)) <=mod/2 if type(y) == np.ndarray else np.array(y) <=mod/2):
        D = np.array(y)-B/2
        return D
    elif (all(np.array(y)) >mod/2 if type(y) == np.ndarray else np.array(y) >mod/2) and (all(np.array(y)) <=B+mod/2 if type(y) == np.ndarray else np.array(y) <=B+mod/2):
        D = mod/2 - ((y-mod/2)**2 + (B-y+mod/2)**2) /(2*B)
        return D
    elif (all(np.array(y)) >B+mod/2 if type(y) == np.ndarray else np.array(y) >B+mod/2):
        D = mod - (np.array(y)-B/2)
        return D
    else:
        D = np.zeros([1 if type(y) == float or int else len (y)])
        return D


def getROC(f, y):
    """
    Cacluate ROC metrics for interpret the results of classification.
    
    Orders scores from largest to smallest and evaluates performance for each value. This assumes an analyst will order the predicted scores and start investigating the linkage claim in this order
    
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
    order=(-f).argsort()
    f=np.array(f)[order]
    y=np.array(y)[order]
    df=pd.DataFrame(zip(f, y), columns = ['f', 'y'])
    uniq=np.array(df.duplicated(keep='last'))
    npos=sum(y==1)
    nneg=sum(y==0)
    TP=np.cumsum(y==1)[uniq]
    FP=np.cumsum(y==0)[uniq]
    Total=np.array(list(range(1,len(f)+1)))[uniq]
    TPR=TP/npos
    FPR=FP/nneg
    PPV=TP/Total
    TotalRate=Total/len(f)
    threshold=f[uniq]
    res=pd.DataFrame({'FPR':FPR,'TPR':TPR,'PPV':PPV,'Total':Total,'TotalRate':TotalRate,'threshold':threshold})
    return res


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


def julian_date(date):
    """
    Extract the weekday on the Julian time.
    
    Parameters
    ----------
    date : [datetime64, datetime, Timestamp] object representing date of the crime.
    
    Returns
    ----------
    Returns the number of days (possibly fractional) since the origin.
    
    Notes
    ----------
    Julian Day Number is the number of days since noon UTC on the first day of 4317 BC.
    """
    julian_datetime = (367 * date.year - int((7 * (date.year + int((date.month + 9) / 12.0))) / 4.0) + int(
        (275 * date.month) / 9.0) + date.day + 1721013.5 + (
                          date.hour + date.minute / 60.0 + date.second / math.pow(60,
                                                                                  2)) / 24.0 - 0.5 * math.copysign(
        1, 100 * date.year + date.month - 190002.5) + 0.5)-2440587.5
    return round(julian_datetime,3)


def makebreaks(x, mode='quantile', nbins='NULL', binwidth='NULL'):
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
    if nbins != 'NULL' and binwidth != 'NULL' or nbins == 'NULL' and binwidth == 'NULL':
        print ('Specify exactly one of nbins or width')
    else:
        if mode == 'width':
            rng=min(x),max(x)
            if binwidth != 'NULL':
                bks=list(set(np.linspace(rng[0],rng[1],binwidth)))
            else:
                bks = list(np.linspace(rng[0],rng[1],nbins+1))
            return bks
        if mode == 'quantile':
            if binwidth != 'NULL':
                probs = list(np.linspace(0, 1,binwidth))
            else:
                probs = list(np.linspace(0, 1, nbins+1))
            bks = list(x.quantile(probs))
            return bks


def makeGroups(X, method=1):
    """
    Generates crime groups from crime series data.

    This function generates crime groups that are useful for making unlinked pairs and for agglomerative linkage

    Parameters
    ----------
    X : [DataFrame] crime series data, generated from makeSeriesData with offender IDs, crime IDs and the event datetime.
    method : [int, default 1] method forms crimes groups: Method=1 forms groups by finding the maximal connected offender subgraph. Method=2 forms groups from the unique group of co-offenders. Method=3 forms from groups from offenderIDs.

    Returns
    ----------
    Vector of crime group labels.

    Notes
    ----------
    Method=1 forms groups by finding the maximal connected offender subgraph. So if two offenders have ever co-offended, then all of their crimes are assigned to the same group. Method=2 forms groups from the unique group of co-offenders. So for two offenders who co-offended, all the co-offending crimes are in one group and any crimes committed individually or with other offenders are assigned to another group. Method=3 forms groups from the offender(s) responsible. So a crime that is committed by multiple people will be assigned to multiple groups.
    """
    if method==1:
        df1=pd.concat(g for _, g in X.groupby("crimeID") if len(g) > 1)
        df1.drop(['crimeID'], axis='columns', inplace=True)
        df1=list(df1['offenderID'])
        G=set(X['offenderID'])
        def pairwiseUC(object, to=[]):
            for i in range(0,len(object),2):
                j = i+2
                to2=tuple(object[i:j])
                to.append(to2)
            return to
        lpairs=pairwiseUC(df1)
        dfo = pd.DataFrame(lpairs)
        dfo=dfo.dropna()
        Gdf = pd.DataFrame(G, columns =['unigueID'])
        Gm=graphDataFrame(dfo, directed=False, vertices=Gdf).simplify()
        Gcl=Gm.clusters().membership
        CG = pd.DataFrame(list(zip(Gcl, G)), columns =['cl', 'offender'])
        CG1=pd.DataFrame(X['offenderID'],columns =['offenderID'])
        CGdata = pd.merge(CG1,CG,left_on=['offenderID'], right_on=['offender'], how='left')
        CGdata.drop(['offender'], axis='columns', inplace=True)
        return CGdata
    if method==2:
        ID=pd.DataFrame({'crimeID':X['crimeID'],'offenderID':X['offenderID']})
        res = ID.groupby('crimeID')['offenderID'].apply(list)
        res1=pd.DataFrame(res)
        res1=pd.DataFrame(res1.offenderID.tolist(), columns=['offenderID', 'ofl2'])
        res1 = res1.fillna(value=0)
        res1.ofl2[res1.ofl2 == 0] = res1.offenderID
        res1_1 = res1.loc[res1['offenderID'] == res1['ofl2']]
        res1_2 = res1.loc[res1['offenderID'] != res1['ofl2']]
        res1_1 = res1_1.drop(columns='ofl2')
        resUnique=res1_1.drop_duplicates(['offenderID'], keep='first')
        resUnique['cl'] = resUnique.reset_index().index+1
        CGdata = pd.merge(res1_1, resUnique, left_on=['offenderID'], right_on=['offenderID'], how='left')
        res1_2['cl'] = res1_2.reset_index().index+len(resUnique)+1
        CGdata2=pd.DataFrame({'offenderID':res1_2['offenderID'],'cl':res1_2['cl']})
        CGdata3=pd.DataFrame({'offenderID':res1_2['offenderID'],'cl':res1_2['cl']})
        CGdata = pd.concat([CGdata, CGdata2, CGdata3], ignore_index=True)
        return CGdata
    if method==3:
        ID=pd.DataFrame({'offenderID':X['offenderID']})
        offunique=set(X['offenderID'])
        offunique = pd.DataFrame(offunique, columns=['offenderID'])
        offunique['cl'] = offunique.reset_index().index+1
        CGdata = pd.merge(ID, offunique, left_on=['offenderID'], right_on=['offenderID'], how='left')
        return CGdata


def makeLinked(X, crimedata, valtime=365):
    """
    Generates unique indices for linked crime pairs (with weights).

    Parameters
    ----------
    X : crime series data, generated from makeSeriesData with offender IDs, crime IDs and the event datetime.
    crimedata : dataframe of crime incidents.
    valtime : the threshold (in days) of allowable time distance (default valtime=365).

    Returns
    ----------
    Dataframe of all linked pairs (with weights).

    Notes
    ----------
    For linked crime pairs, the weights are such that each crime series contributes a total weight of no greater than 1. Specifically, the weights are Wij = min(1/Nm: Vi,Vj) in Cm, where Cm is the crime series for offender m and Nm is the number of crime pairs in their series (assuming Vi and Vj are together in at least one crime series). Such that each crime series contributes a total weight of 1. Due to co-offending, the sum of weights will be smaller than the number of series with at least two crimes.
    """
    for res5 in X:
        res5=[]
    nOffenders = X['offenderID'].value_counts().rename_axis('offenderID').reset_index(name='Count')
    newOffenders = nOffenders.loc[~nOffenders['Count'].isin([1,2])]
    newOffenders.drop(['Count'], axis = 1)
    newOffenders = list(newOffenders['offenderID'])
    def makeLink(A, crimedata, valtime2=valtime):
        offenderID2=set(X['offenderID'])
        rt=list(offenderID2)    
        def getCS2(*arg):
            for x in arg:
                ID3 = X.loc[x==X['offenderID'], 'crimeID']
                return ID3    
        z=getCS2(A)
        df=pd.DataFrame(list(distinct_combinations(z, 2)),columns=['i1','i2'])
        result = df.assign(offenderID=A)
        dfTIME=pd.DataFrame({'crimeID':crimedata['crimeID'],'DT.FROM':crimedata['DT.FROM']})
        result = pd.merge(result,dfTIME,left_on=['i1'], right_on=['crimeID'], how='left')
        result.drop(['crimeID'], axis='columns', inplace=True)
        dfTIME2=pd.DataFrame({'crimeID':crimedata['crimeID'],'DT.TO':crimedata['DT.TO']})
        result = pd.merge(result,dfTIME2,left_on=['i2'], right_on=['crimeID'], how='left')
        result.drop(['crimeID'], axis='columns', inplace=True)
        for res2 in result:
            res2=[]
            for i in list(range(max(result.index)+1)):
                data=result['DT.TO'][i]-result['DT.FROM'][i]
                res2.append(data.days)
        val=res2
        result['valTime']=val
        new2 = result.loc[result['valTime'] < valtime2]
        new3=new2.groupby(new2['offenderID'],as_index=True).size()
        new4 = pd.DataFrame(new3, columns=['Freq'])
        new4['Var1'] = new4.index
        new4.index = np.arange(len(new4))
        new2= pd.merge(new2,new4,left_on=['offenderID'], right_on=['Var1'], how='left')
        new2.drop(['Var1'], axis='columns', inplace=True)
        EL=(1/new2['Freq'])
        new2['wt'] = EL
        new2.drop(['offenderID','DT.FROM','DT.TO','valTime','Freq'], axis='columns', inplace=True)
        return new2
    for i in list(range(0,len(newOffenders)-1)):
        res5.append(makeLink(newOffenders[i],crimedata,valtime))
    my_df = pd.concat([pd.DataFrame(x) for x in  res5], ignore_index=True)
    return my_df


def makePairs(X, crimedata, method=1, valtime=365):
    """
    Generates indices of linked and unlinked crime pairs (with weights).
    
    These functions generate a set of crimeIDs for linked and unlinked crime pairs. Linked pairs are assigned a weight according to how many crimes are in the crime series. For unlinked pairs, crimes are selected from each crime group and pairs them with crimes in other crime groups.
    
    Parameters
    ----------
    X : [DataFrame] crime series data, generated from makeSeriesData with offender IDs, crime IDs and the event datetime.
    crimedata : [DataFrame] dataframe of crime incidents.
    method : [int, default 1] method forms crimes groups: Method=1 forms groups by finding the maximal connected offender subgraph. Method=2 forms groups from the unique group of co-offenders. Method=3 forms from groups from offenderIDs.
    valtime : [int] the threshold (in days) of allowable time distance (default valtime=365).

    Returns
    ----------
    Dataframe of indices of crime pairs with weights. The last column 'type' indicates if the crime pair is linked or unlinked.
    
    Notes
    ----------
    Method=1 forms groups by finding the maximal connected offender subgraph. So if two offenders have ever co-offended, then all of their crimes are assigned to the same group. Method=2 forms groups from the unique group of co-offenders. So for two offenders who co-offended, all the co-offending crimes are in one group and any crimes committed individually or with other offenders are assigned to another group. Method=3 forms groups from the offender(s) responsible. So a crime that is committed by multiple people will be assigned to multiple groups.
    makePairs is a Convenience function that calls makeLinked and makeUnlinked and combines the results.
    """
    linkedPairs=makeLinked(X=X, crimedata=crimedata, valtime=valtime)
    unlinkedPairs=makeUnlinked(X=X, crimedata=crimedata, method=method, valtime=valtime)
    linkedPairs['type']='linked'
    unlinkedPairs['type']='unlinked'
    allPairs=pd.concat([linkedPairs,unlinkedPairs],ignore_index=True)
    return allPairs


def makeSeriesData(crimedata, offenderTable, time='early'):
    """
    Make dataframe of crime series data.

    Creates a dataframe with index to crimedata and offender information. It is used to generate the linkage data

    Parameters
    ----------
    crimedata : [DataFrame] dataframe of crime incidents. crimedata must have columns named: crimeID, DT.FROM and DT.TO.
    offenderTable : [DataFrame] offender table that indicates the offender(s) responsible for solved crimes. offenderTable must have columns named - offenderID and crimeID.
    time : [str] the event time to be returned: 'average', 'early' (default) or 'later'.

    Returns
    ----------
    Dataframe representation of the crime series present in the crimedata. It includes the crime IDs ('crimeID'), index of that crimeID in the original crimedata ('Index'), the crime series ID ('CS') corresponding to each offenderID and the event time ('TIME').

    Notes
    ----------
    The creates a crimeseries data object that is required for creating linkage data. It creates a crime series ID ('CS') for every offender. Because of co-offending, a single crime ('crimeID') can belong to multiple crime series.
    """
    offenderID=set(offenderTable['offenderID'])
    s3 = pd.Series(sorted(list(offenderID)))
    DF=pd.DataFrame({'index': s3.index, 'offenderID':s3})
    ID2=offenderTable['offenderID']
    data_pd = pd.merge(ID2,DF,on=['offenderID'], how='left')
    CS=data_pd['index']
    indexO=offenderTable.rename(index = lambda x: x + 1)
    index=pd.DataFrame(indexO.index)
    OffenderID=data_pd['offenderID']
    CrimeID=offenderTable["crimeID"]
    SeriesData=pd.DataFrame({'crimeID':offenderTable["crimeID"],'Index': index[0], 'CS': CS, 'offenderID': OffenderID})
    if time=='early':
        df=pd.DataFrame({'crimeID':crimedata['crimeID'],'DT.FROM':crimedata['DT.FROM']})
        SeriesData = pd.merge(SeriesData,df,on=['crimeID'], how='left')
        SeriesData.rename(columns={'DT.FROM': 'TIME'}, inplace=True)
    if time=='later':
        df=pd.DataFrame({'crimeID':crimedata['crimeID'],'DT.TO':crimedata['DT.TO']})
        SeriesData = pd.merge(SeriesData,df,on=['crimeID'], how='left')
        SeriesData.rename(columns={'DT.TO': 'TIME'}, inplace=True)
    if time=='average':
        for res in crimedata:
            res=[]
            for i in list(range(max(crimedata.index)+1)):
                data=crimedata['DT.TO'][i]-crimedata['DT.FROM'][i]
                res.append(crimedata['DT.TO'][i] - datetime.timedelta(days=(data.days/2)))
        df=pd.DataFrame({'crimeID':crimedata['crimeID'],'TIME':res})
        SeriesData = pd.merge(SeriesData,df,on=['crimeID'], how='left')
    SeriesData=SeriesData.dropna()
    return SeriesData


def makeUnlinked(X, crimedata, method=1, valtime=365):
    """
    GGenerates a sample of indices of unlinked crime pairs.

    This function generates a set of crimeIDs of unlinked crime pairs. It selects first, crime groups are identifyed as the maximal connected offender subgraphs. Then indices are drawn from each crime group and paired with crimes from other crime groups according to weights to ensure that large groups don't give the most events

    Parameters
    ----------
    X : [DataFrame] crime series data, generated from makeSeriesData with offender IDs, crime IDs and the event datetime.
    crimedata : [DataFrame] dataframe of crime incidents.
    method : [int] method forms crimes groups: Method=1 (default) forms groups by finding the maximal connected offender subgraph. Method=2 forms groups from the unique group of co-offenders. Method=3 forms from groups from offenderIDs.
    valtime : [int] the threshold (in days) of allowable time distance (default valtime=365).

    Returns
    ----------
    Dataframe of all unlinked pairs (with weights).

    Notes
    ----------
    To form the unlinked crime pairs, crime groups are identified as the maximal connected offender subgraphs. Then indices are drawn from each crime group (with replacment) and paired with crimes from other crime groups according to weights that ensure that large groups don't give the most events.
    """
    xCG=makeGroups(X, method=method)
    CG=xCG['cl']
    nCG=len(CG.value_counts())
    nCrimes = xCG['cl'].value_counts().rename_axis('cl').reset_index(name='Count')
    Y=pd.DataFrame({'crimeID':X['crimeID'],'CG':xCG['cl'],'TIME':X['TIME']})
    Y = pd.merge(Y, nCrimes, left_on=['CG'], right_on=['cl'], how='left')
    Y.drop(['cl'], axis='columns', inplace=True)
    Y['wt']=(1/(Y['Count']*nCG))
    Y.drop(['Count'], axis='columns', inplace=True)
    def combinator(B):
        comb=list(combinations_with_replacement(B, 2))
        return comb
    def pairwiseUCun(object):
        YUnique=[]
        Y_df  = pd.DataFrame(columns = ['i1', 'i2'])
        for i in range(len(object)-10):
            Yd=object[i:].drop_duplicates('CG',keep='first')
            Yd=pd.DataFrame({'crimeID':Yd['crimeID']})
            Yv=np.array(Yd.apply(combinator))
            XX=np.concatenate(Yv)
            Ydf1=pd.DataFrame({'i1':[i[0] for i in XX],'i2':[i[1] for i in XX],})
            YUnique.append(Ydf1)
        my_df = pd.concat([pd.DataFrame(x) for x in YUnique],ignore_index=True)
        Yser=my_df.reset_index(drop=True)
        return Yser
    YYY=pairwiseUCun(Y)
    YYUnique=YYY.drop_duplicates()
    YYUnique=YYUnique[YYUnique.i1 != YYUnique.i2]
    for res in crimedata:
        res=[]
        for i in list(range(max(crimedata.index)+1)):
            data=crimedata['DT.TO'][i]-crimedata['DT.FROM'][i]
            res.append(crimedata['DT.TO'][i] - datetime.timedelta(days=(data.days/2)))
    timeDf=pd.DataFrame({'crimeID':crimedata['crimeID'],'TIME':res})
    YYUnique = pd.merge(YYUnique,timeDf,left_on=['i1'], right_on=['crimeID'], how='left')
    YYUnique.drop(['crimeID'], axis='columns', inplace=True)
    YYUnique = pd.merge(YYUnique,timeDf,left_on=['i2'], right_on=['crimeID'], how='left')
    YYUnique.drop(['crimeID'], axis='columns', inplace=True)
    for timeint in YYUnique:
        timeint=[]
        for i in list(range(max(YYUnique.index)+1)):
            data=abs(YYUnique['TIME_y'][i]-YYUnique['TIME_x'][i])
            timeint.append(data.days)
    YYUnique['val']=timeint
    YYUnique.drop(['TIME_x','TIME_y'], axis='columns', inplace=True)
    YYUnique = YYUnique.loc[YYUnique['val'] < valtime]
    YYUnique = pd.merge(YYUnique,Y,left_on=['i2'], right_on=['crimeID'], how='left')
    YYUnique.drop(['val','crimeID','CG','TIME'], axis='columns', inplace=True)
    YYperc = [np.percentile(YYUnique['wt'], i) for i in [25, 50, 75]]
    YYUniq = YYUnique.drop(YYUnique[(YYUnique['wt'] > YYperc[2])].index)
    YYUn = YYUniq.drop(YYUniq[(YYUniq['wt'] < YYperc[0])].index)
    YYUn['wt']=1
    YYUndf = YYUn.drop_duplicates(subset = ['i1', 'i2'],keep = 'first').reset_index(drop = True)
    return YYUndf


def naiveBayes(data, var, partition='quantile', df=20, nbins=30):
    """
    Naive bayes classifier using histograms and shrinkage.
    
    Fits a naive bayes model to continous and categorical/factor predictors
    
    Parameters
    ----------
    data : [DataFrame] dataframe of the evidence variables of the crimes incident data and including columne of binary vector indicating linkage of crime pairs (1 = linked, 0 = unlinked).
    var : [list, str] list of the names or column numbers of specific predictors.
    partition : [str] one of 'width' (fixed width) or 'quantile' (default) binning.
    df : [int] the effective degrees of freedom for the variables density estimates (default df=20).
    nbins : [int] number of bins (default nbins=30).
    
    Returns
    ----------
    BF a bayes factor object representing list of component bayes factors.
    
    Notes
    ----------
    After binning, this adds pseudo counts to each bin count to give df approximate degrees of freedom. If partition=quantile, this does not assume a continuous uniform prior over support, but rather a discrete uniform over all (unlabeled) observations points.
    
    Example
    ----------
    X=compareCrimes(allPairs,Crimes,varlist=varlist)
    Y=pd.DataFrame(np.where(allPairs['type']=='linked',1,0),columns=['Y'])
    D=pd.concat([X,Y],axis=1)
    train,test = train_test_split(D,test_size=0.3)
    var=['spatial','temporal','tod','dow','Location','MO','Weapon','AgeV']
    NB=naiveBayes(train,var,df=10,nbins=15)
    """
    var=var
    X=data[var]
    Y=data['Y']
    weights=data['wt']
    if any(weights)<0:
        raise ValueError('negative weights not allowed')
    NB=naiveBayesfit(X,Y,weights,partition=partition,df=df,nbins=nbins)
    return NB


def naiveBayesfit(X, y, weights=None, partition='quantile', df=20, nbins=30):
    """
    Direct call to naive bayes classifier.
    
    Parameters
    ----------
    X : [DataFrame] dataframe of the evidence variables of the crimes incident data.
    y : [array-like of shape] binary vector indicating linkage of crime pairs (1 = linked, 0 = unlinked).
    weights : vector of observation weights or the column name in data that corresponds to the weights (default weights=None).
    partition : [str] one of 'width' (fixed width) or 'quantile' (default) binning.
    df : [int] the effective degrees of freedom for the variables density estimates (default df=20).
    nbins : [int] number of bins.
    
    Returns
    ----------
    List of component bayes factors.
    """
    BF=[]
    var=list(X)
    nvar=len(var)
    if weights is None:
        weights=pd.DataFrame({'wt':np.repeat(1, len(y), axis=0)})
    for i in range(0,nvar):
        x=X[var[i]]
        if x.dtypes == 'float64':
            bks=makebreaks(x,mode=partition,nbins=nbins)
        BFi=getBF(x,y,weights,breaks=bks,df=df)
        BFi=BFi[0]
        BF.append(BFi)
    return BF


def plot_hcc(Z, labels, figsize=(15,8), **kwargs):
    """
    Plot a hierarchical crime clustering object.

    This function creates a dendrogram object and then plots it used log Bayes factor

    Parameters
    ----------
    Z : [array-like of shape] an object produced from function crimeClust_hier.
    labels : [array-like of shape] Crime IDs used to plot hierarchical crime clustering.
    figsize : [tuple: int] a method used to change the dimension of plot window, width, height in inches (default: figsize=(15,8)).
    **kwargs : arguments of the plotting functions from a collection matplotlib.pyplot.

    Returns
    ----------
    A dendrogram.
    """
    plt.figure(figsize=figsize,facecolor='w')
    plt.style.use('classic')
    plt.ylabel('log Bayes factor',fontsize=14)
    lab=list(np.round(list(np.arange(min(dhc), max(dhc), round(max(dhc)/10,1))),3))
    lab2=lab.copy()
    lab2.reverse()
    plt.yticks(lab2,lab)
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
    BF=BF.copy()
    varls=var[0:4]
    nnn=len(BF)
    nlen = int([nnn/2 if (nnn%2 ==0) else (nnn+1)/2][0])
    fig = plt.figure(figsize=figsize, facecolor='white')
    plt.style.use(plotstyle)
    for i in range(len(BF)):
        ax = fig.add_subplot(nlen, 2, i+1)
        BFi=BF[i]
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


def predictBF(BF, x, log=True):
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
    breaks=list(np.unique(BF.iloc[:, 0:2].values.flatten()))
    if breaks is not None:
        x=pd.cut(x, bins=breaks)
        x=x.to_frame(name='value')
    bf=pd.merge(x,BF,left_on=['value'],right_on=['value'], how='left').loc[:, ['BF']].fillna(1)
    if log == True:
        bf=np.log(bf)
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
    A=compareCrimes(allPairs,X,varlist=varlist)
    A1=A[A.columns[3:]]
    res2=gb.predict_proba(A1)
    Result=A[A.columns[0:2]]
    res3=pd.DataFrame({'link':res2[:, 1]})
    Result=pd.concat([Result, res3],axis=1).sort_values(by=['link'],ascending=False)
    return Result


def predictnaiveBayes(result, newdata, var, components=True, log=True):
    """
    Generate prediction (sum of log bayes factors) from a naiveBayes object.
    
    Parameters
    ----------
    result : [object] a naive bayes object from naiveBayes.
    newdata : [DataFrame] a dataframe of new predictors, column names must match NB names and var.
    var : [list, str] a list of the names or column numbers of specific predictors.
    components : [bool, default True] return the log bayes factors from each component (components=True) or return the sum of log bayes factors (components=False).
    log : [bool, default True] if log=True, return the log bayes factor estimate.
    
    Returns
    ----------
    Estimated (log) bayes factor from a single predictor.

    Notes
    ----------
    This does not include the log prior odds, so will be off by a constant.
    """
    var=var
    nvars=len(var)
    BF=pd.DataFrame(columns=var)
    for i in range(nvars):
        BFi=predictBF(result[i],newdata[var[i]],log=log)
        BF[var[i]] = BFi['BF']
    if components==True:
        BF=BF.sum(axis = 1, skipna = True)
    return BF


def predictNB_classes(result, newdata, var, components=True, log=True):
    """
    Generate prediction are labels from a naiveBayes object.
    
    Parameters
    ----------
    result : [object] a naive bayes object from naiveBayes.
    newdata : [DataFrame] a dataframe of new predictors, column names must match NB names and var.
    var : [list, str] a list of the names or column numbers of specific predictors.
    components : [bool, default True] return the log bayes factors from each component (default components=True) or return the sum of log bayes factors (components=False).
    log : [bool, default True] if log=True, return the log bayes factor estimate.
    
    Returns
    ----------
    The class of prediction.
    """
    var=var
    nvars=len(var)
    BF=pd.DataFrame(columns=var)
    for i in range(nvars):
        BFi=predictBF(result[i],newdata[var[i]],log=log)
        BF[var[i]] = BFi['BF']
    if components==True:
        BF=BF.sum(axis = 1, skipna = True)
    res=np.where(BF > 0,1,0)
    return res


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
    EvVar=compareCrimes(allPairsM, crimeData, varlist=varlist)
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


def seriesOffenderID(crime, unsolved, solved, seriesdata, varlist, estimateBF, n=10, groupmethod=3):
    """
    Crime series identification.

    Performs crime series identification by finding the crime series that are most closely related (as measured by Bayes Factor) to an unsolved crime

    Parameters
    ----------
    crime : [str] an crime ID that is in unsolved.
    unsolved : [DataFrame] incident data for the unsolved crimes. Must have a column named 'crimeID'.
    solved : [DataFrame] incident data for the solved crimes. Must have a column named 'crimeID'.
    seriesdata : [DataFrame] crime series data, generated from makeSeriesData.
    varlist : [dict] a list with elements named: crimeID, spatial, temporal and categorical. Each element should be a column names of crimedata corresponding to that feature: crimeID - crime ID for the crimedata that is matched to unsolved and solved, spatial - X,Y coordinates (in long and lat) of crimes, temporal - DT.FROM, DT.TO of crimes, categorical - categorical crime variables.
    estimateBF : [function] function to estimate the log bayes factor from evidence variables.
    n : [int] number of crimes to return.
    groupmethod : [int, default 3] method forms crimes groups: groupmethod=1 forms groups by finding the maximal connected offender subgraph. groupmethod=2 forms groups from the unique group of co-offenders. groupmethod=3 forms from groups from offenderIDs.

    Returns
    ----------
    Dataframe with three columnes: 'group' - indicating groups of solved crimes, 'BF-link' - Bayes Factor, 'offenderID' - an offender ID from data of solved crimes.

    Notes
    ----------
    Method=1 forms groups by finding the maximal connected offender subgraph. So if two offenders have ever co-offended, then all of their crimes are assigned to the same group. Method=2 forms groups from the unique group of co-offenders. So for two offenders who co-offended, all the co-offending crimes are in one group and any crimes committed individually or with other offenders are assigned to another group. Method=3 forms groups from the offender(s) responsible. So a crime that is committed by multiple people will be assigned to multiple groups.

    Example
    ----------
    seriesOffenderID('Crime3',UnsolvedData,Crimes,seriesData,varlist,estimateBF)
    """
    if crime in list(solved['crimeID']):
        raise ValueError("Error in unsolved crime ID")
    crimedata = unsolved[unsolved['crimeID'].isin([crime])]
    crimedata=pd.concat([crimedata,solved],ignore_index=True)
    pairs=pd.DataFrame({'i1':crimedata['crimeID'][0],'i2':crimedata['crimeID'][1:]})
    pairs=pd.DataFrame({'i1':crimedata['crimeID'][0],'i2':crimedata['crimeID'][1:]})
    X=compareCrimes(pairs,crimedata,varlist=varlist)
    bf=pd.DataFrame({'crimeID':X['i2'],'BF':estimateBF(X)})
    CG=makeGroups(seriesdata,method=groupmethod)
    SD=pd.DataFrame({'crimeID':seriesdata['crimeID'],'cl':CG['cl']})
    SDBF=pd.merge(SD, bf, left_on=['crimeID'], right_on=['crimeID'], how='left')
    SDBF.dropna(axis='columns')
    listcl=list(SDBF['cl'].unique())
    grdf_2  = pd.DataFrame(columns = ['group', 'average','single','complete'])
    for i in range(len(listcl)):
        grdf=SDBF[SDBF['cl']==listcl[i]]
        l_1,l_2,l_3,l_4=grdf['cl'].unique(),round(grdf['BF'].mean(),6),round(grdf['BF'].max(),6),round(grdf['BF'].min(),6)
        grdf_2.loc[i] = l_1,l_2,l_3,l_4
    Y=grdf_2.nlargest(n=n, columns='average', keep='all').reset_index(drop=True)
    data_types_dict = {'group': int}
    Y = Y.astype(data_types_dict)
    Y=pd.merge(Y, CG, left_on=['group'], right_on=['cl'], how='left').drop_duplicates(subset=['cl']).reset_index(drop=True)
    Y.drop(['cl'], axis='columns', inplace=True)
    Y.index += 1
    return Y


