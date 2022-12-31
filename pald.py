import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix

def df_to_dist(df):
    dff = pd.DataFrame(distance_matrix(df.values, df.values), index=df.index, columns=df.index)
    return dff
def cohesion_matrix(df):
    d = df_to_dist(df)
    n = d.shape[1]
    c = np.zeros((n,n))
    for x in range(0,n-1):
      for y in range(x+1,n):
        dx = d.iloc[:,x]
        dy = d.iloc[:,y]
        uxx = dx[dx <= d.iloc[y,x]].index
        uyy = dy[dy <= d.iloc[y,x]].index
        uxy = list(set().union(uxx,uyy))
        wx = 1 * (dx[uxy] < dy[uxy]) + .5 * ((dx[uxy] == dy[uxy]))
        c[x, uxy] = c[x, uxy] + 1 / (len(uxy)) * wx
        c[y, uxy] = c[y, uxy] + 1 / (len(uxy)) * (1 - wx)
    c = pd.DataFrame(c / (n - 1))
    c.index = d.index
    c.columns = d.index
    return(c)
def local_depth(c):
    return c.apply(np.sum,axis=1)

def strong_threshold(c):
    return np.mean(np.diag(c)/2)

def cohesion_strong(c,symmetric = True):
    threshold = strong_threshold(c)
    c[c < threshold] = 0
    if symmetric == True:
        c = np.minimum(c,c.transpose())
    return c


# ddf = pd.DataFrame((1,2,4.5,5,6))
# df = np.array([1,1,1,2,2,3,5,5,1,2,4])
# # print(df_to_dist(ddf))
# print(cohesion_matrix(ddf))
# print(local_depth(cohesion_matrix(ddf)))
# print(strong_threshold(cohesion_matrix(ddf)))
# print(cohesion_strong(cohesion_matrix(ddf)))

def index_pair(list1):
    result = []
    for i in range(len(list1)):
        for j in range(i,len(list1)):
            result.append([list1[i],list1[j]])
    return result
