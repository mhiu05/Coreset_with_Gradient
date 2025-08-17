import itertools
import os
import subprocess
import time
import gc

from nearpy import Engine
from nearpy.distances import EuclideanDistance
from nearpy.filters import NearestFilter
from nearpy.hashes import RandomBinaryProjections
import matplotlib.pyplot as plt
import numpy as np
from lazy_greedy import FacilityLocation, lazy_greedy_heap
import scipy.spatial
# from eucl_dist.cpu_dist import dist
# from eucl_dist.gpu_dist import dist as gdist

from multiprocessing.dummy import Pool as ThreadPool
from itertools import repeat
import sklearn


SEED = 100
EPS = 1E-8
PLOT_NAMES = ['lr', 'data_loss', 'epoch_loss', 'test_loss']  # 'cluster_compare', 'cosine_compare', 'euclidean_compare'

def similarity(X, metric):
    start = time.time()
    dists = sklearn.metrics.pairwise_distances(X, metric=metric, n_jobs=1)
    elapsed = time.time() - start

    if metric == 'cosine':
        S = 1 - dists
    elif metric == 'euclidean' or metric == 'l1':
        m = np.max(dists)
        S = m - dists
    else:
        raise ValueError(f'unknown metric: {metric}')
    return S, elapsed

def get_facility_location_submodular_order(S, B, c, smtk=0, no=0, stoch_greedy=0, weights=None):
    N = S.shape[0]
    no = smtk if no == 0 else no

    if smtk > 0:
        print(f'Calculating ordering with SMTK... part size: {len(S)}, B: {B}', flush=True)
        np.save(f'/tmp/{no}/{smtk}-{c}', S)
        if stoch_greedy > 0:
            p = subprocess.check_output(
                f'/tmp/{no}/smtk-master{smtk}/build/smraiz -sumsize {B} \
                 -stochastic-greedy -sg-epsilon {stoch_greedy} -flnpy /tmp/{no}/{smtk}-{c}.'
                f'npy -pnpv -porder -ptime'.split())
        else:
            p = subprocess.check_output(
                f'/tmp/{no}/smtk-master{smtk}/build/smraiz -sumsize {B} \
                             -flnpy /tmp/{no}/{smtk}-{c}.npy -pnpv -porder -ptime'.split())
        s = p.decode("utf-8")
        str, end = ['([', ',])']
        order = s[s.find(str) + len(str):s.rfind(end)].split(',')
        greedy_time = float(s[s.find('CPU') + 4 : s.find('s (User')])
        str = 'f(Solution) = '
        F_val = float(s[s.find(str) + len(str) : s.find('Summary Solution') - 1])
    else:
        V = list(range(N))
        start = time.time()
        F = FacilityLocation(S, V)
        order, _ = lazy_greedy_heap(F, V, B)
        greedy_time = time.time() - start
        F_val = 0

    order = np.asarray(order, dtype=np.int64)
    sz = np.zeros(B, dtype=np.float64)
    for i in range(N):
        if weights is None:
            sz[np.argmax(S[i, order])] += 1
        else:
            sz[np.argmax(S[i, order])] += weights[i]
    collected = gc.collect()
    return order, sz, greedy_time, F_val

def faciliy_location_order(c, X, y, metric, num_per_class, smtk, no, stoch_greedy, weights=None):
    class_indices = np.where(y == c)[0]
    print(c)
    print(class_indices)
    print(len(class_indices))
    S, S_time = similarity(X[class_indices], metric=metric)
    order, cluster_sz, greedy_time, F_val = get_facility_location_submodular_order(
        S, num_per_class, c, smtk, no, stoch_greedy, weights)
    return class_indices[order], cluster_sz, greedy_time, S_time

def get_orders_and_weights(B, X, metric, smtk, no=0, stoch_greedy=0, y=None, weights=None, equal_num=False, outdir='.'):
    N = X.shape[0]
    if y is None:
        y = np.zeros(N, dtype=np.int32)  # assign every point to the same class
    classes = np.unique(y)
    C = len(classes)  # number of classes

    if equal_num:
        class_nums = [sum(y == c) for c in classes]
        num_per_class = int(np.ceil(B / C)) * np.ones(len(classes), dtype=np.int32)
        minority = class_nums < np.ceil(B / C)
        if sum(minority) > 0:
            extra = sum([max(0, np.ceil(B / C) - class_nums[c]) for c in classes])
            for c in classes[~minority]:
                num_per_class[c] += int(np.ceil(extra / sum(minority)))
    else:
        num_per_class = np.int32(np.ceil(np.divide([sum(y == i) for i in classes], N) * B))
        print('not equal_num')

    order_mg_all, cluster_sizes_all, greedy_times, similarity_times = zip(*map(
        lambda c: faciliy_location_order(c, X, y, metric, num_per_class[c], smtk, no, stoch_greedy, weights), classes))

    order_mg, weights_mg = [], []
    if equal_num:
        props = np.rint([len(order_mg_all[i]) for i in range(len(order_mg_all))])
    else:
        # merging imbalanced classes
        class_ratios = np.divide([np.sum(y == i) for i in classes], N)
        props = np.rint(class_ratios / np.min(class_ratios))
        print(f'Selecting with ratios {np.array(class_ratios)}')
        print(f'Class proportions {np.array(props)}')

    order_mg_all = list(order_mg_all)
    cluster_sizes_all = list(cluster_sizes_all)
    for i in range(int(np.rint(np.max([len(order_mg_all[c]) / props[c] for c in classes])))):
        for c in classes:
            if isinstance(order_mg_all[c], (list, np.ndarray)):  # Kiểm tra kiểu dữ liệu
                ndx = slice(i * int(props[c]), int(min(len(order_mg_all[c]), (i + 1) * props[c])))
                order_mg = np.append(order_mg, order_mg_all[c][ndx])
                weights_mg = np.append(weights_mg, cluster_sizes_all[c][ndx])
    order_mg = np.array(order_mg, dtype=np.int32)

    weights_mg = np.array(weights_mg, dtype=np.float32)
    ordering_time = np.max(greedy_times)
    similarity_time = np.max(similarity_times)

    order_sz = []  # order_mg_all[rows_selector, cluster_order].flatten(order='F')
    weights_sz = [] # cluster_sizes_all[rows_selector, cluster_order].flatten(order='F')
    vals = order_mg, weights_mg, order_sz, weights_sz, ordering_time, similarity_time
    return vals