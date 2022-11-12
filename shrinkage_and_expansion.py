#!/usr/bin/env python3

import pandas as pd
import numpy as np

from utils import  extract_np_from_df


def log_probability(probs: np.ndarray) -> np.ndarray:
    return np.log(probs)


def calculate_ll_per_position(probs: np.ndarray, index_matrix: pd.DataFrame):
    '''calculates the normalised likelihood of getting each row
    of index_matrix according to probs'''
    lp = log_probability(probs)
    index_2d = extract_np_from_df(index_matrix)

    ll = np.array([], dtype=np.int64).reshape(199,0)

    for chunk in np.array_split(index_2d, max(1,(index_2d.shape[0]/200000))):
        inflate_index = np.tile(chunk.T,(lp.shape[0],1,1))
        inflate_lp = (np.tile(lp.T, (chunk.shape[0],1,1))).T

        ll_partial = (inflate_index * inflate_lp).sum(1)
        ll_partial_norm = ll_partial/(chunk.sum(axis=1, keepdims=True).T)
        ll = np.append(ll, ll_partial_norm, axis=1)
    return ll


def calculate_ML_distance_per_position(probs: np.ndarray, index_matrix: pd.DataFrame):
    '''calculates the ML distance of index_matrix according to probs'''
    lp = log_probability(probs)
    index_2d = extract_np_from_df(index_matrix)

    ml_distances = np.array([], dtype=np.int64).reshape(0,)

    for chunk in np.array_split(index_2d, max(1,(index_2d.shape[0]/200000))):
        inflate_index = np.tile(chunk.T,(lp.shape[0],1,1))
        inflate_lp = (np.tile(lp.T, (chunk.shape[0],1,1))).T

        ll_partial = (inflate_index * inflate_lp).sum(1)
        distances = ll_partial.argmax(axis=0) + 2

        ml_distances = np.append(ml_distances, distances)
    return ml_distances


def calculate_ll(probs: pd.DataFrame, count_matrix: pd.DataFrame):
    '''calculates the likelihood of getting the count_matrix
    according to matching distances in the probs'''
    count_matrix = count_matrix.loc[probs['dist'] - 2][['M -> M', 'U -> M', 'M -> U', 'U -> U']]
    probs = probs[['M -> M', 'U -> M', 'M -> U', 'U -> U']].values
    lp = log_probability(probs)
    counts_2d = extract_np_from_df(count_matrix)

    return np.sum(counts_2d*lp)/np.sum(counts_2d)


def filter_low_coverage(index_matrix: pd.DataFrame, threshold=15):
    index_2d = extract_np_from_df(index_matrix)
    return index_matrix[index_2d.sum(axis=1)>threshold]
