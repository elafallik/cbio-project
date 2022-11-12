import os.path

import pandas as pd
import numpy as np
from typing import Tuple
from utils import MAX_DIST, read_CpG_dict, save_dist_list_to_file, load_dist_list_from_file

indexer = "CCTTC"  # instead of a switch
chunksize = 10 ** 5  # for IO limitation


def analyze_pat_row(row, distances_list, count_2d, count_per_index):
    if len(row.pattern)<2:
        return
    for position in range(len(row.pattern)-1):
        # get the pattern's type (CC,CT,TT,TC)
        pattern  = row.pattern[position:position+2]
        if (pattern.find(".") > -1):
            continue
        ind = indexer.index(pattern)
        # get the distance between the CpG indices
        distance = distances_list[row.CpGindex + position] -2 # -2 for min
        if (distance > count_2d.shape[0]-1):
            continue
        # update the count matrices
        count_per_index[row.CpGindex + position, ind] += row.multiplicity
        count_2d[distance, ind] += row.multiplicity


def analyze_pat(patfile, distances_list, count_2d, count_per_index):
    for chunk in pd.read_csv(
            patfile, sep='\t', chunksize=chunksize, compression='gzip',
            names=["chromosome", "CpGindex", "pattern", "multiplicity"]):
        for row in chunk.itertuples():
            analyze_pat_row(row, distances_list, count_2d, count_per_index)

def build_count_2d(max_distance = MAX_DIST) -> np.ndarray:
    '''
    creates a numpy 2d array, with max_distance rows and four columns:
        [M -> M , M -> U , U -> U , U -> M]
    '''
    return np.zeros((max_distance-1, 4), dtype=np.int32)


def build_count_per_index(number_of_indices) -> np.ndarray:
    '''
    creates a numpy 2d array, with number_of_indices rows and four columns:
        [M -> M , M -> U , U -> U , U -> M]
    '''
    return np.zeros((number_of_indices, 4), dtype=np.int32)


def count_2d_to_count_matrix(count_2d: np.ndarray) -> pd.DataFrame:
    """
    Save the count_2d in a DataFrame of the form - 

    dist | M -> M | M -> U | U -> M | U -> U
    ...  |   ...  |   ...  |   ...  |  ...

    Parameters
    ----------
    count_2d : np.ndarray
        the counts 2d array

    Returns
    -------
    pd.DataFrame
        The count matrix
    """
    return pd.DataFrame(data=np.c_[2:MAX_DIST+1,count_2d[:,[0,1,3,2]]],
                 columns = ["dist","M -> M","M -> U","U -> M","U -> U"])


def count_per_index_to_index_matrix(count_per_index: np.ndarray) -> pd.DataFrame:
    """
    Save the count_per_index in a DataFrame of the form - 

    CpGindex | M -> M | M -> U | U -> M | U -> U
      ...    |   ...  |   ...  |   ...  |  ...

    Parameters
    ----------
    count_per_index : np.ndarray
        the count per index array

    Returns
    -------
    pd.DataFrame
        The index matrix. zero rows are removed
    """
    df = pd.DataFrame(
        data=np.c_[0:count_per_index.shape[0],count_per_index[:,[0,1,3,2]]],
        columns = ["CpGindex","M -> M","M -> U","U -> M","U -> U"])
    return df[count_per_index.sum(axis=1)>0]


def save_matrix(path: str, martix):
    """
    Save the count matrix to a tsv file
    
    Parameters
    ----------
    path : str
        path to save csv to.

    count_martix : pd.DataFrame
        The count matrix
    """
    martix.to_csv(path, sep='\t')


def process_pat(path_to_pat, path_to_dictionary) -> Tuple[pd.DataFrame, pd.DataFrame]:
    '''
    '''
    # obtaining CpG distances list
    if os.path.isfile(path_to_dictionary + "_distances.npy"):
        distances_list = load_dist_list_from_file(path_to_dictionary + "_distances.npy")
    else:
        distances_list = read_CpG_dict(path_to_dictionary)
        save_dist_list_to_file(distances_list, path_to_dictionary + "_distances.npy")
    
    count_2d = build_count_2d()
    count_per_index = build_count_per_index(len(distances_list))
    analyze_pat(path_to_pat, distances_list, count_2d, count_per_index)
    
    return count_2d_to_count_matrix(count_2d), count_per_index_to_index_matrix(count_per_index)

# # temp for debugging
# path_full_dict = "data/hg19.CpG.bed.gz"
# path_list = "data/hg19.CpG.bed.gz_distances.npy"
# path_pat = "data/Cardiomyocyte-44G.chr1.pat.gz"


# if __name__ == '__main__':
#     count_matrix, index_matrix = process_pat(path_pat,path_full_dict)
#     # save_matrix(path_pat + "_count_matrix.tsv", count_matrix)
#     # save_matrix(path_pat + "_index_matrix.tsv", index_matrix)

