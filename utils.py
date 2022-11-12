import numpy as np
import pandas as pd
import re
from typing import Tuple
from os import path, mkdir
import warnings
import pickle

DIST_COL = 'dist'
MAX_DIST = 200
CHR = ['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7',
       'chr8', 'chr9', 'chr10', 'chr11', 'chr12', 'chr13', 'chr14',
       'chr15', 'chr16', 'chr17', 'chr18', 'chr19', 'chr20', 'chr21', 'chr22',
       'chrM', 'chrX', 'chrY']


def read_count_matrix(path: str) -> pd.DataFrame:
    """
    Read the count matrix from TSV file
    Count matrix of the form -

    dist | M -> M | M -> U | U -> M | U -> U
    ...  |   ...  |   ...  |   ...  |  ...

    Parameters
    ----------
    path : str
        path to tsv

    Returns
    -------
    pd.DataFrame
        The count matrix
    """
    return pd.read_csv(path, sep='\t', index_col=[0])


def read_index_matrix(path: str) -> pd.DataFrame:
    """
    Read the index matrix from TSV file
    Count matrix of the form -

    CpGindex | M -> M | M -> U | U -> M | U -> U
    ...  |   ...  |   ...  |   ...  |  ...

    Parameters
    ----------
    path : str
        path to tsv

    Returns
    -------
    pd.DataFrame
        The index matrix
    """
    return pd.read_csv(path, sep='\t', index_col=[0])


def count_matrix_to_probs(count_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize each pair of columns to a probablity distribution.

    Parameters
    ----------
    count_matrix : pd.DataFrame
    """
    def norm_row(row):
        s = row.sum()
        if s > 0:
            return row / s
        return 0
    # TODO: division by zero here!
    return pd.concat(axis=1, objs=[
        count_matrix['dist'],
        count_matrix.apply(lambda x: norm_row(x[1:3]), axis=1),
        count_matrix.apply(lambda x: norm_row(x[3:]), axis=1)])


# Functions handeling CpG dictionaries and distances lists
def read_CpG_dict(dict_path: str) -> np.ndarray:
    '''
    Given a gzipped CpG dictionary, create an ndarray of distances.
    The distances are between the CpG pairs, where list[1] is the
    distance between the CpG indexed 1 and the CpG with index 2.
    Three exception types:
    1. Index 0 is irrelevant (-1)
    2. Jumps between chromosomes are negative numbers (-1)
    3. the centromers cause large gaps (huge numbers)
    Using the list for proper reads will never be troubled by that
    '''
    df = pd.read_csv(dict_path, sep='\t', compression='gzip',
                     names=["chromosome", "position", "index"])
    df["index"] -= 1
    d_list = df.position.to_numpy() - np.roll(df.position.to_numpy(), 1)
    d_list[d_list < 0] = -1
    return d_list


def save_dist_list_to_file(dists_list, path):
    np.save(path, dists_list)


def load_dist_list_from_file(path) -> np.ndarray:
    return np.load(path)


def sum_counts_for_indices(
        index_matrix: pd.DataFrame, indices, distances_list: np.ndarray
        ) -> np.ndarray:
    '''
    create Counts(N) from index matrix and indices list

    Parameters
    ----------
    index_matrix: pd.DataFrame
        the full index matrix

    indices: iterable
        the indices to obtain the data from

    distances_list: np.ndarray
        the distances of the different CpG from one another


    Returns
    -------
    pd.DataFrame
        The count matrix for the relevant indices
    '''
    sliced = index_matrix[index_matrix.CpGindex.isin(indices)]
    sliced.insert(0, 'dist', distances_list[indices])
    del sliced['CpGindex']

    summed = sliced.groupby('dist').sum()
    summed.reset_index(level=0, inplace=True) # turn dist into a column

    full = pd.DataFrame(
        data=np.c_[2:MAX_DIST+1],dtype='int64',
        columns = ["dist"])

    full = full.merge(summed, how='left', on='dist')

    return full.replace(np.NaN, 0).astype('int')


def extract_np_from_df(df_matrix: pd.DataFrame, dtype='int') -> np.ndarray:
    return df_matrix[["M -> M","M -> U","U -> M","U -> U"]].to_numpy(dtype=dtype)


def get_island_idx(dict_df, islands_path):
    """
    Get CpG indices that are inside CpG island.
    Parameters
    ----------
    dict_df
    islands_path

    Returns
    -------
    list of lists: for each chromosome, a list of the indices for this chr.
    """
    islands_df = pd.read_csv(islands_path, sep='\t', compression='gzip', names=["chromosome", "start", "end"])
    islands = []
    for c in CHR:
        if path.exists(f'data/islands/islands_idx_{c}.txt'):
            with open(f'data/islands/islands_idx_{c}.txt') as f:
                islands.append(eval(f.read()))
        else:
            dict_df_c = dict_df[dict_df.chromosome == c]
            islands_df_c = islands_df[islands_df.chromosome == c]
            idx_ = np.logical_and(dict_df_c.position[:, None] >= islands_df_c.start.values,
                                  dict_df_c.position[:, None] < islands_df_c.end.values)
            islands_idx = ([list(dict_df_c[idx_[:, i]].index.values) for i in range(idx_.shape[1])])
            with open(f'data/islands/islands_idx_{c}.txt', 'w') as f:
                f.write(repr(islands_idx))
            islands.append(islands_idx)
    return islands


def get_count_mat(df) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns count_mat and prob_mat from df.
    Parameters
    ----------
    df : part of index matrix (columns=['dist', 'M -> M', 'M -> U', 'U -> M', 'U -> U'])

    Returns
    -------
    count_mat, prob_mat
    """
    count_mat = np.array([df[df.dist == n].sum().values[2:] for n in range(2, MAX_DIST + 1)], dtype='int32')
    prob_mat = np.array(np.copy(count_mat), dtype='float64')
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prob_mat[:, :2] = prob_mat[:, :2] / np.sum(prob_mat[:, :2], axis=1)[:, None]
        prob_mat[:, 2:] = prob_mat[:, 2:] / np.sum(prob_mat[:, 2:], axis=1)[:, None]

    count_mat = pd.DataFrame(count_mat, columns=['M -> M', 'M -> U', 'U -> M', 'U -> U'])
    count_mat.insert(0, 'dist', np.arange(2, MAX_DIST + 1))
    prob_mat = pd.DataFrame(prob_mat, columns=['M -> M', 'M -> U', 'U -> M', 'U -> U'])
    prob_mat.insert(0, 'dist', np.arange(2, MAX_DIST + 1))
    return count_mat, prob_mat.dropna()


def get_prob_from_index(index_mat_path, dict_path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    From index matrix returns counts and probability matrices.
    Parameters
    ----------
    index_mat_path
    dict_path : path to CpG indices

    Returns
    -------
    count_mat, prob_mat
    """
    df = pd.read_csv(index_mat_path, header=0, sep='\t',
                     names=["CpGindex", "M -> M", "M -> U", "U -> M", "U -> U"])
    dists = read_CpG_dict(dict_path)
    df.insert(1, 'dist', dists[df.CpGindex])
    return get_count_mat(df)


def get_prob_from_index_by_chr(index_mat_path, dict_path):
    """
    From index matrix returns counts and probability matrices for each chromosome.
    Parameters
    ----------
    index_mat_path
    dict_path : path to CpG indices

    Returns
    -------
    dict, 'chr{i}': (count_mat, prob_mat)
    """
    df = pd.read_csv(index_mat_path, header=0, sep='\t',
                     names=["CpGindex", "M -> M", "M -> U", "U -> M", "U -> U"])
    dists_df = pd.read_csv(dict_path, sep='\t', compression='gzip',
                     names=["chromosome", "position", "index"])

    dists_df["index"] -= 1
    dists = dists_df.position.to_numpy() - np.roll(dists_df.position.to_numpy(), 1)
    dists[dists < 0] = -1
    df.insert(1, 'dist', dists[df.CpGindex])

    mats = {c: get_count_mat(df[dists_df.chromosome[df.CpGindex] == c]) for c in CHR}
    return mats


def get_prob_islands(index_mat_path, dict_path, islands_path, per_chr=False):
    """
    From index matrix returns counts and probability matrices for indices inside CpG islands.
    Parameters
    ----------
    index_mat_path
    dict_path : path to CpG indices
    islands_path: path to islands indices

    Returns
    -------
    dict, 'chr{i}': (count_mat, prob_mat)
    """
    df = pd.read_csv(index_mat_path, header=0, sep='\t',
                     names=["CpGindex", "M -> M", "M -> U", "U -> M", "U -> U"])
    dists_df = pd.read_csv(dict_path, sep='\t', compression='gzip',
                     names=["chromosome", "position", "index"])
    dists_df["index"] -= 1
    dists = dists_df.position.to_numpy() - np.roll(dists_df.position.to_numpy(), 1)
    dists[dists < 0] = -1
    df.insert(1, 'dist', dists[df.CpGindex])
    island_idx = get_island_idx(dists_df, islands_path)
    idx_in = [item for sublist in [x for sublist_c in island_idx for x in sublist_c] for item in sublist if item in df.CpGindex]
    temp = np.array([True] * (np.max(df.CpGindex) + 1))
    temp[idx_in] = False
    mats = {'all_islands': get_count_mat(df.loc[idx_in]), 'all_outside_islands': get_count_mat(df[temp[df.CpGindex]])}
    if per_chr:
        for i, c in enumerate(CHR):
            df_c = df[dists_df.chromosome[df.CpGindex] == c]
            idx_c = [item for sublist in island_idx[i] for item in sublist if item in df_c.CpGindex]
            mats[c + '_islands'] = get_count_mat(df_c.loc[idx_c])
            idx_c = np.array([item for item in df_c.CpGindex if item not in island_idx[i]])
            mats[c + '_outside_islands'] = get_count_mat(df_c.loc[idx_c])
    return mats


def save_p(index_mat_path, dict_path, islands_path):
    """
    Calculate and save all mats
    Parameters
    ----------
    index_mat_path
    dict_path : path to CpG indices
    islands_path: path to islands indices
    """
    sample_name = re.match(r'data/(.*).2CpGs.pat.gz_index_matrix.tsv', index_mat_path).group(1)
    sample_dir = f'data/{sample_name}/'
    if not path.exists(sample_dir):
        mkdir(sample_dir)
    if not path.exists(sample_dir + 'mats.pkl'):
        temp1 = {'all_genome': get_prob_from_index(index_mat_path, dict_path)}
        temp2 = get_prob_from_index_by_chr(index_mat_path, dict_path)
        temp3 = get_prob_islands(index_mat_path, dict_path, islands_path, per_chr=True)
        mats = {**temp1, **temp2, **temp3}
        with open(sample_dir + 'mats.pkl', 'wb') as handle:
            pickle.dump(mats, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    index_mat_path = "data/Heart-Cardiomyocyte-Z0000044G.2CpGs.pat.gz_index_matrix.tsv"
    dict_path = "data/CpG.bed.gz"
    path_islands = "data/hg19.CpG-islands.bed.gz"
    mats = get_prob_islands(index_mat_path, dict_path, path_islands, per_chr=True)
    mats2 = get_prob_islands(index_mat_path, dict_path, path_islands, per_chr=False)

    c=0





