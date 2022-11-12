import numpy as np
from utils import CHR
import matplotlib.pyplot as plt
import re
from os import path, mkdir
import pickle
import pandas as pd
import seaborn as sns
from shrinkage_and_expansion import calculate_ll
CLR = plt.rcParams["axes.prop_cycle"].by_key()['color']

def ci(p, n):
    return np.sqrt(p * (1 - p) / n)


def plot_mat_by_col(mats, c, prob, ax=plt):
    mat = mats[c][1 if prob else 0]
    counts = mats[c][0]
    for i, col in enumerate(['M -> M', 'U -> M', 'M -> U', 'U -> U']):
        ax.scatter(mat['dist'], mat[col], s=5, label=col, c=CLR[i], zorder=-1)
        ax.errorbar(mat['dist'], mat[col], 1.96 * ci(mat[col].values, counts.loc[mat['dist'] - 2][col].values), fmt='.', alpha=0.5, c=CLR[i], zorder=-1)
        if col in ['M -> M', 'U -> U']:
            x = mat['dist']
            C = np.min(mat[col])
            N0 = np.max(mat[col]) - C
            dec = - np.log((mat[col].values[mat['dist'] == 50] - C) / N0) / 50
            print(col, dec)
            y = N0 * np.exp(-(x - 2) * dec) + C
            ax.plot(x, y, alpha=0.5, c='black', zorder=1)


def plot_p_all_cols(sample_name, mats, prob=True, c='all_genome', title='', by_chr=False, exclude_chr=None):
    if by_chr:
        for k in CHR:
            fig_name = c[len('all_'):] + f'_{k}'
            fig_title = f'{title}, {k}'
            k = k + (c[len('all'):] if c != 'all_genome' else '')
            if exclude_chr is None or k not in exclude_chr:
                plot_mat_by_col(mats, k, prob)
            plt.xlabel('n')
            plt.ylabel('p')
            plt.title(f'Empirical P(n), {fig_title}')
            plt.legend()
            plt.ylim((0, 1))
            print(f'plots/p_n/{sample_name}/all_transitions_{fig_name}.png')
            plt.savefig(f'plots/p_n/{sample_name}/all_transitions_{fig_name}.png')
            plt.clf()

    else:
        plot_mat_by_col(mats, c, prob)
        plt.xlabel('n')
        plt.ylabel('p')
        plt.title(f'Empirical P(n), {title}')
        plt.legend()
        plt.ylim((0, 1))
        plt.savefig(f'plots/p_n/{sample_name}/all_transitions_{c}.png')
        plt.clf()


def plot_p(sample_name, mats, col, prob=True, c='all_genome', title='', by_chr=False, exclude_chr=None):
    if by_chr:
        fig_name = c[len('all_'):] + '_by_chr'
        title = title + ' by chr'
        for k in CHR:
            if exclude_chr is None or k not in exclude_chr:
                k = k + (c[len('all'):] if c != 'all_genome' else '')
                mat = mats[k][1 if prob else 0]
                plt.scatter(mat['dist'], mat[col], s=5)
    else:
        fig_name = c
        mat = mats[c][1 if prob else 0]
        plt.scatter(mat['dist'], mat[col], s=5)
        counts = mats[c][0]
        plt.errorbar(mat['dist'], mat[col], 1.96 * ci(mat[col].values, counts.loc[mat['dist'] - 2][col].values),
                     fmt='.', alpha=0.5)

    plt.xlabel('n')
    plt.ylabel('p' if prob else 'count')
    plt.title(f'Empirical P(n)[{col}], ' + title)
    print(f'plots/p_n/{sample_name}/{col[0]}{col[-1]}_{fig_name}.png')
    plt.savefig(f'plots/p_n/{sample_name}/{col[0]}{col[-1]}_{fig_name}.png')
    plt.clf()


def plot_all_p_by_col(sample_name, mats, labels, prob=True):
    if not path.exists(f'plots/p_n/{sample_name}/'):
        mkdir(f'plots/p_n/{sample_name}/')
    for col in ['M -> M', 'U -> M', 'M -> U', 'U -> U']:
        for k, v in labels.items():
            plot_p(sample_name, mats, col, prob=prob, c=k, title=v, by_chr=False)
            plot_p(sample_name, mats, col, prob=prob, c=k, title=v,
                   by_chr=True, exclude_chr=['chrM'])


def plot_all_p(sample_name, mats, labels, prob=True):
    if not path.exists(f'plots/p_n/{sample_name}/'):
        mkdir(f'plots/p_n/{sample_name}/')
    for k, v in labels.items():
        plot_p_all_cols(sample_name, mats, prob=prob, c=k, title=v, by_chr=True)
        plot_p_all_cols(sample_name, mats, prob=prob, c=k, title=v, by_chr=False)


def plot_p_all_cols_3axis(sample_name, mats, labels, chr=''):
    fig = plt.figure(figsize=(12, 4))
    axs = fig.subplots(1, 3, gridspec_kw={'width_ratios': [2, 2, 1.2]})
    for i, (c, title) in enumerate(labels.items()):
        plot_mat_by_col(mats, c, prob=True, ax=axs[i])
        axs[i].set_ylim((0, 1))
        axs[i].set_xlabel('n')
        axs[i].set_ylabel('p')
        axs[i].set_title(f'{title}')
    fig.suptitle(f'Empirical P(n), Sample {sample_name}' + (', ' + chr if len(chr) > 0 else ''))
    axs[1].legend(loc='upper right')
    plt.tight_layout()
    if len(chr) == 0: plt.savefig(f'plots/p_n/{sample_name}/all_transitions.png', dpi=300)
    else: plt.savefig(f'plots/p_n/{sample_name}/all_transitions_{chr}.png', dpi=300)
    plt.clf()


def get_mats(sample_dir, thr=0):
    with open(sample_dir + 'mats.pkl', 'rb') as handle:
        mats = pickle.load(handle)
    # remove prob rows with less then thr samples
    if thr > 0:
        for k, v in mats.items():
            mat = v[1]
            mat_counts = v[0].loc[mat.dist - 2]
            mat = mat.loc[mat_counts[mat_counts.columns[1:]].sum(axis=1) > thr]
            mats[k] = (v[0], mat)
    return mats


def plot_p_by_idx(index_mat_path, idx_start, idx_end, fig_name):
    df = pd.read_csv(index_mat_path, header=0, sep='\t',
                     names=["CpGindex", "M -> M", "M -> U", "U -> M", "U -> U"])
    df_temp = df[idx_start:idx_end]
    div = df_temp[["M -> M", "M -> U"]] / df_temp[["M -> M", "M -> U"]].sum(axis=1)[:, None]
    div2 = df_temp[["U -> M", "U -> U"]] / df_temp[["U -> M", "U -> U"]].sum(axis=1)[:, None]
    temp = np.logical_not(np.logical_or(np.isnan(div["M -> M"]), np.isnan(div["M -> U"])))
    temp2 = np.logical_not(np.logical_or(np.isnan(div2["U -> U"]), np.isnan(div2["U -> M"])))
    temp = np.logical_and(temp, temp2)
    df2 = pd.DataFrame(df_temp["CpGindex"].loc[temp], columns=["CpGindex"])
    df2[["M -> M", "M -> U"]] = div.loc[temp]
    df2[["U -> M", "U -> U"]] = div2.loc[temp]
    fig = plt.figure(figsize=(20, 8))
    axs = fig.subplots(2, 1)

    axs[0].scatter(df2["CpGindex"], df2["M -> M"], s=5)
    axs[1].scatter(df2["CpGindex"], df2["U -> U"], s=5)
    axs[0].set_ylabel('P[M->M]')
    axs[1].set_ylabel('P[U->U]')
    axs[1].set_xlabel('CpG index')
    axs[0].set_title('P[M->M] for pairs along the genome')
    axs[1].set_title('P[U->U] for pairs along the genome')
    plt.tight_layout()
    plt.savefig(f'plots/prob_along_genome_{fig_name}.png', dpi=300)
    plt.show()


def ll_heatmap(count_mats, prob_mats, labels):
    plt.figure(figsize=(10, 6))

    ll_scores = np.array([[calculate_ll(prob, count) for prob in prob_mats] for count in count_mats])
    # print(ll_scores)
    # means = np.nanmean(ll_scores, axis=1)
    # stds = np.nanstd(ll_scores, axis=1)
    # plt.scatter(np.arange(len(means)), means)
    # plt.errorbar(np.arange(len(means)), means, stds, fmt='.', alpha=0.5)
    # plt.ylabel('LL')
    # plt.xticks(np.arange(len(means)), labels, rotation='vertical')
    # plt.show()

    sns.heatmap(ll_scores, xticklabels=labels, yticklabels=labels, annot=True)
    plt.xlabel('count matrix')
    plt.ylabel('prob matrix')
    plt.title('LL score (normalized) of genome section by prob matrix')
    plt.show()
    c=2



if __name__ == '__main__':
    samples = ['data/Heart-Cardiomyocyte-Z0000044G.2CpGs.pat.gz_index_matrix.tsv',
               'data/Heart-Cardiomyocyte-Z0000044K.2CpGs.pat.gz_index_matrix.tsv',
               'data/Heart-Cardiomyocyte-Z0000044Q.2CpGs.pat.gz_index_matrix.tsv',
               'data/Liver-Hepatocytes-Z0000043Q.2CpGs.pat.gz_index_matrix.tsv',
               'data/Liver-Hepatocytes-Z0000044H.2CpGs.pat.gz_index_matrix.tsv',
               'data/Liver-Hepatocytes-Z0000044M.2CpGs.pat.gz_index_matrix.tsv']
    dict_path = "data/CpG.bed.gz"
    islands_path = "data/hg19.CpG-islands.bed.gz"
    labels_all = {'all_genome': 'All genome', 'all_outside_islands': 'Outside CpG islands', 'all_islands': 'Inside CpG islands'}
    labels_chr = lambda c: {c: 'All genome', c + '_outside_islands': 'Outside CpG islands', c + '_islands': 'Inside CpG islands'}
    # labels2 = {c: c for c in CHR}
    # labels3 = {c + '_islands': 'inside CpG islands, ' + c for c in CHR}
    # labels4 = {c + '_outside_islands': 'outside CpG islands, ' + c for c in CHR}
    # labels = {**labels1, **labels2, **labels3, **labels4}


    # labels_chr_heatmat = lambda c_arr: {'all_genome': 'All genome', **{c: c for c in c_arr}}
    #                                     # 'all_outside_islands': 'Outside islands',
    #                                     # **{c + '_outside_islands': 'Outside, ' + c for c in c_arr},
    #                                     # 'all_islands': 'Inside islands',
    #                                     # **{c + '_islands': 'Inside, ' + c for c in c_arr}}
    #
    # sample_name = re.match(r'data/(.*).2CpGs.pat.gz_index_matrix.tsv', samples[4]).group(1)
    # sample_dir = f'data/{sample_name}/'
    # mats = get_mats(sample_dir, thr=1000)
    # # ll_heatmap(mats, labels_chr_heatmat(['chr1', 'chr20', 'chrY']))
    # prob_mats, count_mats, labels = [], [], []
    # for i, (c, title) in enumerate(labels_chr_heatmat(CHR).items()):
    #     count_mats.append(mats[c][0])
    #     prob_mats.append(mats[c][1])
    #     labels.append(title)
    # # ll_heatmap(count_mats, prob_mats, labels)
    #
    # prob_mats, count_mats, labels = [], [], []
    # for i in range(len(samples)):
    #     sample_name = re.match(r'data/(.*).2CpGs.pat.gz_index_matrix.tsv', samples[i]).group(1)
    #     sample_dir = f'data/{sample_name}/'
    #     mats = get_mats(sample_dir, thr=1000)
    #     for i, (c, title) in enumerate(labels_all.items()):
    #         count_mats.append(mats[c][0])
    #         prob_mats.append(mats[c][1])
    #         labels.append(sample_name)
    # ll_heatmap(count_mats, prob_mats, labels)


    for i in range(len(samples)):
        sample_name = re.match(r'data/(.*).2CpGs.pat.gz_index_matrix.tsv', samples[i]).group(1)
        sample_dir = f'data/{sample_name}/'
        mats = get_mats(sample_dir, thr=1000)
        print(sample_name)
        # plot_all_p(sample_name, mats, labels)
        # plot_all_p(sample_name, mats, labels)
        # for chr in CHR:
        #     plot_p_all_cols_3axis(sample_name, mats, labels_chr(chr), chr=chr)
        plot_p_all_cols_3axis(sample_name, mats, labels_all)

    # index_mat_path = "data/Heart-Cardiomyocyte-Z0000044G.2CpGs.pat.gz_index_matrix.tsv"
    # plot_p_by_idx(index_mat_path, idx_start=4000, idx_end=9000, fig_name='Heart-Cardiomyocyte-Z0000044G_4000_9000')


    c=0