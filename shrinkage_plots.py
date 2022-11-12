#!/usr/bin/env python3
#%% Imports and constants

import pickle

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from shrinkage_and_expansion import filter_low_coverage, log_probabilety
from shrinkage_and_expansion import calculate_ML_distance_per_position, calculate_ll_per_position

from utils import read_index_matrix, read_count_matrix, extract_np_from_df
from utils import count_matrix_to_probs, load_dist_list_from_file
from learn_rate import r_to_predicted_count

DATA_DIR = '/Users/yotamcon/hujiGD/CBIO-Project/data/'
SAMPLE = "Heart-Cardiomyocyte-Z0000044G"

#%% Load basic data

distances_list = load_dist_list_from_file(
    DATA_DIR + "hg19.CpG.bed.gz_distances.npy")

with open(DATA_DIR + SAMPLE + '/mats.pkl', 'rb') as f:
    mat = pickle.load(f)
    del f

with open(DATA_DIR + SAMPLE + '/rates_from_counts.pkl', 'rb') as f:
    learned = pickle.load(f)
    del f

#%% Load index matrices

# dataframe with transitions per CpG index
full_index_matrix = read_index_matrix(
    DATA_DIR + SAMPLE + ".2CpGs.pat.gz_index_matrix.tsv")

index_matrix = filter_low_coverage(full_index_matrix, 15)

# same dataframe, but as np.ndarray
index_2d = extract_np_from_df(index_matrix)


#%% unpickle probabilities

CATEGORY = 'all_genome'
# CATEGORY = 'chr1'

counts = mat[CATEGORY][0]
probs_df = count_matrix_to_probs(counts)
probs = extract_np_from_df(probs_df, dtype='float')
lp = log_probabilety(probs)

#%% learnt p(n) from alpha and beta
probs_l = r_to_predicted_count(learned[CATEGORY][59][0],probs_df['dist'])
# probs = probs_l

#%% exctract CpG islands for area of interest

with open(DATA_DIR + '/islands/islands_idx_chr1.txt') as f:
    islands = eval(f.read())
    del f

island_indices = []

for indices in islands[21:44]:
    island_indices.extend(indices)

del islands, indices

#%% learn area of interest

START_IDX = 11642
END_IDX = 16806

subindex_matrix = index_matrix.loc[START_IDX: END_IDX]
ll_per_position = calculate_ll_per_position(probs, subindex_matrix)
# ll_per_position = calculate_ll_per_position(probs_l, subindex_matrix)


#%% plot small group
start, stop = 100, 110

fig, ax = plt.subplots(1)
ax.plot(probs_df['dist'], ll_per_position[:,start:stop])

true_dist = distances_list[subindex_matrix.CpGindex[start:stop]]
ml_dist = ll_per_position[:,start:stop].argmax(axis=0) + 2

ax.plot(true_dist, ll_per_position[true_dist-2,range(start,stop)], 'b*')
ax.plot(ml_dist, ll_per_position[ml_dist-2,range(start,stop)], 'g^')

ax.legend([*[rf"$S_{ {index} }$" for index in subindex_matrix.CpGindex[start:stop]], r"$D_T(S_j)$", r"$D_{ML}(S_j)$"], bbox_to_anchor=(1.05, 1))

ax.set_xlim(0,202)
ax.set_ylim(-1.6, 0)


# plt.title("Likelihood of observing transition in site according to different distances")
plt.xlabel('Distance')
plt.ylabel('Log likelihood')
fig.savefig(DATA_DIR+'/../'+f'plots/Shrinkage/Likelihood per position by distances - {CATEGORY}.png', bbox_inches='tight', dpi=300)


#%% plot by genome position

fig = plt.figure(figsize=(20,4), constrained_layout=True)
gs = fig.add_gridspec(1,4)
ax1 = fig.add_subplot(gs[0,:3])
ax2 = fig.add_subplot(gs[0,3], sharey=ax1)

true_dist = distances_list[subindex_matrix.CpGindex]
ml_dist = ll_per_position.argmax(axis=0) + 2

ax1.plot(subindex_matrix.CpGindex, np.log((ml_dist)/true_dist), 'b.')
ax1.plot(island_indices, np.zeros_like(island_indices)-4.5, 'r*')
ax1.set_xlabel('CpG index')

ax2.hist(np.log((ml_dist)/true_dist), density=True,
         bins=50, orientation="horizontal", color='blue')
ax2.set_xlabel('Density')
ax2.set_xlim(0,0.35)

ax1.set_ylim(-5, 5)
ax1.set_ylabel('$LDR$')
ax2.get_yaxis().tick_right()

fig.suptitle('$LDR$ per position in the genome')

fig.savefig(DATA_DIR+'/../'+f'plots/Shrinkage/Ratio per position - {CATEGORY}.png', bbox_inches='tight', dpi=300)



#%% calculate normalized log likelihood

ll = calculate_ll_per_position(probs, index_matrix[:100])

plt.plot(ll[:,:10])
plt.legend(range(10))


#%% plot ml distanse vs true

start, stop = 000000, 15000000
true_dist = distances_list[index_matrix.CpGindex[start:stop]]
empirical_ml_dist = calculate_ML_distance_per_position(probs, index_matrix[start:stop])
learnt_ml_dist = calculate_ML_distance_per_position(probs_l, index_matrix[start:stop])

dist = distances_list[distances_list<200]
rnd_numbers = np.random.randint(2, 201, size=dist.shape[0])

#%% Plot accuracy histogram

# ml_dist = empirical_ml_dist
ml_dist = learnt_ml_dist

fig, ax = plt.subplots(1)
fig.set_size_inches(6,4)

ml_vs_true = np.histogram(abs(true_dist - ml_dist), density=True, bins=np.arange(-0.5,0.5+max(abs(true_dist - ml_dist))))

rand_uniform = np.histogram(abs(dist - rnd_numbers), density=True, bins=np.arange(-0.5,0.5+max(abs(dist - rnd_numbers))))

true_shuffled = true_dist.copy()
np.random.shuffle(true_shuffled)
rand_shuffle = np.histogram(abs(true_shuffled - true_dist), density=True, bins=np.arange(-0.5,0.5+max(abs(true_shuffled - true_dist))))

ml_hist = np.histogram(ml_dist, density=True, bins=np.arange(-0.5,0.5+max(abs(ml_dist))))

ax.plot(ml_vs_true[1][:-1], ml_vs_true[0], 'b.', label="$|D_{ML}-D_T|$")
ax.plot(rand_uniform[1][:-1], rand_uniform[0], 'orange', label="random uniform")
ax.plot(rand_shuffle[1][:-1], rand_shuffle[0], 'g-', label="random shuffle")
ax.plot(ml_hist[1][:-1][[ml_hist[0]>0.01]], ml_hist[0][ml_hist[0]>0.01]/300, 'r*', label="ML distances spikes")

ax.plot([np.mean(abs(true_dist - ml_dist))], [-0.002], color='blue',
        marker='v', linestyle='', label="Mean $|D_{ML}-D_T|$")
ax.plot([np.mean(abs(dist - rnd_numbers))], [-0.002], color='orange',
        marker='v', linestyle='', label="Mean random uniform")
ax.plot([np.mean(abs(true_shuffled - true_dist))], [-0.002], color='green',
        marker='v', linestyle='', label="Mean random shuffle")

# , linestyle='dashed', marker='o', markerfacecolor='blue'

ax.legend()

fig.savefig(DATA_DIR+'/../'+'plots/Shrinkage/accuracy distribution.png', bbox_inches='tight', dpi=300)


