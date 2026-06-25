import os
import numpy as np
import pandas as pd
from collections import Counter
import random
from election_model import Election

box_path = '/Users/chanuwasaswamenakul/Library/CloudStorage/Box-Box'
random.seed(1012)

alpha_range = np.linspace(1,3,num=21)

fixed_params = {'N': 5000,
                'nom_rate': 12,
                'rep_num': 12,
                'party_num': 2,
                'party_sd': 0.1,
                'party_loc': [-0.66, 0.66],
                'district_num': 1,
                'elect_system': 'proportional_rep',
                'voting': 'deterministic',
                'opinion_dist_dict': {'dist': "beta", 'a': 1, 'b': 1},
                'ideo_sort': 0,
                'alpha': 0,
                'beta': 0,
                'ps': 0.1}

n_sim = 25
n_iter = 20
print_interval = 10
# max_js_distance = 0.8325546111576977

all_step_results = []

for i in range(len(alpha_range)):
    alpha = alpha_range[i]
    fixed_params['opinion_dist_dict']['a'] = alpha

    print('alpha =', alpha)

    # iterate over simulation run
    sim_dist_list = []
    for j in range(n_sim):
        if j % print_interval == 0:
            print('sim: {}'.format(j))

        simple_election = Election(**fixed_params)
        close_dist_list = []

        # initialize districts and residents
        resident_opis = []
        for k in range(fixed_params['district_num']):
            district = simple_election.districts[k]
            resident_opis.extend([resident.x for resident in district.residents])

        resident_opis = np.array(resident_opis)

        # iterate over electoral cycles
        for k in range(n_iter):
            if k % print_interval == 20:
                print('iteration: {}'.format(k))
            simple_election.step()

            elected_opis = np.array([elected.x for elected in simple_election.elected_pool])

            # distance to closest elected candidate in an election
            close_dists = np.min(np.abs(np.subtract.outer(resident_opis, elected_opis)), axis=1)
            close_dist_list.append(close_dists)

        avg_dist = np.mean(close_dist_list) # average dissimilarity over all electoral cycles
        sim_dist_list.append(avg_dist)

    close_dist_df = pd.DataFrame({'avg_dist': sim_dist_list})
    close_dist_df['a'] = alpha

    all_step_results.append(close_dist_df)

all_step_results = pd.concat(all_step_results).reset_index(drop=True)
print('simulation is complete.')

biased_repr_file = f'elected_biased_representation_{fixed_params["elect_system"]}.csv'
all_step_results.to_csv(os.path.join(box_path, 'ComplexElection', 'results', biased_repr_file),
                        index=False)
