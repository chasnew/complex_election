import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import distance
from election_model import Election

fixed_params = {'N': 2000,
                'nom_rate': 0.05,
                'rep_num': 10,
                'party_num': 5,
                'district_num': 2,
                'voting': 'proportional_rep',
                'opinion_distribution': 'uniform',
                'elect_fb': True}

n_sim = 1
n_iter = 5000
print_interval = 500
max_js_distance = 0.8325546111576977

simple_election = Election(**fixed_params)
trust_list = []

for i in range(n_iter):
    if i % print_interval == 0:
        print('iteration: {}'.format(i))
    simple_election.step()

    if fixed_params['elect_fb']:
        tmp_list = []
        for district in simple_election.districts:
            tmp_list.extend([resident.trust for resident in district.residents])

        trust_list.append(tmp_list)

print('simulation is complete.')
trust_array = np.array(trust_list)
sns.heatmap(trust_array)
plt.show()

elected_opis = np.array([elected.x for elected in simple_election.cum_elected_pool])
resident_opis = []
for i in range(fixed_params['district_num']):
    district = simple_election.districts[i]
    resident_opis.extend([resident.x for resident in district.residents])

resident_opis = np.array(resident_opis)

res_hists = np.histogram(resident_opis, bins=100, range=(-1, 1))[0]
res_hists = res_hists / res_hists.sum()
elect_hists = np.histogram(elected_opis, bins=100, range=(-1, 1))[0]
elect_hists = elect_hists / elect_hists.sum()

js_distance = distance.jensenshannon(res_hists, elect_hists)


# plt.style.use('seaborn-v0_8-colorblind')
fig, ax = plt.subplots(figsize=(12,7))

sns.histplot(elected_opis, stat='probability',
                  bins=np.linspace(-1,1,100), alpha=0.5, ax=ax)
ax.legend(labels=['elected_candidates'])
sns.histplot(resident_opis, stat='probability',
             bins=np.linspace(-1,1,100), alpha=0.5, ax=ax)
ax.set_xlim(-1,1)
# ax.set_ylim(0,0.023)
# ax.set_xticks([])
# ax.set_yticks([])
# ax.set_ylabel('')
# ax.get_legend().remove()
ax.legend(['elected candidates', '_', 'residents'], fontsize=15, framealpha=0)
plt.title('Distributions of representatives and residents opinions (d = {}, N = {})'.format(fixed_params['district_num'],
                                                                                            fixed_params['N']))

# plt.savefig('results/cogs222_res5k_party1.png', transparent=True)

if (fixed_params['voting'] != 'prop_rep'):
    if (fixed_params['party_num'] > 0):
        if (fixed_params['district_num'] > 1):
            plt.savefig('results/reps{}_vs_res10k_party{}_d{}.png'.format(fixed_params['rep_num'],
                                                                         fixed_params['party_num'],
                                                                         fixed_params['district_num']),
                        transparent=True)
        else:
            plt.savefig('results/reps{}_vs_res10k_party{}.png'.format(fixed_params['rep_num'],
                                                                     fixed_params['party_num']),
                        transparent=True)
    else:
        if (fixed_params['district_num'] > 1):
            plt.savefig('results/reps{}_vs_res10k_d{}_basic.png'.format(fixed_params['rep_num'],
                                                                       fixed_params['district_num']),
                        transparent=True)
        else:
            plt.savefig('results/reps{}_vs_res10k_basic.png'.format(fixed_params['rep_num']),
                        transparent=True)
else:
    plt.savefig('results/reps{}_vs_res10k_party{}_d{}_proprep.png'.format(fixed_params['rep_num'],
                                                                         fixed_params['party_num'],
                                                                         fixed_params['district_num']),
                transparent=True)

