import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt
import seaborn as sns
from election_model import Election

N = 10000
nom_rate = 0.05
rep_num = 10
party_num = 1
district_num = 1
voting = 'deterministic'
print_interval = 500

simple_election = Election(N, nom_rate=nom_rate, rep_num=rep_num,
                           party_num=party_num, party_sd=0.2,
                           district_num=district_num)

iterations = 5000
for i in range(iterations):
    if i % print_interval == 0:
        print('iteration: {}'.format(i))
    simple_election.step(voting=voting)

print('simulation is complete.')


elected_opis = np.array([elected.x for elected in simple_election.elected_pool])
resident_opis = []
for i in range(district_num):
    district = simple_election.districts[i]
    resident_opis.extend([resident.x for resident in district.residents])

resident_opis = np.array(resident_opis)

# plt.style.use('seaborn-v0_8-colorblind')
fig, ax = plt.subplots(figsize=(12,7))

sns.histplot(elected_opis, stat='probability',
                  bins=np.linspace(-1,1,100), alpha=0.5, ax=ax)
ax.legend(labels=['elected_candidates'])
sns.histplot(resident_opis, stat='probability',
             bins=np.linspace(-1,1,100), alpha=0.5, ax=ax)
ax.set_xlim(-1,1)
# ax.set_ylim(0,0.05)
ax.legend(['elected candidates', '_', 'residents'])
plt.title('Distributions of representatives and residents opinions (d = {}, N = {})'.format(district_num, N))

if (voting != 'prop_rep'):
    if (party_num > 0):
        if (district_num > 1):
            plt.savefig('../results/reps{}_vs_res5k_party{}_d{}.png'.format(rep_num, party_num, district_num))
        else:
            plt.savefig('../results/reps{}_vs_res5k_party{}.png'.format(rep_num, party_num))
    else:
        if (district_num > 1):
            plt.savefig('../results/reps{}_vs_res5k_d{}_basic.png'.format(rep_num, district_num))
        else:
            plt.savefig('../results/reps{}_vs_res5k_basic.png'.format(rep_num))
else:
    plt.savefig('../results/reps{}_vs_res5k_party{}_d{}_proprep.png'.format(rep_num, party_num, district_num))