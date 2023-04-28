import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt
import seaborn as sns
import multiprocessing as mp
from election_model import Election

import sys
sys.path.extend(['/Users/chanuwasaswamenakul/Documents/workspace/complex_election/src'])

params = {'N': 10000,
          'nom_rate': 0.05,
          'rep_num': 10,
          'party_num': 1,
          'district_num': 1,
          'voting': 'deterministic',
          'opinion_distribution': 'uniform'}

n_sim = 5
iterations = 5000
print_interval = 1000
process_num = 2

# initiate multicore-processing pool
if process_num == -1:
    process_num = mp.cpu_count()

## There are issues with multicore processing
## [Errno 2] No such file or directory: '/Users/chanuwasaswamenakul/Documents/workspace/complex_election/<input>'
'''
RuntimeError: 
        An attempt has been made to start a new process before the
        current process has finished its bootstrapping phase.

        This probably means that you are not using fork to start your
        child processes and you have forgotten to use the proper idiom
        in the main module:

            if __name__ == '__main__':
                freeze_support()
                ...

        The "freeze_support()" line can be omitted if the program
        is not going to be frozen to produce an executable.
'''
if process_num > 1:
    pool = mp.Pool(processes=process_num)

def simulate_election(params, n_iter = 5000, print_interval = None):

    election_model = Election(**params)

    for i in range(n_iter):
        if print_interval != None and (i % print_interval == 0):
            print('iteration: {}'.format(i))
        election_model.step()

    elected_opis = np.array([elected.x for elected in election_model.elected_pool])
    resident_opis = []

    for i in range(params['district_num']):
        district = election_model.districts[i]
        resident_opis.extend([resident.x for resident in district.residents])

    resident_opis = np.array(resident_opis)

    res_hists = np.histogram(resident_opis, bins=100, range=(-1, 1))[0]
    res_hists = res_hists / res_hists.sum()
    elect_hists = np.histogram(elected_opis, bins=100, range=(-1, 1))[0]
    elect_hists = elect_hists / elect_hists.sum()

    return distance.jensenshannon(res_hists, elect_hists)

run_data = []
def log_result(result):
    # a callback function for simulate_opf
    run_data.append(result)

if process_num == 1:
    for i in range(n_sim):
        js_dist = simulate_election(params, iterations, print_interval)
        run_data.append(js_dist)
else:
    for i in range(n_sim):
        # alternative to callback is using get()
        pool.apply_async(simulate_election,
                         args=(params, iterations, print_interval,),
                         callback=log_result)

    pool.close()
    pool.join()


print('simulation is complete.')

