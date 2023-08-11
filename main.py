import numpy as np
import os
import pandas as pd
import itertools
import multiprocessing as mp
import yaml
from election_model import Election


def simulate_election(params, model_keys, n_iter = 5000, print_interval = None):
    np.random.seed()

    election_model = Election(**params)

    for i in range(n_iter):
        if print_interval != None and (i % print_interval == 0):
            print('iteration: {}'.format(i), flush=True)
        election_model.step()

    # collecting model results
    datacollector = election_model.report_model(model_keys)

    return datacollector


def iterate_seq(combo_vparams, model_keys, n_sim=10, n_iter=1000, print_interval=500):

    tmp_results = []

    for i in range(len(combo_vparams)):

        # pre-processing parameters
        params = fixed_params.copy()
        params.update(combo_vparams[i])
        print('variable parameters: ', combo_vparams[i], flush=True)

        for j in range(n_sim):
            print('simulation: {}'.format(j))
            sim_result = simulate_election(params, model_keys,
                                           n_iter, print_interval)
            tmp_results.append(sim_result)

    return tmp_results


def iterate_parallel(combo_vparams, model_keys, pool, process_num,
                     n_sim=10, n_iter=1000, print_interval=500):

    tmp_results = []

    for i in range(len(combo_vparams)):

        # pre-processing parameters
        params = fixed_params.copy()
        params.update(combo_vparams[i])
        print('variable parameters: ', combo_vparams[i], flush=True)

        print('simulations with {} processes'.format(process_num))

        sim_results = list(pool.apply_async(simulate_election,
                                        args=(params, model_keys,
                                              n_iter, print_interval,))
                       for j in range(n_sim))
        sim_results = [r.get() for r in sim_results]
        tmp_results.extend(sim_results)

    return tmp_results


if __name__ == '__main__':

    # load config file for hyper-parameters
    with open('election_config.yaml') as file:
        config_params = yaml.safe_load(file)

    fixed_params = config_params['fixed_params']

    result_path = config_params['result_path']
    n_sim = config_params['n_sim']
    n_iter = config_params['iter_num']
    print_interval = config_params['print_interval']
    elect_system = config_params['scenario']
    process_num = 2  # int(os.getenv('SLURM_CPUS_ON_NODE'))
    max_js_distance = 0.8325546111576977

    # keys for collecting data from the model
    model_keys = ['party_num', 'district_num', 'rep_num', 'voting', 'distribution', 'js_distance']

    # initiate multicore-processing pool
    if process_num == -1:
        process_num = mp.cpu_count()

    with mp.Pool(processes=process_num) as p:

        results = []

        if elect_system == 'deterministic':
            # Baseline scenario
            variable_params = {'rep_num': [1],
                               'party_num': [0],
                               'district_num': [1],
                               'voting': ['deterministic']}

            combo_vparams = [dict(zip(variable_params.keys(), a))
                             for a in itertools.product(*variable_params.values())]

            tmp_results = iterate_parallel(combo_vparams, model_keys, p, process_num,
                                           n_sim, n_iter, print_interval)
            results.extend(tmp_results)


        elif elect_system == 'one_per_party':
            # First-past-the-post
            variable_params = {'rep_num': [1],
                               'party_num': [2],
                               'district_num': [10],
                               'voting': ['one_per_party']}

            combo_vparams = [dict(zip(variable_params.keys(), a))
                             for a in itertools.product(*variable_params.values())]

            tmp_results = iterate_parallel(combo_vparams, model_keys, p, process_num,
                                           n_sim, n_iter, print_interval)
            results.extend(tmp_results)


        elif elect_system == 'proportional_rep':
            # Proportional representation
            variable_params = {'rep_num': [10],
                               'party_num': [2],
                               'district_num': [1],
                               'voting': ['proportional_rep']}

            combo_vparams = [dict(zip(variable_params.keys(), a))
                             for a in itertools.product(*variable_params.values())]
            combo_vparams.extend([{'rep_num': 20, 'party_num': party_num,
                                   'district_num': 5, 'voting': 'proportional_rep'}
                                  for party_num in [2]])

            tmp_results = iterate_parallel(combo_vparams, model_keys, p, process_num,
                                           n_sim, n_iter, print_interval)
            results.extend(tmp_results)

        p.close()
        p.join()

    print('simulation is complete.')

    result_df = pd.DataFrame(results)
    print(result_df.info())
    print(result_df.head())

    # save data
    result_file = os.path.join(result_path, 'elect_{}_results.csv'.format(elect_system))
    result_df.to_csv(result_file, index=False)