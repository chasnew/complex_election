import numpy as np
import os
import pandas as pd
import itertools
import multiprocessing as mp
import yaml
from election_model import Election


def simulate_election(params, model_keys, new_party=None, n_iter=5000,
                      print_interval=None, model_report=True, step_report=False):
    np.random.seed()

    election_model = Election(**params)
    step_data = []
    print(election_model)

    for i in range(n_iter):
        if print_interval != None and (i % print_interval == 0):
            print('iteration: {}'.format(i), flush=True)

        election_model.step()

        # collecting per-timestep summary
        if step_report:
            step_dict = election_model.report_step()
            step_dict['step'] = i
            step_data.append(step_dict)

        # new party attempts to join election
        if (new_party is not None) and (i == int(n_iter/3)):
            election_model.form_new_party(party_pos=new_party)


    # collecting model results
    if model_report:
        datacollector = election_model.report_model(model_keys)

    if step_report and model_report:
        return (step_data, datacollector)
    elif step_report:
        return step_data
    else:
        return datacollector

# need to add process num for parallel district processing
def iterate_seq(combo_vparams, fixed_params, model_keys, new_party=None, n_sim=10,
                n_iter=1000, print_interval=500, model_report=True, step_report=False):

    tmp_results = []

    for i in range(len(combo_vparams)):

        # pre-processing parameters
        params = fixed_params.copy()
        params.update(combo_vparams[i])
        print('variable parameters: ', combo_vparams[i], flush=True)

        for j in range(n_sim):
            print('simulation: {}'.format(j))
            sim_result = simulate_election(params, model_keys, new_party,
                                           n_iter, print_interval,
                                           model_report, step_report)
            tmp_results.append(sim_result)

    return tmp_results


def iterate_parallel(combo_vparams, fixed_params, model_keys, pool, process_num,
                     new_party=None, n_sim=10, n_iter=1000, print_interval=500,
                     model_report=True, step_report=False):

    tmp_results = []

    for i in range(len(combo_vparams)):

        # pre-processing parameters
        params = fixed_params.copy()
        params.update(combo_vparams[i])
        print('variable parameters: ', combo_vparams[i], flush=True)

        print('simulations with {} processes'.format(process_num))

        # parallelize different sim runs with the same params
        sim_results = list(pool.apply_async(simulate_election,
                                        args=(params, model_keys, new_party,
                                              n_iter, print_interval,
                                              model_report, step_report,))
                       for j in range(n_sim))
        sim_results = [r.get() for r in sim_results]
        tmp_results.extend(sim_results)

    return tmp_results

def unpack_step_results(step_results, combo_vparams, n_sim, party_num):

    all_step_results = []

    vprop_cols = [f'vote_prop{i}' for i in range(party_num)]
    sprop_cols = [f'seat_prop{i}' for i in range(party_num)]

    for i, params in enumerate(combo_vparams):

        # iterate over each simulation run for the same parameters
        for j in range(n_sim):

            result_ind = i * n_sim + j
            result_df = pd.DataFrame(step_results[result_ind])

            for key, val in params.items():
                result_df[key] = val

            result_df['sim_id'] = j

            all_step_results.append(result_df)

    all_step_results = pd.concat(all_step_results)
    all_step_results[vprop_cols] = pd.DataFrame(all_step_results['vote_prop'].to_list())
    all_step_results[sprop_cols] = pd.DataFrame(all_step_results['seat_prop'].to_list())

    all_step_results.drop(columns=['vote_prop', 'seat_prop'], inplace=True)

    print(all_step_results.info())
    print(all_step_results.head())

    return all_step_results


if __name__ == '__main__':

    # load config file for hyper-parameters
    with open('election_config.yaml') as file:
        config_params = yaml.safe_load(file)

    sim_type = config_params['sim_type']

    fixed_params = config_params['fixed_params']

    result_path = config_params['result_path']
    n_sim = config_params['n_sim']
    n_iter = config_params['iter_num']
    print_interval = config_params['print_interval']
    elect_system = fixed_params['voting']
    pop_mag = int(fixed_params['N']/1000)

    new_party = config_params['new_party']
    if new_party == 'moderate':
        new_party_pos = 0
    elif new_party == 'extreme':
        new_party_pos = 0.8
    else:
        new_party_pos = None

    process_num = 1  # int(os.getenv('SLURM_CPUS_ON_NODE'))

    model_report = config_params['model_report']
    step_report = config_params['step_report']

    # keys for collecting data from the model
    model_keys = ['party_num', 'voting',
                  'distribution', 'strategic', 'js_distance']

    # initiate multicore-processing pool
    if process_num == -1:
        process_num = mp.cpu_count()

    if process_num > 1:
        p = mp.Pool(processes=process_num)

    # with mp.Pool(processes=process_num) as p:

    results = []

    # testing parameter range of voter behaviors (strategic voting & history bias) that allows party to emerge
    if sim_type == 'when_pemerge':
        variable_params = {'alpha': [np.round(val, decimals=2) for val in np.linspace(0, 1, 3)],
                           'beta': [np.round(val, decimals=2) for val in np.linspace(0, 1, 3)]}
    # testing the degree of ideological sorting in FPTP that allows party to emerge
    elif sim_type == 'fptp_sort':
        fixed_params['beta'] = 0.5
        del fixed_params['ideo_sort']

        variable_params = {'alpha': [0.1, 0.5, 0.9],
                           'ideo_sort': [np.round(val, decimals=1) for val in np.linspace(0, 1, 3)]}

    combo_vparams = [dict(zip(variable_params.keys(), a))
                     for a in itertools.product(*variable_params.values())]

    if process_num > 1:
        tmp_results = iterate_parallel(combo_vparams, fixed_params, model_keys, p, process_num,
                                       new_party_pos, n_sim, n_iter, print_interval,
                                       model_report, step_report)
    else:
        tmp_results = iterate_seq(combo_vparams, fixed_params, model_keys, new_party_pos,
                                  n_sim, n_iter, print_interval, model_report, step_report)

    results.extend(tmp_results)

    if process_num > 1:
        p.close()
        p.join()

    print('simulation is complete.')

    # save data
    mres_filename = 'elect{}k_{}_p{}_{}_{}{}.csv'.format(pop_mag, elect_system, fixed_params['party_num'],
                                                      new_party, sim_type, config_params['batch_id'])
    sres_filename = 'estep{}k_{}_p{}_{}_{}{}.csv'.format(pop_mag, elect_system, fixed_params['party_num'],
                                                      new_party, sim_type, config_params['batch_id'])

    mresult_file = os.path.join(result_path, mres_filename)
    sresult_file = os.path.join(result_path, sres_filename)

    final_party_num = fixed_params['party_num']
    if (new_party is not None):
        final_party_num += 1

    if model_report and step_report:
        unzip_res = list(zip(*results)) # split aggregate results and time-step results

        model_results = pd.DataFrame(unzip_res[1])
        sim_ids = [i for i in range(n_sim)] * len(combo_vparams)
        model_results['sim_id'] = sim_ids

        print(model_results.info())
        print(model_results.head())

        step_results = unzip_res[0]
        all_step_results = unpack_step_results(step_results, combo_vparams, n_sim, final_party_num)

        model_results.to_csv(mresult_file, index=False)
        all_step_results.to_csv(sresult_file, index=False)

    elif model_report:
        model_results = pd.DataFrame(results)
        sim_ids = [i for i in range(n_sim)] * len(combo_vparams)
        model_results['sim_id'] = sim_ids

        print(model_results.info())
        print(model_results.head())

        model_results.to_csv(mres_filename, index=False)

    else:
        all_step_results = unpack_step_results(results, combo_vparams, n_sim, final_party_num)
        all_step_results.to_csv(sresult_file, index=False)