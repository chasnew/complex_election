import numpy as np
import os
import pandas as pd
import itertools
import multiprocessing as mp
import yaml
from election_model import Election


def simulate_election(params, model_keys, sim_type='when_pemerge',
                      new_party=None, delta=0.05, move_period=50, n_iter=5000,
                      print_interval=None, model_report=True, step_report=False):
    np.random.seed()

    election_model = Election(**params)
    step_data = []
    print(election_model)

    # Party that moves position in radicalization or vote capture scenarios
    mov_party = election_model.parties[0]
    delta = delta # how much party moves each time in radicalization scenarios

    for i in range(n_iter):
        if print_interval != None and (i % print_interval == 0):
            print('iteration: {}'.format(i), flush=True)

        # in radicalization scenario, stop simulations when the party move beyond the bounds
        if (sim_type == 'radicalization') or (sim_type == 'vote_capture'):
            if (i == 0) and (move_period < 5):
                # warm-up steps
                for j in range(4):
                    election_model.step()

            if (mov_party.x < 0) or (mov_party.x > 1):
                break

        election_model.step()

        # collecting per-timestep summary
        if step_report:
            step_dict = election_model.report_step()
            step_dict['step'] = i
            step_data.append(step_dict)

        # new party attempts to join election (party emergence scenarios)
        if (sim_type == 'when_pemerge') or (sim_type == 'geo_sort'):
            if (new_party is not None) and (i == 5):
                election_model.form_new_party(party_pos=new_party)

        # party moves (party radicalization and vote capture scenarios)
        elif sim_type == 'radicalization':
            if ((i+1) % move_period == 0):
                mov_party = election_model.parties[0]
                mov_party.x = np.round(mov_party.x + delta, decimals=2)
                print('move party:', mov_party.x)
        elif sim_type == 'vote_capture':
            if ((i+1) % move_period == 0):
                mov_party = election_model.parties[0]
                mov_party.x = np.round(mov_party.x - delta, decimals=2)
                print('move party:', mov_party.x)
        elif sim_type == 'radicalization_and_emerge':
            mov_party = election_model.parties[0]
            mov_ppos = mov_party.x
            party_num = len(election_model.parties)

            if (mov_ppos == 0.6 and party_num < 3):
                election_model.form_new_party(party_pos=new_party)

            if ((i+1) % move_period == 0) and (mov_ppos < 0.6):
                mov_party.x = np.round(mov_party.x + delta, decimals=2)
                print('move party:', mov_party.x)

    # collecting model results
    if model_report:
        datacollector = election_model.report_model(model_keys)

    if step_report and model_report:
        return (step_data, datacollector)
    elif step_report:
        return step_data
    else:
        return datacollector


def iterate_seq(combo_vparams, fixed_params, model_keys, sim_type,
                new_party=None, delta=0.05, move_period=50, n_sim=10,
                n_iter=1000, print_interval=500, model_report=True, step_report=False):

    tmp_results = []

    for i in range(len(combo_vparams)):

        # pre-processing parameters
        params = fixed_params.copy()
        params.update(combo_vparams[i])
        print('variable parameters: ', combo_vparams[i], flush=True)

        for j in range(n_sim):
            print('simulation: {}'.format(j))
            sim_result = simulate_election(params, model_keys, sim_type,
                                           new_party, delta, move_period, n_iter,
                                           print_interval, model_report, step_report)
            tmp_results.append(sim_result)

    return tmp_results

def iterate_parallel(combo_vparams, fixed_params, model_keys, pool, process_num,
                     sim_type, new_party=None, delta=0.05, move_period=50, n_sim=10, n_iter=1000,
                     print_interval=500, model_report=True, step_report=False):

    tmp_results = []

    for i in range(len(combo_vparams)):

        # pre-processing parameters
        params = fixed_params.copy()
        params.update(combo_vparams[i])
        print('variable parameters: ', combo_vparams[i], flush=True)

        print('simulations with {} processes'.format(process_num))

        # parallelize different sim runs with the same params
        sim_results = list(pool.apply_async(simulate_election,
                                        args=(params, model_keys, sim_type,
                                              new_party, delta, move_period, n_iter,
                                              print_interval, model_report, step_report,))
                       for j in range(n_sim))
        sim_results = [r.get() for r in sim_results]
        tmp_results.extend(sim_results)

    return tmp_results

def unpack_step_results(step_results, combo_vparams, fixed_params,
                        n_sim, new_party, sim_type, delta=0.05, move_period=50):

    all_step_results = []

    party_num = fixed_params['party_num']
    if (new_party is not None):
        party_num += 1

    init_p1 = fixed_params['party_loc'][0] # used in "radicalization" and "vote_capture" scenarios

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

            if sim_type == 'radicalization':
                mov_ppos = []
                nunique_pos = int(result_df.shape[0] / move_period) # number of positions of the moving party
                max_pos = np.round(init_p1 + (delta * (nunique_pos - 1)), decimals=2) # latest position
                unique_pos = np.linspace(init_p1, max_pos, nunique_pos) # enumerate all positions

                for pos in unique_pos:
                    mov_ppos.extend([pos] * move_period)

                result_df['mov_ppos'] = mov_ppos

            elif sim_type == 'vote_capture':
                mov_ppos = []
                nunique_pos = int(result_df.shape[0] / move_period)  # number of positions of the moving party
                max_pos = np.round(init_p1 - (delta * (nunique_pos - 1)), decimals=2)  # latest position
                unique_pos = np.linspace(init_p1, max_pos, nunique_pos)  # enumerate all positions

                for pos in unique_pos:
                    mov_ppos.extend([pos] * move_period)

                result_df['mov_ppos'] = mov_ppos

            elif sim_type == 'radicalization_and_emerge':
                mov_ppos = []
                nunique_pos = int(result_df.shape[0] / move_period)  # number of positions of the moving party
                max_pos = np.round(init_p1 + (delta * (nunique_pos - 1)), decimals=2)  # latest position
                unique_pos = np.linspace(init_p1, max_pos, nunique_pos)  # enumerate all positions
                unique_pos = np.clip(unique_pos, a_min=0, a_max=0.6) # radicalized party stop moving at 0.6

                for pos in unique_pos:
                    mov_ppos.extend([pos] * move_period)

                result_df['mov_ppos'] = mov_ppos

            all_step_results.append(result_df)

    all_step_results = pd.concat(all_step_results).reset_index(drop=True)
    all_step_results[vprop_cols] = pd.DataFrame(all_step_results['vote_prop'].to_list())
    all_step_results[sprop_cols] = pd.DataFrame(all_step_results['seat_prop'].to_list())

    all_step_results.drop(columns=['vote_prop', 'seat_prop'], inplace=True)

    print(all_step_results.info())
    print(all_step_results.head())
    # print(all_step_results['mov_ppos'].unique())

    return all_step_results


if __name__ == '__main__':

    # load config file for hyper-parameters
    with open('election_config.yaml') as file:
        config_params = yaml.safe_load(file)

    # 'when_pemerge', 'geo_sort', 'radicalization', 'vote_capture'
    sim_type = config_params['sim_type']

    fixed_params = config_params['fixed_params']

    result_path = config_params['result_path']
    n_sim = config_params['n_sim']
    n_iter = config_params['iter_num']
    print_interval = config_params['print_interval']
    elect_system = fixed_params['elect_system']
    voting = fixed_params['voting']
    pop_mag = int(fixed_params['N']/1000)
    delta = config_params['delta']
    move_period = config_params['move_period']

    new_party = config_params['new_party']
    if new_party == 'moderate':
        new_party_pos = 0
    elif new_party == 'extreme':
        new_party_pos = 0.75
    else: # use "null" in yaml to specify None
        new_party_pos = new_party

    process_num = 1 #int(os.getenv('SLURM_CPUS_ON_NODE'))

    model_report = config_params['model_report']
    step_report = config_params['step_report']

    # keys for collecting data from the model
    model_keys = ['party_num', 'elect_system', 'voting',
                  'distribution', 'js_distance']

    # initiate multicore-processing pool
    if process_num == -1:
        process_num = mp.cpu_count()

    if process_num > 1:
        p = mp.Pool(processes=process_num)

    # with mp.Pool(processes=process_num) as p:

    results = []

    # testing parameter range of voter behaviors (strategic voting & history bias) that allows party to emerge
    if sim_type == 'when_pemerge':
        variable_params = {'alpha': [np.round(val, decimals=2) for val in np.linspace(0, 1, 21)],
                           'beta': [np.round(val, decimals=2) for val in np.linspace(0, 1, 21)]}
    # testing the degree of ideological sorting in FPTP that allows party to emerge
    elif sim_type == 'geo_sort':
        # fixed_params['beta'] = 0.5
        del fixed_params['ideo_sort']

        variable_params = {'alpha': [0.1, 0.5, 0.9],
                           'beta': [0.3, 0.5, 0.7],
                           'ideo_sort': [np.round(val, decimals=1) for val in np.linspace(0, 1, 11)]}

    elif (sim_type == 'radicalization') or (sim_type == 'vote_capture'):
        # fixed_params['beta'] = 0.4
        variable_params = {'alpha': [np.round(val, decimals=1) for val in np.linspace(0, 1, 11)],
                           'beta': [np.round(val, decimals=1) for val in np.linspace(0, 1, 11)]}

    elif (sim_type == 'radicalization_and_emerge'):
        variable_params = {'alpha': [np.round(val, decimals=1) for val in np.linspace(0, 1, 11)],
                           'beta': [np.round(val, decimals=1) for val in np.linspace(0, 1, 11)]}

    elif (sim_type == 'baseline'):
        variable_params = {'alpha': [0], 'beta': [0]}

    combo_vparams = [dict(zip(variable_params.keys(), a))
                     for a in itertools.product(*variable_params.values())]

    if process_num > 1:
        tmp_results = iterate_parallel(combo_vparams, fixed_params, model_keys, p, process_num,
                                       sim_type, new_party_pos, delta, move_period, n_sim, n_iter,
                                       print_interval, model_report, step_report)
    else:
        tmp_results = iterate_seq(combo_vparams, fixed_params, model_keys, sim_type, new_party_pos,
                                  delta, move_period, n_sim, n_iter, print_interval, model_report, step_report)

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

    if model_report and step_report:
        unzip_res = list(zip(*results)) # split aggregate results and time-step results

        model_results = pd.DataFrame(unzip_res[1])
        sim_ids = [i for i in range(n_sim)] * len(combo_vparams)
        model_results['sim_id'] = sim_ids

        print(model_results.info())
        print(model_results.head())

        step_results = unzip_res[0]
        all_step_results = unpack_step_results(step_results, combo_vparams, fixed_params, n_sim,
                                               new_party, sim_type, delta, move_period)

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
        all_step_results = unpack_step_results(results, combo_vparams, fixed_params, n_sim,
                                               new_party, sim_type, delta, move_period)
        all_step_results.to_csv(sresult_file, index=False)