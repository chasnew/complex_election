import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

scenario_list = ['deterministic', 'one_per_party', 'proportional_rep']

result_list = []

for scenario in scenario_list:
    tmp_df = pd.read_csv('results/{}_result.csv'.format(scenario))
    result_list.append(tmp_df)

result_df = pd.concat(result_list)