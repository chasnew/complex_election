import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

max_js_distance = 0.8325546111576977
scenario_list = ['deterministic', 'one_per_party', 'proportional_rep']

result_list = []

for scenario in scenario_list:
    tmp_df = pd.read_csv('results/elect10k_{}_results.csv'.format(scenario))
    result_list.append(tmp_df)

result_df = pd.concat(result_list).reset_index(drop=True)

det_results = result_df[result_df['voting'] == 'deterministic']
det_results['rep_dist_num'] = det_results['rep_num'].astype(str) + '_' + det_results['district_num'].astype(str)
det_results['party_num'] = det_results['party_num'].astype(str)
sns.stripplot(det_results, x='rep_dist_num', y='js_distance', hue='party_num')
plt.show()

major_results = result_df[result_df['voting'] == 'one_per_party']
major_results['party_num'] = major_results['party_num'].astype(str)
sns.stripplot(major_results, x='district_num', y='js_distance', hue='party_num')
plt.show()

propr_results = result_df[result_df['voting'] == 'proportional_rep']
propr_results['rep_dist_num'] = propr_results['rep_num'].astype(str) + '_' + propr_results['district_num'].astype(str)
propr_results['party_num'] = propr_results['party_num'].astype(str)
sns.stripplot(propr_results, x='rep_dist_num', y='js_distance', hue='party_num')
plt.show()


# Contrasting Plurality voting to Proportional representation (rep * district = 10)
cond1 = (result_df['voting'] == 'deterministic') & (result_df['district_num'] == 10) &\
        (result_df['rep_num'] == 1) & (result_df['party_num'].isin([2,5,10]))
cond2 = (result_df['voting'] == 'deterministic') & (result_df['district_num'] == 1) &\
        (result_df['rep_num'] == 10) & (result_df['party_num'].isin([2,5,10]))
cond3 = (result_df['voting'] == 'one_per_party') & (result_df['district_num'] == 10)
cond4 = (result_df['voting'] == 'proportional_rep') & (result_df['district_num'] == 1) &\
        (result_df['rep_num'] == 10)

comb1_results = result_df[cond1 | cond2 | cond3]
comb1_results['voting'] = comb1_results['voting'] + '_' + comb1_results['rep_num'].astype(str) +\
                          '_' + comb1_results['district_num'].astype(str)
comb1_results['party_num'] = comb1_results['party_num'].astype(str)
sns.stripplot(comb1_results, x='voting', y='js_distance', hue='party_num')
plt.show()



# Contrasting Plurality voting to Proportional representation (rep * district = 50)
cond1 = (result_df['voting'] == 'one_per_party') & (result_df['district_num'] == 50)
cond2 = (result_df['voting'] == 'proportional_rep') & (result_df['district_num'] == 5) &\
        (result_df['rep_num'] == 10)

comb2_results = result_df[cond1 | cond2]
comb2_results['party_num'] = comb2_results['party_num'].astype(str)
sns.stripplot(comb2_results, x='voting', y='js_distance', hue='party_num')
plt.show()



# Contrasting Plurality voting to Proportional representation (rep * district = 100)
cond1 = (result_df['voting'] == 'one_per_party') & (result_df['district_num'] == 100)
cond2 = (result_df['voting'] == 'proportional_rep') & (result_df['district_num'] == 10) &\
        (result_df['rep_num'] == 10)
cond3 = (result_df['voting'] == 'proportional_rep') & (result_df['district_num'] == 5) &\
        (result_df['rep_num'] == 20)

comb3_results = result_df[cond1 | cond2 | cond3]
comb3_results['party_num'] = comb3_results['party_num'].astype(str)
sns.stripplot(comb3_results, x='voting', y='js_distance', hue='party_num')
plt.show()