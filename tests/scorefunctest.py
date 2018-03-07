import os
os.chdir('/Home/gavin.wu/PythonScorer')

import json
import codecs
import numpy as np
import pandas as pd
import xgboost as xgb
from scorefunc import score

full = pd.read_csv('train_data.csv', dtype=object)
dev = full[full["data_type"] == "dev"]
test = pd.read_csv('test_data.csv', dtype=object)
scorer = score()
tt = val

def pfunc():
  scorer.get_score(tt)

import cProfile
cProfile.run('pfunc()','pfunc_val.prof')

from vprof import profiler
profiler.run(pfunc, 'cmh')

dict_path = 'data_dict_fortest.txt'
def load_dict(dict_path):
    data_dict = []
    with codecs.open(dict_path,'rU','utf-8') as f:
        for line in f:
           data_dict.append(json.loads(line))
    return data_dict
tdict = load_dict(dict_path)


import random
test = pd.read_csv('marketing_test.csv', dtype = object)
scorer = score()

# Break some but not all input columns
tt = dev.copy().reset_index().drop('index',1)
aodata = pd.DataFrame(columns=dev.columns)
for index, row in tt.iterrows():
    if (index < 20):
        odata=row.copy()
        total_col = len(tdict)
        col_check_type = ['type_error', 'out_of_range', 'missing']
        num_col_check = random.sample(range(total_col), 1)
        which_col_check = random.sample(range(total_col), num_col_check[0])
        for col in which_col_check:
            col_name = tdict[col]['col_name']
            cct = random.sample(col_check_type, 1)
            if (cct[0] == 'out_of_range'):
                if (tdict[col]['data_type']=='con'):
                    odata[col_name] = 11.0 * float(tdict[col]['valid_val'][1]) - 10.0 * float(tdict[col]['valid_val'][0])
                if (tdict[col]['data_type']=='cat'):
                    odata[col_name] = 'nonsense'
            if (cct[0] == 'type_error'):
                if (tdict[col]['data_type']=='con'):
                    odata[col_name] = 'nonsense'
                if (tdict[col]['data_type']=='cat'):
                    odata[col_name] = 1e5
            if (cct[0] == 'missing'):
                odata[col_name] = np.nan
        aodata.loc[index]=odata
print scorer.get_score(aodata)

# Break all input columns
tt = dev.copy().reset_index().drop('index',1)
aodata = pd.DataFrame(columns=tt.columns)
anomaly = [np.nan, '', ' ', 0, '0', 'nonsense', 1e20, -1e20, 1e-20, -1e-20]
for index, jk in enumerate(anomaly):
    odata = tt.ix[0,]
    odata2 = pd.Series([jk for o in odata], index=odata.index)
    aodata.loc[index]=odata2
print scorer.get_score(aodata)

# Shuffle the order of the input columns
tt = dev.copy().reset_index().drop('index',1)
oschema = tt.columns.tolist()
nschema = pd.Series(oschema).take(np.random.permutation(len(oschema))).tolist()
tt2 = tt[nschema]
print scorer.get_score(tt2)
