import os
os.chdir('/Home/gavin.wu/Python_Scorer')

import json
import codecs
import numpy as np
import pandas as pd
import xgboost as xgb
from scorefunc import score

test = pd.read_csv('marketing_test.csv', dtype=object)
scorer = score()
tt = test
ypred = scorer.get_score(tt)
print(ypred)
