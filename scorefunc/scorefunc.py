import json
import codecs
import pandas as pd
import numpy as np
import xgboost as xgb

class score:
    def __init__(self):
        self.dict_path = 'data_dict.txt'
        self.schema_path = 'final_schema.txt'
        self.model_path = 'marketing_model_python'
        self.ddict = self.load_dict()
        self.dschema = self.load_schema()
        self.model = self.prep_model()

    def get_score(self, df):
		"""
		This function does preprocessing and generate scores
		"""
        try:
            topred = self.prep_data(self.ddict, self.dschema, df)
            ypred = self.model.predict(topred)
        except:
            print('Scoring went wrong')
        return ypred

    def load_dict(self):
		"""
		This function loads column meta data for the input data, which is required for data preprocessing
		"""
        data_dict = []
        with codecs.open(self.dict_path,'rU','utf-8') as f:
            for line in f:
               data_dict.append(json.loads(line))
        return data_dict

    def load_schema(self):
		"""
		This function loads the final schema, which contains the list and order of required columns for prediction
		"""
        final_schema = pd.read_csv(self.schema_path, header=None)
        final_schema = pd.Series.tolist(final_schema[0])
        return final_schema

    def prep_data(self, ddict, dschema, df):
		"""
		This function preprocesses the input data
		"""
        df = pd.DataFrame(df)
        if df.shape[1] == 1:
            df = df.T
        indata = df.replace('-1', np.nan).replace('Discharge NA', np.nan).replace('', np.nan)
        out_data = pd.DataFrame([])
        columns_in = indata.columns.tolist()
        for i in range(len(ddict)):
            col_name = ddict[i]['col_name']
            default = np.nan if ddict[i]['default_val']=='NA' else ddict[i]['default_val']
            if (col_name in columns_in):
                col_out = indata[col_name]
                if (ddict[i]['data_type']=='con'):
                    try:
                        col_out = pd.to_numeric(col_out)
                        col_out[(~pd.isnull(col_out)) & (~((col_out>=float(ddict[i]['valid_val'][0])) & (col_out<=float(ddict[i]['valid_val'][1]))))]=float(default)
                    except:
                        col_out = float(default)
                    out_data[col_name] = col_out
                if (ddict[i]['data_type']=='cat'):
                    col_out = pd.Series([v if ((pd.isnull(v)) | (v in ddict[i]['valid_val'])) else default for v in col_out])
                    out_data[col_name] = pd.Series(pd.Categorical([v for v in col_out],categories=ddict[i]['valid_val']))
            else: out_data[col_name] = default
        out_data2 = pd.get_dummies(out_data, prefix_sep='__')
        out_data2 = out_data2[dschema]
        dm = xgb.DMatrix(out_data2, missing=np.nan)
        return dm

    def prep_model(self):
		"""
		This function loads the model binary required for scoring
		"""
        bst = xgb.Booster({'nthread':4})
        bst.load_model(self.model_path)
        return bst
