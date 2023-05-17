import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from datetime import datetime, timedelta
import time
from pathlib import Path
import statsmodels.api as sm
import yaml

import factor_pca_functions 
import nowcast_dfm_functions 

def generate_dfm(data_root, results_root, config, pca_factors=True,
                 dfm_factors=True):
    '''
    generates the results for DFM, along with saving factors to be used in 
    signatures. These factors can be PCA principal components based on the 
    same factor grouping or they can be ones computed by DFM
    
    Parameters  data_root: root directory to find the raw data and to 
                           save the factors in
                results_root: root directory to save the DFM results to
                config: dictionary containing the data configs, namely:
                        - 'data_file': string giving name of data file
                        - 'meta_file': string giving name of metadata file
                        - 'pub_lag_info': string giving name of lag info file
                        - 'ref_date': the starting date for when to 
                                      take data from,
                        - 'start_date': the start date for nowcast
                        - 'end_date': the end date for nowcast
                        - 'global_order': order of the autoregressive process 
                                          for the global factor,
                        - 'global_multiplicity': number of global factors
                pca_factors: Bool indicating whether to save PCA factors
                dfm_factors: Bool indicating whether to save DFM factors
    
    '''

    # Create the folders
    print(f'Reading from {data_root}, where factors will be saved to.')

    results_folder = results_root / 'dfm'
    Path(results_folder).mkdir(parents=True, exist_ok=True)
    print(f'DFM results will be saved to{results_folder}')
    
    vars_df, pub_lag_df, vars_dict, vars_meta, factor_blocks_list = \
    nowcast_dfm_functions.gen_factor_vars(data_root,
                                          config['data_file'],
                                          config['meta_file'], 
                                          config['pub_lag_info'])

    start_date = datetime.strptime(config['start_date'], '%Y-%m-%d')
    ref_date = datetime.strptime(config['ref_date'], '%Y-%m-%d')
    end_date = datetime.strptime(config['end_date'], '%Y-%m-%d')

    num_horizons = end_date - start_date
    num_horizons = (num_horizons.days / 7) + 1
    day_iter = nowcast_dfm_functions.gen_horizon_intervals(0,
                                                           int(num_horizons),
                                                           7)

    vars_df_shift = vars_df.shift(1)
    pred_dict = {}
    
    # to store runtime values
    runtime_list = []

    # loop across horizons
    for i in day_iter:

        df_available_at_horizon = pd.DataFrame()
        horizon = start_date + relativedelta(days=int(i))

        df_available_at_horizon = \
        nowcast_dfm_functions.gen_data_for_horizons(vars_df_shift, pub_lag_df,
                                                    ref_date, horizon,
                                                    df_available_at_horizon)

        # fall back from "current" month to display correct info available
        # for each month
        df_available_at_horizon = df_available_at_horizon.shift(-1)
        
        # drop columns with missing values for entire period
        # this only occurs for PPIFIS pre-2010
        
        if horizon < datetime.strptime('2010-02-19', '%Y-%m-%d'):
            df_available_at_horizon = \
            df_available_at_horizon.drop(['PPIFIS'], axis=1)

        if pca_factors:
            factors_df = factor_pca_functions\
            .append_factor_loadings(df_available_at_horizon,
                                    vars_dict, factor_blocks_list)
            
            pca_factors_folder = data_root / 'pca_factors'
            Path(pca_factors_folder).mkdir(parents=True, exist_ok=True)
            data_date = datetime.strftime(horizon, "%Y_%m_%d")
            factors_df.to_csv(f'{pca_factors_folder}/'
                              f'factors_{data_date}.csv')

        endog_m_nyfed, endog_q_nyfed = \
        nowcast_dfm_functions.format_for_dfm(df_available_at_horizon,
                                             config['ref_date'])

        factors = {row['var']: [row['global_'], row['local_factor']]
                   for i, row in vars_meta.iterrows()}

        factor_orders = {'Global': config['global_order']}
        factor_multiplicities = {'Global': config['global_multiplicity']}

        # construct the dynamic factor model
        model = sm.tsa.DynamicFactorMQ(
            endog_m_nyfed, endog_quarterly=endog_q_nyfed,
            factors=factors, factor_orders=factor_orders,
            factor_multiplicities=factor_multiplicities)

        # generate current prediction quarter
        q_horizon = nowcast_dfm_functions.gen_end_of_q_datetime(horizon)
        
        print('Current horizon: ' f'{horizon}'[:-9])
        print('Current prediction quarter: ' f'{q_horizon}'[:-12])
        print('Horizon iteration: ' f'{int((i / 7) + 1)}/{len(day_iter)}')

        if i == 0:
            print('Runtime estimation generated in next iteration')
        else:
            # update estimation timings where possible
            if len(runtime_list) > 5:
                time_left = np.average(runtime_list[-5:]) \
                * (len(day_iter) - int((i / 7)))

            time_left = np.average(runtime_list) * \
            (len(day_iter) - int((i / 7)))
            print('Estimated time remaining: '
                  f'{str(timedelta(seconds=time_left))}')

        # fit model and update runtimes
        start_est_time = time.time()
        results = model.fit(disp=30)

        pred_res = results.get_prediction(f'{q_horizon}'[:-12])
        point_pred = pred_res.predicted_mean['GDPC1']
        point_pred = point_pred.resample('Q').last()

        point_pred = {f'{horizon}'[:-9]: point_pred[0]}
        pred_dict.update(point_pred)

        end_est_time = time.time()
        runtime_list.append(end_est_time - start_est_time)

        if dfm_factors:
            factors_folder = data_root/ 'true_factors'
            Path(factors_folder).mkdir(parents=True, exist_ok=True)
            data_date = datetime.strftime(horizon, "%Y_%m_%d")
            results.factors.filtered.to_csv(f'{factors_folder}/filtered_'
                                            f'factors_{data_date}.csv')
            results.factors.smoothed.to_csv(f'{factors_folder}/smoothed_'
                                            f'factors_{data_date}.csv')

    # Save DFM results
    for k, v in pred_dict.items():
        pred_dict[k] = float(v)
    with open(results_folder/'dfm_predictions.yaml', 'w') as file:
        yaml.dump(pred_dict, file)

if __name__ == '__main__':
    
    config = {
        'data_file': 'nyfed_data.csv',
        'meta_file': 'nyfed_metadata.csv',
        'pub_lag_info': 'nyfed_indicators.csv',
        #'pca_ncomp': 4,
        'ref_date': '2000-01-01',
        'start_date': '2016-01-01', 
        'end_date': '2019-12-31',
        'global_order': 1,
        'global_multiplicity': 1
    }
    
    data_root = Path(__file__).resolve().parent.parent / 'data_test'
    results_root = Path(__file__).resolve().parent.parent /\
    'results_test'
    
    generate_dfm(data_root, results_root, config)