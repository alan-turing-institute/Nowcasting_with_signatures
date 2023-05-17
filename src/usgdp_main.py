import sys
import yaml
from pathlib import Path
import numpy as np
import pandas as pd

import compute_linear_sigs as sig_funcs
import signature_helper_funcs as helper_funcs
from dfm_extract import generate_dfm

def prepare_gdp_labels(input_dir):

    '''
    Prepare the prediction labels from nyfed_data.csv
    
    Parameters  input_dir: path where the data is stored
    
    Returns     df_gdp: dataframe of data
    '''
    
    df_gdp = pd.read_csv(f'{str(input_dir)}/nyfed_data.csv', 
                         usecols=['DATE', 'GDPC1'])
    df_gdp = df_gdp.join(df_gdp[~df_gdp['GDPC1'].isnull()]\
                         ['GDPC1'].shift().rename('GDPC1_prev1'))
    df_gdp = df_gdp.join(df_gdp[~df_gdp['GDPC1'].isnull()]\
                         ['GDPC1'].shift(2).rename('GDPC1_prev2'))

    return df_gdp


def merge_data(df_factors, df_gdp, configs, how='inner'):
    '''
    Merge the dataframe of factors and GDP together
    
    Parameters  df_factors: dataframe of factors/principal components
                df_gdp: dataframe of US GDP
                configs: Contains model parameters, in particular:
                          - target: target variable to nowcast
                          - target_lag: the number of days to apply the
                                        shift in target to obtain the 
                                        previous value (depends on 
                                        publication lag)
                          - window_type: whether we are taking a rolling
                                         window according to number of
                                         days or number of rows
                how: string indicating type of join, we join at first 
                     through an outer join in any case, but subsequently
                     remove empty rows if inner join is specified here
    Returns     df_merge: dataframe of both observed and target data merged
    
    '''
    
    df_merge = pd.merge(df_factors, df_gdp, how='outer')
    df_merge['DATE'] = pd.to_datetime(df_merge['DATE'])
    
    df_merge = df_merge.sort_values('DATE')
    df_merge.reset_index(inplace=True, drop=True)
    
    df_merge['GDPC1'] = df_merge['GDPC1'].ffill()
    df_merge['GDPC1_prev1'] = df_merge['GDPC1_prev1'].ffill()
    df_merge['GDPC1_prev2'] = df_merge['GDPC1_prev2'].ffill()
    
    df_merge.index = df_merge['DATE']
    # Find the most recent GDP, used a lag of 125 rather than 120 so that
    # there is 1 month in the new quarter without the information of the old
    # quarter.
    
    df_merge = \
    helper_funcs.append_previous_target(df_merge.rename({'DATE':'t'}, axis=1),
                                        df_merge, configs)
    
    current_quarter = [pd.to_datetime(str(df_merge['DATE'][i]\
                                          .to_period('Q')))\
                       for i in range(len(df_merge))]
    
    # Find time elapsed in quarter
    df_merge['quarter_time'] = [df_merge['DATE'][i]- current_quarter[i] \
                                for i in range(len(df_merge))]
    df_merge['quarter_time'] = df_merge['quarter_time'].dt.days
    
    if how == 'inner':
        df_merge = df_merge[~df_merge['Labour'].isnull()]
        df_merge.reset_index(inplace=True, drop=True)

    return df_merge


def find_signatures(df_merge, configs):
    '''
    Finds the signature terms from the merged dataframe for USGDP
    
    Parameters  df_merge: dataframe of both observed and target data
                configs: a dictionary of the config parameters
    Returns     observation_sigs: signatures of the observed data
    '''

    observation_filled = df_merge.drop(['GDPC1', 'GDPC1_prev1',
                                        'GDPC1_prev2'], axis=1)
    observation_filled.rename({'DATE':'t'}, axis=1, inplace=True)
    observation_filled.set_index('t', inplace=True, drop=False)
    # Convert the time index to days
    observation_filled['t'] = (observation_filled['t']-\
                               observation_filled['t'][0]).dt.days.values

    multiplier_target = df_merge[['DATE', configs['target']]].copy()
    multiplier_target.rename({'DATE':'t'}, axis=1, inplace=True)
    multiplier_target.set_index('t', inplace=True)

    observation_sigs = sig_funcs.compute_sigs_dates(observation_filled, 
                                                    configs,
                                                    multiplier_target)

    return observation_sigs


def iteration_on_files(factor_file, df_gdp, config, results_list, 
                       results_dir):
    '''
    Fit model on a factor file
    
    Parameters  factor_file: filename of the factors
                df_gdp: dataframe for US GDP             
                configs: a dictionary of the config parameters
                results_list: current list of results
                results_dir: directory to store the results
    Returns     results_list: updated list of results  
                target_pred: dataframe with predicted value(s) for target
    
    '''
    
    file_name = str(factor_file).split("/")[-1]
    print(f'Nowcasting with {file_name}')

    # Obtain the date of the data from the file name
    data_date = file_name[-14:-4]

    df_factors = pd.read_csv(factor_file, 
                             usecols=['DATE', 'Labour', 'Real',
                                      'Soft', 'Global'])
    df_merge = merge_data(df_factors, df_gdp, config, how='inner')

    if not config['use_prev_value']:
        df_merge.drop(['recent_target'], axis=1, inplace=True)

    if not config['use_quarter_time']:
        df_merge.drop(['quarter_time'], axis=1, inplace=True)
    observation_sigs = find_signatures(df_merge, config)

    (results_dir/'predictions'/data_date).mkdir(parents=True,
                                                exist_ok=True)
    results_list, target_pred = \
    helper_funcs.fit_model_range(observation_sigs, df_merge,
                                 results_dir/'predictions'/data_date,
                                 config, results_list = results_list)
    
    return results_list, target_pred
    
    
def gather_predictions(results_dir):
    '''
    Looks through all subfolders in a results directory and collects 
    the signature predictions (if we fit model the latest data,
    hence the test set is effectively only 1 point/quarter)
    
    Parameters  results_dir: path where the results are stored
    
    Returns     df_results: dataframe of nowcast results
    '''

    p = (results_dir/'predictions').glob('**/*')
    directories = [x for x in p if x.is_dir()]
    directories.sort()

    results_list = []

    for a_directory in directories:  

        # infer date of the prediction by directory name
        date = [str(a_directory).split('/')[-1]]

        df = pd.read_csv(a_directory/'predictions.csv')

        # add the results on this date to list
        results_list.append(date+list(df.iloc[-1]))

    df_results = pd.DataFrame(results_list, 
                              columns=['date', 'time_index', 
                                       'true_value', 'prediction'])

    df_results.to_csv(f'{str(results_dir)}/all_predictions.csv', index=False)

    return df_results    


def collect_rmse(results_root_dir): 
    '''
    Collect the RMSE in a result directory together and saves a csv of results
    
    Parameters  results_root_dir: the directory where the results are stored
    '''
    
    p = Path(results_root_dir).glob('**/*rmse*')
    files = [x for x in p if x.is_file()]

    all_results = []
    for file in files:
        all_results.append([file.parent.name, pd.read_csv(file)['rmse'][0]])

    df_all_rmse = pd.DataFrame(all_results, columns=['configs', 'rmse'])

    df_all_rmse.to_csv(f"{results_root_dir/'collected_rmse.csv'}", 
                       index=False)

def experiment_rmse(results_dir, config):
    '''
    collect all rmse that can be done separate to analysis, e.g. in case
    the experiments crash half way through
    
    Parameters  results_dir: Path indicating the root directory of where
                             results are saved to
                config: a dictionary containing the keys 'experiment_name'
                        and 'factor_type' to construct the results path
    '''
    results_root_dir = results_dir / config['experiment_name']/ \
    config['factor_type']
    collect_rmse(results_root_dir)
    
    
def find_files(path_names, end_date=None):
    '''
    Looks through a list of paths to find the relevant files for nowcasting
    
    Parameters  path_names: list of paths which include folders and files
                end_date: the end date for nowcasting
    
    Returns     files: list of paths of files to use for nowcasting
    '''
    
    files = [x for x in path_names if x.is_file()]
        
    if end_date:
        end_date = pd.to_datetime(end_date, format="%Y-%m-%d")
        keep_files = [pd.to_datetime(files[i].name[-14:-4], 
                                     format="%Y_%m_%d") \
                      <= end_date for i in range(len(files))]
        files = [i for (i, v) in zip(files, keep_files) if v]
        
    return files


def set_save_folder(configs):
    '''
    Expands a list of configs so that each set of hyperparameters will have
    its own unique folder.
    
    Parameters  configs: the configs used for the experiment to be unpacked
                         for the name of the save folder
    
    Returns     save_folder: the folder name with the expanded dictionary keys
    '''
    
    key_list = ['max_length', 'level', 't_level', 'regularize', 'alpha',
                'fill_method', 'standardize', 'fit_intercept',
                'use_multiplier', 'basepoint']

    if configs['regularize'] == 'elasticnet':
        key_list.append('l1_ratio')

    save_folder = ''
    for key in key_list:
        save_folder += f'{key}_{str(configs[key])}_'

    # Remove extra underscore
    save_folder = save_folder[:-1]
    
    return save_folder


def run_experiments_for_configs(input_dir, results_root, df_gdp, results_list, 
                                prediction_rmse, config, results_folder=None):
    '''
    Fit signature models for each particular set of config
    
    Parameters  input_dir: str or path for the input directory where the 
                           factors are stored
                results_root: Path indicating the root directory of where
                              results are saved to     
                df_gdp: dataframe of the target information
                results_list: a list to append new results from analysis to
                prediction_rmse: a list to append the rmse of the results of 
                                 this config to
                config: dictionary of the configs to be used for the 
                        experiment 
    
    Returns     results_list: an updated list with new entries (the number 
                              of entries is equal to the number of 
                              predictions/nowcasts)
                prediction_rmse: an updated list with a new rmse for this 
                                 config
    
    '''
    
    if config['factor_type'] == 'pca':     
        factor_dir = input_dir/'pca_factors'
        p = Path(factor_dir).glob('**/*')   
    else:
        factor_dir = input_dir/'true_factors'
        p = Path(factor_dir).glob(f"*{config['factor_type']}*")

        # In addition, df_gdp needs slightly different format
        df_gdp['DATE'] = pd.to_datetime(df_gdp['DATE'])
        df_gdp['DATE'] = df_gdp['DATE'].dt.strftime('%Y-%m')

    # find all factor/data files in input_dir
    files = find_files(p, config['end_date']) 

    # define save folder based on configs
    # save_folder = set_save_folder(config)
    
    save_folder = 'signature_results'

    if not results_folder:
        results_folder = results_root / config['experiment_name']/ \
        config['factor_type'] 
        
    results_dir = results_folder / save_folder

    print(f'saving results to {results_dir}')
    results_dir.mkdir(parents=True, exist_ok=True)

    for factor_file in files:

        results_list, target_pred = iteration_on_files(factor_file, 
                                                       df_gdp,
                                                       config, 
                                                       results_list,
                                                       results_dir)

    df_results = gather_predictions(results_dir)
    # Find RMSE of all predictions
    rmse = np.sqrt(np.sum((df_results['true_value']-\
                           df_results['prediction'])**2)/len(df_results))
    # prediction_rmse.append([save_folder, rmse])
    prediction_rmse.append([config, rmse])

#     df_config_rmse = pd.DataFrame([rmse], columns=['rmse'])
#     df_config_rmse.to_csv(results_dir/'rmse.csv', index=False)

#     with open(results_dir/'configs_used.yaml', 'w') as file:
#         yaml.dump(config, file)
        
    return results_list, prediction_rmse


def main(input_dir, results_root, configs):
    '''
    Main function for the US GDP analysis with signatures with hyperparameter
    optimisation
    
    Parameters  input_dir: The directory where the data is stored
                results_root: Path indicating the root directory of where
                              results are saved to  
                configs: a dictionary of the config parameters, which can have
                         lists in some key values if multiple values should be
                         tried for hyperparameter optimisation 
    '''

    df_gdp = prepare_gdp_labels(input_dir)

    results_list = []
    prediction_rmse = []
    
    results_folder = results_root / configs['experiment_name']/ \
    configs['factor_type']
    
    results_folder.mkdir(parents=True, exist_ok=True)
    
    with open(results_folder/'configs_searched.yaml', 'w') as file:
        yaml.dump(configs, file)
    
    configs2 = helper_funcs.modify_dict_hyperparameters(configs)
            
    for config in configs2:    
        results_list, prediction_rmse = \
        run_experiments_for_configs(input_dir, results_root, df_gdp, 
                                    results_list, prediction_rmse, config)
        
    df_results = pd.DataFrame(results_list, columns=['configs', 'train_mean',
                                                     'train_var', 'test_mean',
                                                     'test_var', 'rmse', 
                                                     'directory'])
    #df_results = df_results.sort_values(['configs'])
    print(f'saving collected results to {results_folder}')
    df_results.to_csv(results_folder/'all_sig_results.csv',
                      index=False)

    # collect all rmse   
    df_rmse = pd.DataFrame(prediction_rmse, columns=['configs', 'rmse'])
    df_rmse.to_csv(results_folder/'rmse_results.csv', 
                   index=False)

    
def get_best_configs(results_root, configs):
    '''
    Find the configs the resulted in the least RMSE for the experiment
    
    Parameters  results_root: Path indicating the root directory of where
                              results are saved to  
                configs: dictionary of the configs used for the experiment
                         which may include lists in certain keys for 
                         hyperparameter optimisation, but also specifically 
                         the name of the save folder
    Returns     best_configs: dictionary of the configs that resulted in the
                              minimum RMSE over the hyperparameters searched
    
    '''
    
    output_dir = results_root /configs['experiment_name']/\
    configs['factor_type']

    df = pd.read_csv(f'{output_dir}/rmse_results.csv')
    best_configs = df.loc[np.argmin(df['rmse']), 'configs']
    best_configs = yaml.safe_load(best_configs)
          
#     best_configs = helper_funcs.read_yaml(output_dir/best_setup \
#                                           /'configs_used.yaml')

    best_configs['train_end'] = None
    return best_configs


def rerun_with_configs(input_dir, results_root, configs, new_end_date):
    '''
    Rerun the experiment with a modified end date
    
    Parameters  input_dir: str or path of the root directory where the factors 
                           are stored
                results_root: Path indicating the root directory of where
                              results are saved to  
                configs: dictionary of the configs to be used for the 
                         experiment, it is assumed any hyperparameter would 
                         have been done prior to this, therefore configs 
                         should not contain any lists
                new_end_date: a string to update the end time to, in the 
                              format of YYYY-MM-DD
    
    '''
    
    configs.update({'end_date':new_end_date})
        
    df_gdp = prepare_gdp_labels(input_dir)

    results_list = []
    prediction_rmse = []
    
    results_folder = results_root / configs['experiment_name']/ \
    configs['factor_type']/'rerun'
    
    results_folder.mkdir(parents=True, exist_ok=True)
       
    results_list, prediction_rmse = \
    run_experiments_for_configs(input_dir, results_root, df_gdp, results_list,
                                prediction_rmse, configs, results_folder)
        
    df_results = pd.DataFrame(results_list, columns=['configs', 'train_mean',
                                                     'train_var', 'test_mean',
                                                     'test_var', 'rmse', 
                                                     'directory'])

    print(f'saving collected results to {results_folder}')
    df_results.to_csv(results_folder/'all_sig_results.csv',
                      index=False)

    df_rmse = pd.DataFrame(prediction_rmse, columns=['configs', 'rmse'])
    df_rmse.to_csv(results_folder/'rmse_results.csv', 
                   index=False)


if __name__ == '__main__':
    
    # check for custom config
    if len(sys.argv) > 1:
        yaml_name = sys.argv[1]
    else:
        yaml_name = 'default_sig_config.yaml'
        
    configs = \
    helper_funcs.read_yaml(Path(__file__).resolve().parent/yaml_name)

    input_dir = Path(__file__).resolve().parent.parent/'data'
    print(f'reading from {input_dir}')
    
    results_root = Path(__file__).resolve().parent.parent / 'results'
    print(f'saving results to {results_root}')
    
    if configs['generate_data']:
        print('Generating DFM results and factors')
        
        data_config = {
            'data_file': 'nyfed_data.csv',
            'meta_file': 'nyfed_metadata.csv',
            'pub_lag_info': 'nyfed_indicators.csv',
            'ref_date': '2000-01-01',
            'start_date': '2016-01-01', 
            'end_date': '2019-12-31',
            'global_order': 1,
            'global_multiplicity': 1
            }
        
        generate_dfm(input_dir,results_root, data_config)

    factor_types = configs['factor_type']
    for factor_type in factor_types:
        configs['factor_type'] = factor_type
        if configs['hyperparameter_search']:
            print('Searching for the best set of hyperparameters')

            main(input_dir, results_root, configs)

        if configs['fit_best_configs']:
            print('Evaluating on best config')

            best_configs = get_best_configs(results_root, configs)
            rerun_with_configs(input_dir, results_root, best_configs, 
                               new_end_date = '2019-12-31')
