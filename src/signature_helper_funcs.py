import itertools
import pickle
import yaml

import numpy as np
import pandas as pd
from statsmodels.multivariate.pca import PCA
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn import preprocessing
from pathlib import Path
import compute_linear_sigs as sig_funcs

def reduce_dim(df, k, factor_structure=None, fill_method='backfill'):
    '''
    Reduce dimension of the input variables by PCA (optionally specifying
    structure)
    
    Parameters  df: the dataframe to reduce dimension of
                k: an integer or a list of integers with the number of 
                   principal components to find
                factor_structure: an optional input to the function which
                                  specifies structure to compute principal
                                  components from
                fill_method: method by which to fill in the missing data (only 
                             backfill/fill-em are implemented)
    Returns     df_pca: a dataframe of the principal components
    '''

    fill = None
    if fill_method=='backfill':
        # forward fill data first before filling in missing data at start
        df = df.ffill()
        df = df.bfill()
    else:
        fill = 'fill-em'

    if not factor_structure:
        res_pca = PCA(df, ncomp=k, method='eig', normalize=False,
                      missing=fill)
        df_pca = res_pca.factors
        df_pca.columns = [f'global_{i}' for i in range(k)]
    else:
        all_pca = []
        # find k principal components of each subgroup (k can be scalar or
        # vector)
        if not isinstance(k, (list, tuple, np.ndarray)):
            k = np.repeat(k, len(factor_structure))
        for a_k, var_subset in zip(k, factor_structure):
            res_pca = PCA(df[factor_structure[var_subset]], ncomp=a_k, 
                          method='eig', normalize=False, missing=fill)
            df_pca = res_pca.factors
            df_pca.columns = [f'{var_subset}_{i}' for i in range(a_k)]
            all_pca.append(df_pca)

        df_pca = pd.concat(all_pca, axis=1)
    return df_pca


def reduce_dim2(observation_train, observation_test, k, factor_structure=None, fill_method='backfill'):
    '''
    Reduce dimension of the input variables by PCA (optionally specifying
    structure)
    
    Parameters  observation_train: the dataframe of observed values in the
                                   training set to reduce the dimension of
                observation_test: the dataframe of observed values in the
                                   training set to reduce the dimension of
                k: an integer or a list of integers with the number of 
                   principal components to find
                model_params: dictionary of all the model parameters
                factor_structure: an optional input to the function which
                                  specifies structure to compute principal
                                  components from
                fill_method: method by which to fill in the missing data (only 
                             backfill/fill-em are implemented)
    Returns     df_pca: a dataframe of the principal components for both train
                        and test set
    '''

    fill = None
    if fill_method=='backfill':
        # forward fill data first before filling in missing data at start
        df = df.ffill()
        df = df.bfill()
    else:
        fill = 'fill-em'

    if not factor_structure:
        res_pca = PCA(observation_train, ncomp=k, method='eig',
                      normalize=False, missing=fill)
        df_pca = res_pca.factors
        df_pca.columns = [f'global_{i}' for i in range(k)]
       
        df_pca_test = pd.DataFrame(np.dot(observation_test, 
                                      res_pca.eigenvecs),
                               index=observation_test.index)
        df_pca_test.columns = [f'global_{i}' for i in range(k)]
        df_pca = pd.concat([df_pca, df_pca_test]) 
        
    else:
        all_pca = []
        # find k principal components of each subgroup (k can be scalar or
        # vector)
        if not isinstance(k, (list, tuple, np.ndarray)):
            k = np.repeat(k, len(factor_structure))
        for a_k, var_subset in zip(k, factor_structure):
            res_pca = PCA(observation_train[factor_structure[var_subset]],
                          ncomp=a_k, method='eig', normalize=False,
                          missing=fill)
            df_pca = res_pca.factors
            df_pca.columns = [f'{var_subset}_{i}' for i in range(a_k)]
            
            df_pca_test = pd.DataFrame(np.dot(observation_test, 
                                              res_pca.eigenvecs),
                                       index=observation_test.index)
            df_pca_test.columns = [f'{var_subset}_{i}' for i in range(k)]
            df_pca = pd.concat([df_pca, df_pca_test])
            
            all_pca.append(df_pca)
            

        df_pca = pd.concat(all_pca, axis=1)
    return df_pca


def apply_pca(df, model_params, df_grouping, target_full):
    '''
    Main function to apply the signature method
    
    Parameters  df: input dataframe of data available at current time/horizon
                model_params: dictionary containing all necessary parameters 
                              for fitting the model
                df_grouping: dataframe containing factor groupings
                target_full: the filled target data
    Returns     df: dataframe after dimension reduction
    '''
    if df_grouping: 
        factor_structure = {}
        for group in df_grouping['factor_group'].unique():
            new_group = {str(group): \
                         list(df_grouping[df_grouping['factor_group']\
                                          ==group].factor_name)}
            factor_structure.update(new_group)
    else: 
        factor_structure = None
    
    df_obs = df.drop(model_params['target'], axis=1)
    
    # Slightly hacky, df here only contains observations, and we
    # do not need the target so just use df for both arguments below
    _, observation_train, _, observation_test = split_data(df_obs, df_obs, 
                                                           model_params)

    df = reduce_dim2(observation_train, observation_test,
                    model_params['k'],
                    factor_structure=factor_structure, 
                    fill_method=model_params['pca_fill_method'])
    df[model_params['target']] = target_full
    
    return df


def modify_dict_hyperparameters(configs):
    '''
    Deconstruct configs into a list of configs if key values contain lists in 
    the original config. This is for the purpose of hyperparameter 
    optimisation
    
    Parameters  configs: a dictionary of the config parameters, with lists in 
                         some key values if multiple values should be tried 
                         for hyperparameter optimisation
    Returns     modified: a list of configurations, with cartesian products on
                          the keys where values are lists    
    '''
    
    modified = []
    
    var = {k: i for k, i in configs.items() if isinstance(i, list)}
    fixed = {k: i for k, i in configs.items() if k not in var}      

    if bool(var):
        keys_var, values_var = zip(*var.items())
        new = [dict(zip(keys_var, v)) for v in itertools.product(*values_var)]

        modified = [{**fixed, **i} for i in new]
    else:
        modified.append(configs)
                           
    return modified


def split_data(target, observation, model_params, prediction_offset=None,
               offset_type=None, target_freq='Q'):
    
    '''
    split the data by either specified date or proportion
    Parameters  target: dataframe of prediction target data
                observation: dataframe of observed predictors
                model_params: parameters of the model, if no prediction_offset
                prediction_offset: integer indicating amount to offset the 
                                   target variable by to determine the cutoff 
                                   for the end date. Note this can be 0, 
                                   otherwise split via proportion is assumed.
                offset_type: the unit that prediction_offset refers to, 
                             accepts
                target_freq: frequency of the target, this determines an end 
                             date to ensure that nowcasts for the same period 
                             are split correctly into the test set
    Returns     target_train: training data for target variable
                observation_train: training data of observed variables 
                target_test: test data for target variable
                observation_test: test data of observed variables 
    '''
    
    if prediction_offset is not None:
        if offset_type == 'year':
            offset = pd.DateOffset(years=prediction_offset)

        elif offset_type == 'month':
            offset = pd.DateOffset(months=prediction_offset)

        elif offset_type == 'day':
            offset = pd.DateOffset(days=prediction_offset)
            
        else:
            warnings.warn("offset type not specified, defaulting to days")
            offset = pd.DateOffset(days=prediction_offset)
            
        prediction_q = str((model_params['t']- offset)\
                           .to_period(target_freq))
        
        target_train, observation_train, target_test, observation_test = \
        split_data_range(target, observation, train_end=prediction_q)
    else:
        target_train, observation_train, target_test, observation_test = \
        split_data_proportion(target, observation,
                              model_params['train_proportion'])
        
    return target_train, observation_train, target_test, observation_test
        

def split_data_range(target, observation, train_end=None):
    '''
    split the data so that only data from prediction quarter or specified
    training end date is in the test set
    Parameters  target: dataframe of prediction target data
                observation: dataframe of observed predictors
                train_end: an optional input to specify training end date
    Returns     target_train: training data for target variable
                observation_train: training data of observed variables 
                target_test: test data for target variable
                observation_test: test data of observed variables 
    '''

    if not (train_end and pd.to_datetime(train_end)< observation.index[-1]):
        train_end = str(observation.index[-1].to_period('Q'))
        
    ind = np.where(observation.index < pd.to_datetime(train_end))[0][-1]    
    target_train = target[:ind+1]
    observation_train = observation[:ind+1]
    target_test = target[ind+1:]
    observation_test = observation[ind+1:]

    return target_train, observation_train, target_test, observation_test


def split_data_proportion(target, observation, train_proportion=0.8):
    '''
    split the data into training and test set by specified proportion
    Parameters  target: dataframe of prediction target data
                observation: dataframe of observed predictors
                train_proportion: proportion of data for the training set
    Returns     target_train: training data for target variable
                observation_train: training data of observed variables 
                target_test: test data for target variable
                observation_test: test data of observed variables 
    '''
    
    ind = int(np.floor(train_proportion*len(target)))
    target_train = target[:ind]
    observation_train = observation[:ind]
    target_test = target[ind:]
    observation_test = observation[ind:]
    
    return target_train, observation_train, target_test, observation_test


def apply_model(reg, observation, target):
    '''
    Apply a fitted regression model to data
    
    Parameters  reg: regression model
                observation: dataframe of observed predictors (signatures)
                target: dataframe of prediction target data
    Returns     target_pred: predicted values for the target using the model
                observation_train: training data of observed variables 
                res: residual between prediction and true target    
    '''
    # target is a series, so name returns the name of the feature
    target_pred = pd.DataFrame(reg.predict(observation), 
                                     columns=[target.name])
    res = (target.values-target_pred.values.T[0])
    
    return target_pred, res


def regress(observation_train, target_train, observation_test, target_test, 
            model_params, results_list=[], save=None):
    '''
    Perform regression of the train/test data 
    
    Parameters  observation_train: training data of observed variables 
                target_train: training data for target variable
                observation_test: test data of observed variables 
                target_test: test data for target variable
                model_params: dictionary containing model parameters
                results_list: list of current results (if running iteratively)
    Returns     results_list: updated list of results 
                target_pred: dataframe with predicted value(s) for target
    '''
    if model_params['standardize']:
        scaler = preprocessing.StandardScaler().fit(observation_train)
        observation_train = pd.DataFrame(scaler.transform(observation_train),
                                         index=observation_train.index)
        observation_test = pd.DataFrame(scaler.transform(observation_test),
                                        index=observation_test.index)
    max_iter = 1000
    if model_params['regularize'] == 'l2' and model_params['alpha'] != 0.0:
        reg = Ridge(alpha=model_params['alpha'],
                    fit_intercept=model_params['fit_intercept'],
                    max_iter=max_iter).fit(observation_train, target_train)
        
    elif model_params['regularize'] == 'l1' and model_params['alpha'] != 0.0:
        reg = Lasso(alpha=model_params['alpha'],
                    fit_intercept=model_params['fit_intercept'],
                    max_iter=max_iter).fit(observation_train, target_train)
        
    elif model_params['regularize'] == 'elasticnet' and \
    model_params['alpha'] != 0.0:
        reg = ElasticNet(alpha=model_params['alpha'], 
                         fit_intercept=model_params['fit_intercept'],
                         l1_ratio=model_params['l1_ratio'], 
                         max_iter=max_iter).fit(observation_train,
                                                target_train)
        
    else: 
        reg = LinearRegression(fit_intercept=model_params['fit_intercept'])\
        .fit(observation_train, target_train)
    
    target_train_pred, res_train = apply_model(reg, observation_train, 
                                               target_train)
    
    rmse = np.sqrt(np.sum(res_train**2)/len(res_train))
    
    target_pred, res = apply_model(reg, observation_test, target_test)
    
    if save: 
        save_dir = Path(__file__).resolve().parent.parent/'results'/save
        print(f"saving models to {save_dir}")
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        filename = f'{save_dir}/signature_model.pkl'
        pickle.dump(reg, open(filename, 'wb'))
        
        if model_params['standardize']:
            pickle.dump(scaler, open(f'{save_dir}/scaler.pkl', 'wb'))
        
        df_train = pd.DataFrame(np.array([target_train.values, 
                                          target_train_pred.values.T[0]]).T,
                                columns=['true_value', 'prediction'])
        df_train.index = observation_train.index
        df_train = df_train.reset_index()
        df_train.rename(columns={'index':'time'}, inplace=True)
        df_train.to_csv(f'{str(save_dir)}/train_predictions.csv', index=False)
    
        df_test = pd.DataFrame(np.array([target_test.values, 
                                         target_pred.values.T[0]]).T, 
                               columns=['true_value', 'prediction'])
        df_test.index = observation_test.index
        df_test = df_test.reset_index()
        df_test.rename(columns={'index':'time'}, inplace=True)
        df_test.to_csv(f'{str(save_dir)}/predictions.csv', index=False)
        results_list.append([model_params, res_train.mean(), 
                             res_train.var(), res.mean(), res.var(), 
                             rmse, str(save).split('/')[-1]])
    
    else:
        results_list.append([model_params, res_train.mean(), 
                             res_train.var(), res.mean(), res.var(), rmse])
    
    return results_list, target_pred


def find_signatures(df_merge, model_params):
    '''
    Finds the signature terms from the merged dataframe
    
    Parameters  df_merge: dataframe of both observed and target data
                model_params: a dictionary of the model parameters
    Returns     observation_sigs: signatures of the observed data
    '''

    observation_filled = df_merge.drop(model_params['target'], axis=1)
    # Convert the time index to days
    if model_params['window_type'] == 'days':
        observation_filled['t'] = (observation_filled['t']-
                                    observation_filled['t'][0]).dt.days.values

    target = df_merge[['t', model_params['target']]].copy()
    if model_params['window_type'] == 'days':
        target.set_index('t', inplace=True)
    observation_sigs = sig_funcs.compute_sigs_dates(observation_filled, 
                                                    model_params, target)

    return observation_sigs


def append_previous_target(df, observations, model_params):
    '''
    Append the last known value of the target variable into the dataframe
    of the signatures
    
    Parameters  df: input dataframe of data available at current time/horizon
                observations: dataframe containing the observed data or the
                              signatures of the observed data to which we 
                              append the previous target value
                model_params: Contains model parameters, in particular:
                              - target: target variable to nowcast
                              - target_lag: the number of days to apply the
                                            shift in target to obtain the 
                                            previous value (depends on 
                                            publication lag)
                              - window_type: whether we are taking a rolling
                                             window according to number of
                                             days or number of rows
    Returns     observations: dataframe now with the previous target values
                              appended
    '''
    
    df_shift = df[['t', model_params['target']]]
    df_shift = find_prev_value(df_shift, model_params)
    # Assume that the first value of the hidden process is known
    df_shift['recent_target'] = \
    df_shift['recent_target'].fillna(method='bfill')
    observations['recent_target'] = df_shift['recent_target'].values
    
    return observations


def find_prev_value(df_merge, model_params):
    
    '''
    Find the most recent value of the target variable with a specified shift
    
    Parameters  df: input dataframe of data available at current time/horizon
                model_params: Contains model parameters, in particular:
                              - target: target variable to nowcast
                              - target_lag: the number of days to apply the
                                            shift in target to obtain the 
                                            previous value (depends on 
                                            publication lag)
                              - window_type: whether we are taking a rolling
                                             window according to number of
                                             days or number of rows
    Returns     df_results: a dataframe with predicted value(s)    
    '''
    
    df_merge2 = df_merge.copy()
    target = model_params['target']
    shift = model_params['target_lag']+1
    
    if model_params['window_type'] == 'days':
        df_merge2.index = df_merge2['t']
        df_merge2['date'] = pd.to_datetime(df_merge2['t'].dt.date)
        df_temp = df_merge2[['date', target]].groupby('date').first()
        df_temp.index = pd.to_datetime(df_temp.index)
        df_temp = df_temp.rolling(f'{int(shift)}d', min_periods=2).\
        agg(lambda x:x[0])
        df_temp = df_temp.reset_index()
        df_temp = df_temp.rename({target:'recent_target'}, axis=1)
        
        df_merge2 = df_merge2.merge(df_temp)
        df_merge2 = df_merge2.drop('date', axis=1)
        df_merge2['recent_target'] = \
        df_merge2['recent_target'].fillna(method='ffill')
        
    elif model_params['window_type'] == 'ind':
        df_merge2['recent_target'] = df_merge2[target].shift(shift)
    
    return df_merge2


def fit_model_range(observation_sigs, df_merge, results_dir, configs,
                    results_list=[]):
    '''
    Fit a regression model by using all available data as the training data
    
    Parameters  observation_sigs: dataframe of signatures on the observed data
                df_merge: dataframe of both observed and target data
                results_dir: path indicating the results directory to save to
                configs: a dictionary of the config parameters
                results_list: list of current results (across iterations)
    Returns     results_list: updated list of results
                target_pred: dataframe with predicted value(s) for target
    '''
        
    target_full = df_merge[configs['target']]

    target_train, observation_train, target_test, observation_test = \
    split_data_range(target_full, observation_sigs,
                     train_end=configs['train_end'])
    
    results_list, target_pred = regress(observation_train, target_train,
                                        observation_test, target_test, 
                                        configs, results_list, results_dir)

    return results_list, target_pred


def apply_saved_model(observations, target, model_params, output_dir):
    
    '''
    Applies a saved model onto specified data 
    
    Parameters  observations: dataframe of observations
                target: dataframe of target data
                output_dir: path indicating where the saved model is found
                model_params: a dictionary of the config parameters
    Returns     target_pred: dataframe with predicted value(s) for target
    '''

    loaded_model = pickle.load(open(f'{output_dir}/signature_model.pkl',
                                        'rb'))

    observation_sigs = sig_funcs.compute_sigs_dates(observations, 
                                                    model_params, target)

        
    if model_params['standardize']:
        scaler = pickle.load(open(f'{output_dir}/scaler.pkl', 'rb'))

        observation_sigs = pd.DataFrame(scaler.transform(observation_sigs),
                                                 index=observations.index)

    target_pred = pd.DataFrame(loaded_model.predict(observation_sigs), 
                              columns=[model_params['target']])

    return target_pred


def read_yaml(filepath):
    '''
    A simple function to read in a yaml file
    
    Parameters  filepath: str or Path, filepath of the required yaml file
    Returns     loaded_yaml: dictionary of loaded object
    
    '''
    
    with open(filepath, 'r') as stream:
        loaded_yaml = yaml.safe_load(stream)
        
    return loaded_yaml