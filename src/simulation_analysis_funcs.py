import matplotlib as mpl
from pathlib import Path
import yaml

plot_config_path = Path(__file__).resolve().parent/'plot_configs.yaml'
with open(plot_config_path, 'r') as stream:
    rc_fonts = yaml.safe_load(stream)
rc_fonts['figure.figsize'] = tuple(rc_fonts['figure.figsize'])
        
mpl.rcParams.update(rc_fonts)

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from scipy import stats

import signature_helper_funcs as helper_funcs
import compute_linear_sigs as sig_funcs

np.random.seed(1234)


def transform_prod(val, configs):
    '''
    Alter the values according to the transformation on the input data to use
    in the Kalman filter
    
    Parameters    val: float, original value
                  configs: dictionary, configurations of the experiment
    
    Returns       transformed_val : float, transformed value
    
    '''
    
    if not configs['nonlinear_transform']:
        transform_val = 1
    else:
        if configs['data_transform'] == 'sigmoid':
            if val < 2*np.finfo(float).eps:
                val = 2*np.finfo(float).eps
            transform_val = 1/(val*(1-val))        

        elif configs['data_transform'] == 'exp':
            transform_val = np.log(val)

        elif configs['data_transform'] == 'log':
            transform_val = np.exp(val)

    return transform_val

def kalman_filter(yt, configs, start=None):
    '''
    Implements the Kalman filter
    
    Parameters    yt: DataFrame of all the observed values and the time at 
                      which they are observed 
                  configs: dictionary, configurations of the experiment
                  start: float, if there is a specified starting value
    
    Returns       xt_filter : list, the mean of the filter
    
    '''
        
    y_values = yt['y1']
    ts = yt['t']

    if start:
        xt_filter = [start]
    else:
        xt_filter = [y_values[0]]
    
    F = -1
    R =(F+np.sqrt(F**2+configs['var1']*configs['H']**2))/(configs['H']**2)

    for i in range(1, len(yt)):
        increment = (F-R*configs['H']**2)*xt_filter[i-1]*(ts[i]-ts[i-1])\
                    +R*configs['H']*transform_prod(y_values[i], configs)\
                    *(y_values[i]-y_values[i-1])
        xt_filter.append(xt_filter[i-1]+increment)
    return xt_filter


def recursive_kf(yt, xt, configs):
    '''
    Repeatedly calls the Kalman filter so that the filter is computed over a
    window with a maximum length (i.e. this is an expanding window up to the 
    maximum size, and then a rolling window after that)
    
    Parameters    yt: DataFrame of all the observed values and the time at 
                      which they are observed 
                  xt: DataFrame of the hidden, target values 
                  configs: dictionary, configurations of the experiment
    
    Returns       kf_vals : list, the mean of the filter for the window at each
                            point
    
    '''
    
    
    kf_vals = kalman_filter(yt[:configs['max_length']], configs, 
                            start=xt['x1'][0])
    
    for i in range(configs['max_length'], len(yt)):
        start_ind = i-configs['max_length']+1
        yt2 = yt[start_ind:i+1].reset_index(drop=True)
        xt_filter = kalman_filter(yt2, configs, 
                                  start=xt['x1'][start_ind])
        kf_vals.append(xt_filter[-1])

    return kf_vals


def normalise_variance(input_dir, train_proportion, output_dir):
    
    '''
    Normalise the variance of the signature method with that of the Kalman 
    filter and plot the QQ plot to see how well the distributions align
    
    Parameters    input_dir: Path or str giving the directory to read the 
                             residuals from
                  train_proportion: float, proportion of paths used for 
                                    training
                  output_dir: Path or str giving the directory to save the 
                              plots to
    
    '''
    
    df_res = pd.read_csv(f'{input_dir}/res_sigs.csv')
    df_var = pd.read_csv(f'{input_dir}/filter_vars.csv')

    df_res_norm = df_res['res'].div(np.sqrt(df_var['variance']\
                                            .iloc[-len(df_res):]\
                                            .reset_index(drop=True)))

    print(f'Mean of normalised residual is {df_res_norm.mean()},'
          f'and variance is {df_res_norm.var()}')    

    plt.figure()
    stats.probplot(df_res_norm, plot=plt, dist="norm")
    plt.savefig(f'{output_dir}/sig_qq.pdf', bbox_inches='tight')
    plt.close()

    if output_dir:
        df_res.to_csv(f'{output_dir}/res_sigs_normalised.csv', index=False)


def find_x_labels(input_dir, num_samples):
    '''
    Reads all the files relating to the target and extract the last value to 
    give a dataframe of target values for regression
    
    Parameters    input_dir: Path, the directory where the simulated data is
                             saved to
                  num_samples: int, number of simulated paths
    
    Returns       df_x : DataFrame of all the final values of the target in 
                         each simulated path
    
    '''
    
    x_vals = np.zeros(num_samples)
    for index in range(num_samples):
        df = pd.read_csv(input_dir/f'xt_sample{index}.csv')
        x_vals[index] =df['x1'].values[-1]
        df_x = pd.DataFrame(x_vals, columns=['x1'])

    return df_x        
        

def analysis(input_dir, configs, save_dir, normalise_var=False):
    '''
    Main analysis function for the signature method, this computes the
    signatures on all paths, splits the data into train/test set and performs
    the regression.
    
    Parameters    input_dir: Path, the directory where the simulated data is
                  configs: dictionary of the configurations of the experiment
                  save_dir: Path, the directory to save all the results 
                  normalised_var: bool, whether to normalise the variance and 
                                  plot the QQ plots
                                  
    
    '''
    
    results_list = []
    
    print('Prediction using sigs')
    
    yt_sigs = compute_all_sigs(input_dir, configs, prefix=None, 
                               fill_method='ffill')
        
    xt = find_x_labels(input_dir, configs['num_samples'])
    xt.to_csv(save_dir/'true_values.csv', index=False)    

    yt_sigs.to_csv(save_dir/'sig_data.csv', index=False)

    xt_train, yt_train, xt_test, yt_test = \
    helper_funcs.split_data_proportion(xt, yt_sigs, 
                                       train_proportion=\
                                       configs['train_proportion'])

    xt_train.to_csv(save_dir/'true_values_train.csv', index=False)
    xt_test.to_csv(save_dir/'true_values_test.csv', index=False)

    yt_train.to_csv(save_dir/'sig_data_train.csv', index=False)
    yt_test.to_csv(save_dir/'sig_data_test.csv', index=False)
    
    results_list, pred = helper_funcs.regress(yt_train, xt_train['x1'], 
                                              yt_test, xt_test['x1'],
                                              configs, 
                                              results_list=results_list,
                                              save=save_dir)

    res = xt_test.values-pred.values
    print(f'Mean is {res.mean()}, and variance is {res.var()}')    
     
    df_res = pd.DataFrame(res, columns=['res'])
    df_res.to_csv(f'{save_dir}/res_sigs.csv', index=False)
    
    if normalise_var:
        normalise_variance(save_dir, train_proportion, save_dir)
        
    df_results = pd.DataFrame(results_list, columns=['configs', 'train_mean',
                                                     'train_var', 'test_mean',
                                                     'test_var', 'rmse',
                                                     'directory'])
    
    df_results.to_csv(save_dir/'results_table.csv', index=False)

    
def compute_all_sigs(input_dir, configs,
                     prefix=None, fill_method='ffill'):
    
    '''
    Computes the signatures of all files in a directory. Assumes that time is 
    the zeroth index.
    
    Parameters  input_dir: a Path object of where the observation data are
                           stored
                configs: dictionary with the signature keywords such as
                    num_samples: number of trajectories
                    level: truncation level of the signature
                    keep_sigs: whether to keep only the innermost signatures,
                               all linear signatures or all signatures
                    prefix: a prefix for the new columns of signature terms
                    fill_method: method to fill the dataframe (not needed at 
                                 the moment)
                    t_level: the level of truncation of the time parameter if 
                             different to level
                prefix: str, an optional argument to add a prefix to the
                        signature columns
                fill_method: data imputation method
                
    Returns     df_sigs: a dataframe with the relevant signature features
    
    '''

    all_sigs = []
    
    level = configs['level']

    if not configs['t_level']:
        t_level = level
    else:
        t_level = configs['t_level']
        
    for index in range(configs['num_samples']):
        df = pd.read_csv(input_dir/f'yt_sample{index}.csv')
        df_x = pd.read_csv(input_dir/f'xt_sample{index}.csv')

        dim = len(df.columns)
        
        sigs = sig_funcs.subframe_sig_comp(df, df_x, dim, t_level, configs)
        
        all_sigs.append(sigs)
        
    df_sigs = sig_funcs.make_sig_df(all_sigs, dim, t_level, configs)
    df_sigs = sig_funcs.select_signatures(df_sigs, dim, t_level, configs)
        
    print('Num sigs', len(df_sigs.columns))
    
    return df_sigs    
    
    
def compare_methods(results_dir, train_proportion, save=None, 
                    normalised=False):
    '''
    Compare the residuals of signatures against the Kalman filter by plotting 
    their histograms on the same scale. Also plot the two residuals against 
    each other and fitting a line of best fit with regression

    Parameters    results_dir: Path, the directory where results are saved
                  train_proportion: float, the proportion of paths used in the
                                    training set
                  save: Path or str giving the full path to the save directory
                  normalised: bool, whether the residuals have been normalised

    '''

    res_kf = pd.read_csv(results_dir/'res_kf.csv')
    # Filter for only those that are in the test set for sigs
    ind = int(np.floor(train_proportion*len(res_kf)))
    res_kf = res_kf[ind:].reset_index(drop=True)
    
    if normalised:
        res_sigs = pd.read_csv(results_dir/'res_sigs_normalised.csv')
    else:
        res_sigs = pd.read_csv(results_dir/'res_sigs.csv')
        
    reg = LinearRegression().fit((res_kf), (res_sigs))

    print(f'The line of best fit has gradient {reg.coef_[0]} ',
          f'and intercept {reg.intercept_}.\n',
          f'The R^2 value is {reg.score(res_kf, res_sigs)}')
    
    plt_end = np.max([res_sigs.max(), res_kf.max()])
    plt_start = np.min([res_sigs.min(), res_kf.min()])
    
    freq_sig, _ = np.histogram(res_sigs, bins=20)
    freq_kf, _ = np.histogram(res_kf, bins=20)
    y_max = np.max([freq_sig.max(), freq_kf.max()])+3
    
    plt.figure()
    sig_hist = plt.hist(res_sigs, bins=20, range=[plt_start, plt_end])
    plt.xlabel('residuals', fontsize=28)
    plt.ylabel('frequency', fontsize=28)
    plt.xticks(fontsize=26, rotation=45)
    plt.yticks(fontsize=26)
    plt.ylim(ymin=0, ymax=y_max)
    if save is not None:
        plt.savefig(f'{save}/sig_res.pdf', bbox_inches='tight')
    plt.close()
    
    plt.figure()
    kf_hist = plt.hist(res_kf, bins=20, range=[plt_start, plt_end])
    plt.xlabel('residuals', fontsize=28)
    plt.ylabel('frequency', fontsize=28)
    plt.xticks(fontsize=26, rotation=45)
    plt.yticks(fontsize=26)
    plt.ylim(ymin=0, ymax=y_max)
    
    if save is not None:
        plt.savefig(f'{save}/kf_test_res.pdf', bbox_inches='tight')
        res_kf.to_csv(f'{save}/kf_test_res.csv', index=False)
    plt.close()
            
    plt.figure()
    plt.scatter((res_kf), (res_sigs))
    plt.xlabel('Residual of Kalman Filter')
    plt.ylabel('Residual of regression on signatures')
    x = np.linspace(np.min((res_kf['res'])), np.max((res_kf['res'])), 100)
    y = reg.intercept_ + reg.coef_[0]*x
    plt.plot(x,y, linewidth=2, color='red')
    lim = np.max(np.abs([res_kf, res_sigs]))
    plt.ylim(ymin=-lim, ymax=lim)
    plt.xlim(xmin=-lim, xmax=lim)
    plt.gca().set_aspect('equal', adjustable='box')
    if save is not None:
        plt.savefig(f'{save}/comparison.pdf', bbox_inches='tight')
    plt.close()
    

def filter_example(input_dir, index, output_dir, configs):
    '''
    Apply the Kalman filter recursively and plot the filter mean for a fair 
    illustration when comparing to signatures of a rolling window

    Parameters    input_dir: Path, the directory where the data are saved
                  index: int, the path index to plot
                  output_dir: Path or str giving the full path to the results
                              directory
                  configs: dict, the configurations of the experiment to use 
                           for fitting the Kalman filter

    '''

    xt = pd.read_csv(f'{input_dir}/xt_sample{index}.csv')
    yt = pd.read_csv(f'{input_dir}/yt_sample{index}.csv')

    xt_filter = recursive_kf(yt, xt, configs)

    plt.figure()
    plt.plot(yt['t'], xt_filter, label='filter')
    plt.plot(xt['t'], xt['x1'], label='true values')
    plt.plot(yt['t'], yt['y1'], label='observed values')
    plt.legend()
    plt.savefig(f'{output_dir}/filter_{str(index)}.pdf', bbox_inches='tight')
    plt.close()
    
    df_mean = pd.DataFrame(xt_filter, columns = ['kf_mean'])
    df_mean.to_csv(f'{output_dir}/kf_mean_{str(index)}.csv', index=False)

def filter_results(input_dir, output_dir, configs):

    '''
    Fit a Kalman filter for each path, and save the residuals, and the QQ plot
    of the residuals

    Parameters    input_dir: Path, the directory where the data are saved
                  output_dir: Path or str giving the full path to the results
                              directory
                  configs: dict, the configurations of the experiment to use 
                           for fitting the Kalman filter


    '''

    residuals = np.zeros(configs['num_samples'])

    all_variances = []

    all_kf_pred = []

    for index in range(configs['num_samples']):
        xt = pd.read_csv(f'{input_dir}/xt_sample{index}.csv')
        yt = pd.read_csv(f'{input_dir}/yt_sample{index}.csv')

        xt_filter = kalman_filter(yt, configs)

        residuals[index] = (xt['x1'].values[-1]-xt_filter[-1])
        all_kf_pred.append(xt_filter[-1])

    df_kf = pd.DataFrame(all_kf_pred, columns = ['kf'])
    df_kf.to_csv(f'{output_dir}/kf_estimates.csv', index=False)

    ind = int(np.floor(configs['train_proportion']*len(df_kf)))
    df_kf.iloc[:ind].to_csv(f'{output_dir}/kf_estimates_train.csv',
                            index=False)
    df_kf.iloc[ind:].to_csv(f'{output_dir}/kf_estimates_test.csv',
                            index=False)

    print(f'Mean of residual is {residuals.mean()}, ',
          f'and variance is {residuals.var()}')
    df_res = pd.DataFrame(residuals, columns=['res'])
    df_res.to_csv(f'{output_dir}/res_kf.csv', index=False)

    plt.figure()
    stats.probplot(residuals, plot=plt, dist="norm")
    plt.savefig(f'{output_dir}/kf_qq.pdf', bbox_inches='tight')
    plt.close()

    plt.figure()
    plt.hist(residuals, bins=30)
    plt.savefig(f'{output_dir}/kf_res.pdf', bbox_inches='tight')
    plt.close()
    

def plot_example(input_dir, index, output_dir, configs):

    '''
    Plot an example of the inferred trajectories from the Kalman filter and the 
    signature method

    Parameters    input_dir: Path, the directory where the data are saved
                  index: int, the path index to plot
                  output_dir: Path or str giving the full path to the results
                              directory
                  configs: dict, the configurations of the experiment

    '''

    print(f'Plotting from {input_dir}')

    xt = pd.read_csv(f'{input_dir}/xt_sample{index}.csv')
    yt = pd.read_csv(f'{input_dir}/yt_sample{index}.csv')
    df_mean = pd.read_csv(f'{output_dir}/kf_mean_{index}.csv')    

    x_pred = helper_funcs.apply_saved_model(yt, xt, configs, output_dir)
 
    plt.figure()
    plt.plot(yt['t'], df_mean['kf_mean'], label='KF predictions', 
             marker = 'x', linestyle= '-.', linewidth=2)
    plt.plot(yt['t'], x_pred, label='sig predictions', marker='*', 
             linestyle=':', linewidth=2)
    plt.plot(xt['t'], xt['x1'], label='hidden target $Y_t$', linewidth=2)
    plt.plot(yt['t'], yt['y1'], label='observed values $X_t$', 
             marker = 'o', linestyle= '--', linewidth=2)
    plt.xlabel('time')
    plt.ylabel('value')
    plt.legend()
    plt.savefig(f'{output_dir}/prediction_example_{str(index)}.pdf',
                bbox_inches='tight')
    plt.close()
