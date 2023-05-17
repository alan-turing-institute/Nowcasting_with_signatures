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
import matplotlib.pyplot as plt

def nyfed_published_values(input_dir, method='backcast', start_q='2016Q1',
                           end_q='2021Q4'):
    '''
    Prepare data from the published NYFed Staff Forecast, downloadable from
    https://www.newyorkfed.org/research/policy/nowcast
    
    Parameters  input_dir: str/Path of the directory of the data
                method: either backcast previous quarter or nowcast current 
                        quarter
                start_q: String in the form YYYYQX, for X in {1, 2, 3, 4} 
                         indicating the start of the period of interest
                end_q: String in the form YYYYQX, for X in {1, 2, 3, 4} 
                       indicating the end of the period of interest
    Returns     df_nyfed: dataframe with nowcast/backcast from NY Fed  
    '''
    
    df_nyfed = pd.read_excel(f'{str(input_dir)}/'\
                             'New-York-Fed-Staff-Nowcast_data_2002-'
                             'present.xlsx',
                             sheet_name='Forecasts By Horizon', skiprows=13,
                             usecols=['Forecast date', 'Reference quarter',
                                      'Backcast\n(previous quarter)',
                                      'Nowcast\n(current quarter)'])

    df_nyfed = df_nyfed.rename({'Backcast\n(previous quarter)':'backcast',
                                'Nowcast\n(current quarter)':'nowcast',
                                'Forecast date': 'date',
                                'Reference quarter': 'quarter'}, axis=1)

    df_nyfed['date'] = pd.to_datetime(df_nyfed['date'])
    df_nyfed = df_nyfed[['date', 'quarter', method]].dropna()
    
    # Reduce to period of interest
    df_nyfed = df_nyfed[df_nyfed['quarter'].between(start_q, end_q)]
    df_nyfed.reset_index(drop=True, inplace=True)
    
    return df_nyfed


def plot_usgdp(df_sig, df_dfm, df_nyfed, ind1, ind2, save_dir=None,
               filename=None):
    '''
    Plot US GDP results between specified indices to compare the signature
    method against our dynamic factor model and the published values from the 
    NY Fed.
    
    Parameters  df_sig: dataframe with the true GDP values and the signature
                        results
                df_dfm: dataframe with the dynamic factor model result
                df_nyfed: dataframe with the publised NY Fed values
                ind1: integer indicating the start index
                ind2: integer indicating the end index
                save_dir: Path object indicating directory to save to, if left
                          as default None, then the plot is not saved, and the
                          filename is redundent
                filename: string specifying the filename    
    '''
        
    plt.step(df_sig['date'][ind1:ind2], df_sig['true_value'][ind1:ind2], 
             marker = None, label='GDP', markersize=12, linewidth=3,
             where='post')
    plt.step(df_sig['date'][ind1:ind2], df_sig['prediction'][ind1:ind2], 
             marker = '.', label='Sigs', markersize=12, linewidth=1.5,
             where='post')
    plt.step(df_nyfed['date'][ind1:ind2], df_nyfed['nowcast'][ind1:ind2], 
             marker = 'x', label='NYFed', markersize=12, linewidth=1.5,
             where='post')
    plt.step(df_dfm['date'][ind1:ind2], df_dfm['dfm_prediction'][ind1:ind2],
             marker = '^', label='DFM', markersize=12, linewidth=1.5, 
             where='post')
    plt.xticks(rotation=30)
    plt.ylabel('value')
    plt.legend()
    
    if save_dir:
        if not filename:
            filename = 'USGDP_results'
        
        plt.savefig(f'{save_dir}/{filename}.pdf', bbox_inches='tight')
    
    plt.show()
    
    
def find_rmse(true_values, predictions, ind1, ind2):
    '''
    Finds the RMSE between 2 specified indices with the end points included
    
    Parameters  true_values: pandas Series of true values 
                predictions: pandas Series of predicted values 
                ind1: integer indicating the start index
                ind2: integer indicating the end index
    Returns     rmse: float, the root mean square error 
    
    '''
    rss = np.sum((true_values[ind1:ind2+1]-predictions[ind1:ind2+1])**2)
    rmse = np.sqrt(rss/(ind2-ind1+1))
    
    return rmse


def plot_config(config_folder, experiment, ind1=0, ind2=100, save_dir=None, 
                filename=None, input_dir=None, results_dir=None):
    
    '''
    plots the results for a particular setting
    
    Parameters  config_folder: string giving the folder name of the specific 
                               config was saved in
                experiment: string giving the experiment
                ind1: integer indicating the start index
                ind2: integer indicating the end index
                save_dir: Path object indicating directory to save to, if left
                          as default None, then the plot is not saved, and the
                          filename is redundent
                filename: string specifying the filename 
                input_dir: Path giving the directory of the data, defaults to
                           a folder called "data"
                results_dir: Path of where the results are stored, defaults to
                             a folder called "results"
    Returns     df_rmse: a dataframe of rmse
    
    '''
    
    if not input_dir:
        input_dir = Path(__file__).resolve().parent.parent/'data'
    if not results_dir:
        results_dir = Path(__file__).resolve().parent.parent/'results'
    
    df_nyfed = nyfed_published_values(input_dir, method='nowcast')
    dfm_file = results_dir/'dfm'/'dfm_predictions.yaml'

    with open(dfm_file, 'r') as stream:
            dfm_results = yaml.safe_load(stream)
    df_dfm = pd.DataFrame(dfm_results.items(), columns=['date', 
                                                        'dfm_prediction'])

    df_sig = pd.read_csv(results_dir/experiment/config_folder\
                         /'all_predictions.csv')
    df_sig['date'] = pd.to_datetime(df_sig['date'], format="%Y_%m_%d")
    df_dfm['date'] = pd.to_datetime(df_dfm['date'], format="%Y-%m-%d")
    
    plot_usgdp(df_sig, df_dfm, df_nyfed, ind1, ind2, save_dir=save_dir,
               filename=filename)
    
    all_rmse = {}
    
    rmse_sig = find_rmse(df_sig['true_value'], df_sig['prediction'], ind1,
                         ind2)
    all_rmse['Sig'] = rmse_sig
    
    rmse_dfm = find_rmse(df_sig['true_value'], df_dfm['dfm_prediction'], ind1,
                         ind2)
    all_rmse['DFM'] = rmse_dfm

    rmse_ny = find_rmse(df_sig['true_value'], df_nyfed['nowcast'], ind1, ind2)
    all_rmse['NYFed'] = rmse_ny
    
    df_rmse = pd.DataFrame(all_rmse.items(), columns=['method', 'rmse'])
    
    return df_rmse


def plot_average_quarter(results_dir, config_folder, experiment, 
                         save_dir=None, filename=None):
    '''
    Plots the average error as the quarter progresses
    
    Parameters  results_dir: Path where the results are saved
                config_folder: string of config folder name
                experiment: string giving the name of the experiment
                save_dir: Path object indicating directory to save to, if left
                          as default None, then the plot is not saved, and the
                          filename is redundent
                filename: string specifying the filename 
    '''
    
    df_sig = pd.read_csv(results_dir/experiment/config_folder/\
                         'all_predictions.csv')
    
    df_sig['date'] = pd.to_datetime(df_sig['date'], format="%Y_%m_%d")
    df_sig['quarter'] = df_sig['date'].dt.to_period('Q')
    df_sig['q_week'] = df_sig.groupby('quarter').cumcount()+1
    df_sig['error'] = np.abs(df_sig['true_value'] - df_sig['prediction'])
    
    plt.plot(df_sig[['error', 'q_week']].groupby('q_week').mean())
    
    if save_dir:
        if not filename:
            filename = 'USGDP_results'
        
        plt.savefig(f'{save_dir}/{filename}.pdf', bbox_inches='tight')