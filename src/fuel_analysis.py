import matplotlib as mpl
from pathlib import Path
import yaml

plot_config_path = Path(__file__).resolve().parent/'plot_configs.yaml'
with open(plot_config_path, 'r') as stream:
    rc_fonts = yaml.safe_load(stream)
rc_fonts['figure.figsize'] = tuple(rc_fonts['figure.figsize'])

mpl.rcParams.update(rc_fonts)

import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import signature_inference
import signature_helper_funcs as helper_funcs
from sklearn import preprocessing

from sklearn.linear_model import LinearRegression

from pmdarima.arima import auto_arima
from pmdarima.arima import ADFTest

def prepare_fuel_data(input_dir):
    '''
    Prepare fuel data by merging the weekly data from BEIS, daily data on
    BRNT from Yahoo, and exchange rate from investing.com

    Parameters  input_dir: Path where the data files are stored
    Returns     df: a DataFrame which contains the weekly fuel price, the
                    difference in weekly price, daily BRNT prices, and
                    exchange rates

    '''

    df_weekly = pd.read_excel(input_dir/'Weekly_Fuel_Prices_130323.xlsx',
                             sheet_name='All years', skiprows=7)

    df_petrol = df_weekly[['Date', ' ULSP:  Pump price (p/litre)',
                           'ULSP:  Diff on previous WEEK (p/litre)',
                           'Duty rate ULSP (p/litre)',
                           'VAT (% rate) ULSP']].copy()

    df_petrol.columns = ['Date', 'price', 'diff', 'duty', 'vat']

    # next_price or next_diff are the variables of interest as
    # we would like to infer how future prices move
    df_petrol['next_price'] = df_petrol['price'].shift(-1)
    df_petrol['next_diff'] = df_petrol['diff'].shift(-1)

    # load and merge the daily BRNT prices (in USD)
    df_brnt = pd.read_csv(input_dir/'BRNT.L.csv')
    df_brnt2 = df_brnt[['Date', 'Close']].copy()
    df_brnt2.rename({'Close':'BRNT'}, axis=1, inplace=True)
    df_brnt2['Date'] = pd.to_datetime(df_brnt2['Date'])
    df_brnt3 = df_brnt2.set_index('Date')
    df_brnt3 = df_brnt3.resample('D').first().reset_index()

    df_fuel = df_brnt3.merge(df_petrol, how='left')
    df_fuel['day_num'] = [x.dayofweek for x in df_fuel['Date']]

    # add exchange rate and convert BRNT to GBP
    df_currency = pd.read_csv(Path(__file__).resolve().parent.parent/\
                              'data/USD_GBP_Historical_Data.csv')
    df_currency = df_currency[['Date', 'Price']]
    df_currency['Date'] = pd.to_datetime(df_currency['Date'],
                                         format="%d/%m/%Y")

    df_currency = df_currency.rename({'Price':'e_rate'}, axis=1)

    df = df_fuel.merge(df_currency, how='left')
    df['BRNT_gbp'] = df['BRNT']*df['e_rate']

    # Remove rows with incomplete exchange rate data/missing targets
    df = df[:-9]

    return df


def select_analysis_data(df):
    '''
    Select the relevant columns from the data available for analysis
    on the difference in the fuel prices across weeks

    Parameters  df: a DataFrame which contains the weekly fuel price, the
                    difference in weekly price, daily BRNT prices, and
                    exchange rates
    Returns     df_petrol: a DataFrame which contains next_diff, the next
                           change in target that we are predicting, BRNT value
                           and difference, and day of the week.

    '''

    df2 = df.copy(deep=True)

    df2['BRNT_diff'] = df2['BRNT_gbp'].diff()
    df2 = df2.ffill()[4:]

    df_petrol = df2[['Date','BRNT_gbp', 'BRNT_diff', 'next_diff', 'day_num']]
    df_petrol = df_petrol.rename({'Date':'t'}, axis=1)

    # Since the data has a lag, Monday (the current week's price is released
    # on Tuesday, we shift the price forward by 1 and use the shifted price
    df_petrol['price_shift'] = df2['price'].shift()

    df_petrol = df_petrol.bfill()

    # Scale the variables
    scaler = preprocessing.StandardScaler().fit(df_petrol['BRNT_gbp']\
                                                .to_numpy().reshape(-1, 1))
    df_petrol['BRNT_gbp'] = \
    scaler.transform(df_petrol['BRNT_gbp'].to_numpy().reshape(-1, 1))

    scaler = preprocessing.StandardScaler().fit(df_petrol['price_shift']\
                                                .to_numpy().reshape(-1, 1))
    df_petrol['price_shift'] = \
    scaler.transform(df_petrol['price_shift'].to_numpy().reshape(-1, 1))

    return df_petrol


def plot_fuel_data(df, output_dir, ind=10000):
    '''
    Plot the fuel data: weekly pump price from BEIS in pence, daily
    WisdomTree price in USD and in GDP (after conversion).

    Parameters  df: a DataFrame which contains the weekly fuel price, the
                    difference in weekly price, daily BRNT prices, and
                    exchange rates
                output_dir: Path indicating where to save the plot
                ind: an optional argument if only a plot of the more
                     recent data is required
    '''

    plt.figure()
    plt.plot(df['Date'][-ind:], df['BRNT_gbp'][-ind:], label='brent (gbp)')
    plt.plot(df['Date'][-ind:], df['BRNT'][-ind:], label='brent (usd)')
    plt.scatter(df['Date'][-ind:], df['price'][-ind:],
                label='petrol (gbp*100)')
    plt.legend()
    ticks = plt.xticks(rotation=45, ha='right')
    plt.savefig(output_dir/'fuel_data.pdf', bbox_inches='tight')
    plt.close()

def plot_fuel_diffs(input_dir, train_proportion, output_dir):
    
    '''
    Plot the fuel data showing what is being used as train/validation/test

    Parameters  input_dir: Path indicating the directory of the data
                train_proportion: float, the proportion used for train and
                                  validation. We use x**2 for the validation 
                                  set, and x-x**2 for the train set, where x
                                  is the train_proportion
                output_dir: Path indicating where to save the plot
    '''
    
    df_petrol = prepare_weekly_data(input_dir)
    val_end = int(train_proportion*len(df_petrol))
    train_end = int(train_proportion**2*len(df_petrol))

    df = df_petrol[['t', 'next_diff']].copy()
    df.set_index('t', inplace=True)

    df_train = df[:train_end]
    df_val = df[train_end:val_end]
    df_test = df[val_end:]

    plt.figure()
    plt.plot(df_train, label='training data')
    plt.plot(df_val, label='validation data')
    plt.plot(df_test, label='test data')
    plt.ylabel('week-on-week change in fuel price')
    plt.legend()
    ticks = plt.xticks(rotation=45, ha='right')
    plt.savefig(output_dir/f'fuel_train_test.pdf',
                bbox_inches='tight')
    plt.close()


def plot_fuel_results(df_petrol, df_predictions, train_proportion,
                      output_dir, n=1000):
    '''
    Plot the fuel results for the signature method

    Parameters  df_petrol: a DataFrame which contains the target variable and
                           the data available for nowcast
                df_predictions: a DataFrame with the predicted values
                train_proportion: float indicating the proportion used for
                                  training
                output_dir: Path indicating where to save the plot
                n: int, an optional parameter if we did not want to plot the
                   whole test period

    '''

    plt.figure()
    plt.step(df_predictions['t'][:n],
             df_predictions['realised'][:n], label='realised',
            where ='post')
    plt.step(df_predictions['t'][:n],
             df_predictions['next_diff'][:n], label='signature',
            where ='post')
    plt.legend()
    plt.ylabel('nowcast')
    ticks = plt.xticks(rotation=45, ha='right')
    plt.savefig(output_dir/'fuel_results_weekly.pdf', bbox_inches='tight')
    plt.close()


def find_average_error(df_predictions):
    '''
    Find the average error by the day of the week

    Parameters  df_predictions: a DataFrame with the predicted values, and true
                                labels

    Returns     df_results_summary: a DataFrame with the mean and standard
                                    deviation of the residuals by weekday

    '''

    df_predictions2 = df_predictions.copy(deep=True)
    df_predictions2['day_num'] = [x.weekday() for x in df_predictions2['t']]
    df_predictions2['residuals'] = abs(df_predictions2['next_diff']- \
                                      df_predictions2['realised'])

    df_results_summary = df_predictions2[['day_num', 'residuals']]\
.groupby('day_num').agg({'residuals': ['mean', 'std']})

    df_results_summary = df_results_summary.xs('residuals', axis=1,
                                               drop_level=True)
    df_results_summary = df_results_summary.reset_index()

    return df_results_summary

def plot_average_error(df_results_summary, output_dir):
    '''
    Plot the average error by the day of the week

    Parameters  df_results_summary: a DataFrame with the mean and standard
                                    deviation of the residuals by weekday
                output_dir: Path indicating where to save the plot

    '''

    plt.figure()
    plt.plot(df_results_summary['day_num']+1, df_results_summary['mean'])
    plt.xlabel('day of the week')
    plt.ylabel('average absolute error')
    plt.savefig(output_dir/'fuel_error.pdf', bbox_inches='tight')
    plt.close()


def rmse(true_val, predictions):
    '''
    Find the root-mean-square-error (RMSE)

    Parameters  true_val: Series or array, ground truth
                predictions: Series or array, predicted values from a method

    Returns     err: float, the RMSE of the method
    '''
    
    err = np.sqrt(np.sum((predictions-true_val)**2)/len(predictions))
    
    return err
    

def prepare_weekly_data(input_dir):
    '''
    Prepare weekly data for fuel to rename columns, define targets, and also 
    restrict to the period of interest (where we have BRNT prices)

    Parameters  input_dir: Path, the directory where the data is saved to

    Returns     df_petrol: DataFrame, with the columns: t, diff, next_diff
    '''
    
    df_weekly = pd.read_excel(input_dir/'Weekly_Fuel_Prices_130323.xlsx',
                             sheet_name='All years', skiprows=7)

    df_petrol = df_weekly[['Date', ' ULSP:  Pump price (p/litre)',
               'ULSP:  Diff on previous WEEK (p/litre)']].copy()
    df_petrol.columns = ['Date', 'price', 'diff']
    df_petrol['next_price'] = df_petrol['price'].shift(-1)
    df_petrol['next_diff'] = df_petrol['diff'].shift(-1)
    df_petrol = df_petrol.iloc[455:1030].reset_index(drop=True)

    df_petrol = df_petrol.rename({'Date':'t'}, axis=1)
    df_petrol = df_petrol[['t', 'diff', 'next_diff']]

    return df_petrol


def signatures_nowcast(df_petrol, target, configs):
    '''
    Performs the signature nowcast with the SignatureInference class 

    Parameters  df_petrol: DataFrame pf the fuel data (both daily and weekly)
                target: str, the target variable name
                configs: dict, the configuration of the experiments

    Returns     df_predictions: DataFrame of predicted values, the target 
                                ('realised'), day_num indicating the weekday
                rmse_val: float, the RMSE from the signature method
    '''
    
    fuel_analysis = signature_inference.SignatureInference(df_petrol,
                                                           target,
                                                           configs)

    df_predictions = fuel_analysis.apply_signature()

    df_predictions[['realised', 't']] = df_petrol[[configs['target'],
                                                   't']].iloc\
    [int(configs['train_proportion']*len(df_petrol)):].reset_index(drop=True)

    df_predictions['day_num'] = [x.weekday() for x in df_predictions['t']]

    if configs['cast_weekly']:
        df_predictions = \
        df_predictions[df_predictions['day_num']==1]

    rmse_val = rmse(df_predictions[target], df_predictions['realised'])
    #print(f"RMSE from signatures is: {rmse_val}")

    return df_predictions, rmse_val


def cast_to_daily(target_pred, df_petrol, train_end):
    '''
    Since ARIMA models give weekly forecasts, this function casts these into 
    daily results so that the RMSE can be compared with the signature model.

    Parameters  target_pred: DataFrame of the weekly prediction
                df_petrol: DataFrame of the fuel data (both daily and weekly)
                train_end: int, the index for when the training data ends

    Returns     target_pred: Dataframe with additional rows for daily inference
    '''

    target_pred['realised'] = df_petrol.loc[train_end:,'next_diff'].values
    target_pred = target_pred.set_index('t', drop=True)
    target_pred = target_pred.resample('D').first()
    target_pred['next_diff_lagged'] = target_pred['next_diff'].shift()
    target_pred = target_pred.ffill()
    target_pred = target_pred.bfill()
    target_pred = target_pred.reset_index()

    return target_pred


def ar1(input_dir, train_proportion, cast_daily=False):
    '''
    Using an AR(1) model with intercept for the fuel prediction

    Parameters  input_dir: Path, the directory where the data is saved to
                train_proportion: float, the proportion used for training
                cast_daily: bool, whether to cast these weekly results into a 
                            pseudo-daily prediction (i.e. constant forecasts 
                            until the next observation).

    Returns     target_pred: Dataframe of predicted values from AR(1)
    '''

    df_petrol = prepare_weekly_data(input_dir)
    train_end = int(train_proportion*len(df_petrol))

    observation_train = df_petrol['diff'][:train_end]
    observation_train = np.array(observation_train).reshape(-1, 1)
    target_train = df_petrol['next_diff'][:train_end]

    observation_test = df_petrol['diff'][train_end:]
    observation_test = np.array(observation_test).reshape(-1, 1)
    target_test = df_petrol['next_diff'][train_end:]

    reg = LinearRegression(fit_intercept=True).fit(observation_train,
                                                   target_train)

    target_train_pred = pd.DataFrame(reg.predict(observation_train),
                                 columns=[target_train.name])
    res_train = (target_train.values-target_train_pred.values.T[0])
    target_pred = pd.DataFrame(reg.predict(observation_test),
                               columns=[target_test.name])

    target_pred['t'] = df_petrol['t'][train_end:].values
    target_pred['realised'] = df_petrol.loc[train_end:,'next_diff'].values
    target_pred['residuals'] = target_pred['realised'] - \
    target_pred['next_diff']

    if cast_daily:
        target_pred = cast_to_daily(target_pred, df_petrol, train_end)

        rmse_val = rmse(target_pred['realised'],
                        target_pred['next_diff_lagged'])
    else:
        rmse_val = rmse(target_pred['realised'], target_pred['next_diff'])

    print(f'RMSE from AR(1) is {rmse_val}')


    return target_pred


def plot_comparison(df_predictions, target_pred, label, output_dir,
                    color='tab:green'):
    '''
    Plot the nowcast results, comparing the signature against another method

    Parameters  df_predictions: DataFrame of the results from the signature
                                method
                target_pred: DataFrame of the results from an alternative 
                             method
                label: str, a suffic for the name of the plot file
                output_dir: Path, the output directory to save the figure
                color: str, the colour to be used for plotting for alternative 
                       method
    '''

    plt.plot()
    plt.step(df_predictions['t'],
             df_predictions['realised'], label='realised', where='post')
    plt.step(df_predictions['t'],
             df_predictions['next_diff'], label='signature', where='post')
    plt.step(target_pred['t'],
             target_pred['next_diff'], label=label, where='post',
             color=color)
    plt.legend()
    ticks = plt.xticks(rotation=45, ha='right')

    plt.savefig(output_dir/f'fuel_baselines_{label}.pdf',
                bbox_inches='tight')
    plt.close()


def autoarima(input_dir, train_proportion, cast_daily=False):
    '''
    Automatically select a ARIMA model by minimising the AIC 

    Parameters  input_dir: Path, the directory where the data is saved to
                train_proportion: float, the proportion used for training
                cast_daily: bool, whether to cast these weekly results into a 
                            pseudo-daily prediction (i.e. constant forecasts 
                            until the next observation).

    Returns     target_pred: Dataframe of predicted values from autoarima
    '''

    df_petrol = prepare_weekly_data(input_dir)
    train_end = int(train_proportion*len(df_petrol))

    df = df_petrol[['t', 'next_diff']].copy()
    df.set_index('t', inplace=True)

    df_train = df[:train_end]
    df_test = df[train_end:]

    arima_model = auto_arima(df_train, start_p=0, d=None, start_q=0,
                             max_p=5, max_d=5, max_q=5, start_P=0,
                             seasonal=False, error_action='warn',
                             trace=True, suppress_warnings=True,
                             stepwise=True, random_state=20, n_fits=50,
                             with_intercept=True)

    predictions = []
    prediction = arima_model.predict(n_periods=1).values

    for i in range(len(df_test)):
        predictions.append(prediction)
        arima_model.update(df_test.iloc[i], maxiter=0)
        prediction = arima_model.predict(n_periods=1)

    target_pred = pd.DataFrame(predictions, columns=['next_diff'])

    target_pred['t'] = df_test.index
    target_pred['realised'] = df_test.values
    target_pred['residuals'] = target_pred['realised'] - \
    target_pred['next_diff']

    if cast_daily:
        target_pred = cast_to_daily(target_pred, df_petrol, train_end)

        rmse_val = rmse(target_pred['realised'],
                        target_pred['next_diff_lagged'])
    else:
        rmse_val = rmse(target_pred['realised'], target_pred['next_diff'])

    print(f'RMSE from autoarima model is {rmse_val}')

    return target_pred


def plot_residuals(df_predictions, df_ar1, df_autoarima, output_dir):
    '''
    Plot the residuals, comparing the signature against AR(1) and autoarima

    Parameters  df_predictions: DataFrame of the results from the signature
                                method
                df_ar1: DataFrame of the results from AR(1)
                df_autoarima: DataFrame of the results from autoarima
                output_dir: Path, the output directory to save the figure
    '''

    plt.figure()
    plt.plot(df_predictions['t'],
             df_predictions['residuals'], label='signatures',
             color='tab:orange')
    plt.plot(df_ar1['t'],
             df_ar1['residuals'], label='AR(1)', color='tab:green')
    plt.plot(df_autoarima['t'],
             df_autoarima['residuals'], label='autoarima',
             color='tab:purple')
    plt.ylabel('residuals')
    plt.legend()
    ticks = plt.xticks(rotation=45, ha='right')
    plt.savefig(output_dir/f'fuel_residuals.pdf',
            bbox_inches='tight')
    plt.close()


if __name__ == '__main__':

    # check for custom config
    if len(sys.argv) > 1:
        yaml_name = sys.argv[1]
    else:
        yaml_name = 'default_fuel_configs.yaml'

    all_configs = \
    helper_funcs.read_yaml(Path(__file__).resolve().parent/yaml_name)
    target = all_configs['target']
    train_proportion = all_configs['train_proportion']
    
    all_configs = helper_funcs.modify_dict_hyperparameters(all_configs)

    input_dir = Path(__file__).resolve().parent.parent/'data'
    output_dir = Path(__file__).resolve().parent.parent/'results'/'fuel'

    Path(input_dir).mkdir(parents=True, exist_ok=True)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    df = prepare_fuel_data(input_dir)

    plot_fuel_data(df, output_dir)

    df_petrol = select_analysis_data(df)

    plot_fuel_diffs(input_dir, all_configs[0]['train_proportion'], output_dir)

    results_list = []

    df_petrol_val = df_petrol.iloc\
    [:int(all_configs[0]['train_proportion']*len(df_petrol))]

    for configs in all_configs:

        _, rmse_val = signatures_nowcast(df_petrol_val, target, configs)

        results_list.append([configs, rmse_val])

    df_results = pd.DataFrame(results_list, columns=['configs', 'rmse'])

    best_ind = np.argmin(df_results['rmse'])

    best_configs = df_results.loc[best_ind, 'configs']

    print(f'Best configs is: {best_configs}')
    print(f'val RMSE: {df_results.loc[best_ind, "rmse"]}')

    df_predictions, rmse_val = signatures_nowcast(df_petrol, target,
                                                  best_configs)

    print(f"RMSE from signatures is: {rmse_val}")
    
    df_predictions['residuals'] = df_predictions['realised']-\
    df_predictions[target]

    plot_fuel_results(df_petrol, df_predictions, train_proportion,
                      output_dir)
    
    df_results_summary = find_average_error(df_predictions)
    plot_average_error(df_results_summary, output_dir)
    
    df_ar1 = ar1(input_dir, train_proportion,
                 cast_daily=all_configs[0]['cast_daily'])
    df_autoarima = autoarima(input_dir, train_proportion,
                             cast_daily=all_configs[0]['cast_daily'])

    plot_comparison(df_predictions, df_ar1, label='AR(1)',
                    output_dir=output_dir, color='tab:green')
        
    plot_comparison(df_predictions, df_autoarima, label='autoarima',
                    output_dir=output_dir, color='tab:purple')

    plot_residuals(df_predictions, df_ar1, df_autoarima, output_dir)
