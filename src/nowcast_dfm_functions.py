import numpy as np
import pandas as pd
import json
from datetime import datetime, timedelta
import calendar

def load_json(json_file):
    """
    Load JSON config file.

    Parameters
    ----------
    json_file : String
        String containing path to JSON file.

    Returns
    -------
    config : JSON object
        Configuration parameters.

    """
    with open(json_file) as f:
        config = json.load(f)
    return config


def gen_factor_vars(input_dir, config_data, config_meta, config_pub_lag):
    """
    Generates variables required for DFM from files detailing the 
    data/metadata/publication lags

    Parameters
    ----------
    input_dir : Path
        Path of the input data/files
    config_data : String
        Str containing config filename
    config_meta : String
        Str containing metadata filename
    config_pub_lag : String
        Str containing publication lag filename

    Returns
    -------
    vars_df: pd.DataFrame
        DataFrame containing all variables
    pub_lag_df: pd.DataFrame
        DataFrame containing publication lag of all variables
    vars_dict: dict
        Dictionary mapping variables to factors
    vars_meta: pd.DataFrame
        Datafrane containing variable metadata
    factor_blocks_list
        Array containing factor blocks

    """
    vars_df = pd.read_csv(input_dir/config_data)

    vars_df['DATE'] = pd.to_datetime(vars_df['DATE'], format='%Y-%m-%d')
    vars_df = vars_df.set_index(['DATE'])

    vars_meta = pd.read_csv(input_dir/config_meta,
                            usecols=['Assigned to', 'Factor grouping',
                                     'Global', 'Soft', 'Real', 'Labour'])

    vars_meta.columns = ['var', 'factor', 'global_', 'soft', 'real', 'labour']

    local_conditions = [
        (vars_meta['soft'] == 1),
        (vars_meta['real'] == 1),
        (vars_meta['labour'] == 1)
    ]

    local_values = ['Soft',
                    'Real',
                    'Labour',
                    ]

    vars_meta['local_factor'] = np.select(local_conditions, local_values)
    vars_meta['local_factor'] = np.where(vars_meta.local_factor == '0', 
                                         'No factor block',
                                         vars_meta.local_factor)
    vars_meta['global_'] = np.where(vars_meta.global_ == 1, 'Global',
                                    vars_meta.global_)

    # construct list of factors dictionary
    vars_dict = {row['var']: [row['global_'], row['factor'],
                              row['local_factor']]
                 for i, row in vars_meta.iterrows()}

    factor_blocks = np.array(list(vars_dict.values()))
    # generate arrays of "local" factors, drop "core" factors
    factor_blocks_local = np.delete(factor_blocks, 1, axis=1)
    factor_blocks_local = np.unique(factor_blocks_local, axis=0)

    factor_blocks_list = [factor_blocks_local]

    pub_lag_df = pd.read_csv(input_dir/config_pub_lag,
                             usecols=['pub_lag', 'VariableName_in_File'])

    # rename and remove missing PMI data
    pub_lag_df.columns = ['pub_lag', 'var']
    pub_lag_df = pub_lag_df.dropna().reset_index(drop=True)

    return vars_df, pub_lag_df, vars_dict, vars_meta, factor_blocks_list


def format_for_dfm(input_df, start_date):
    """
    Formats monthly indicators and quarterly target dates as required for
    sm.tsa.DynamicFactorMQ

    Parameters
    ----------
    input_df: pd.DataFrame
        DataFrame containing all variables
    start_date: str
        String detailing the start date in %Y-%m-%d format

    Returns
    -------
    endog_m_nyfed: pd.DataFrame
        DataFrame containing all monthly variables indexed with appropriate
        date format
    endog_q_nyfed:
        Series containing quarterly target indexed with appropriate date 
        format

    """
    # match format required for model
    input_df.reset_index(inplace=True)
    input_df['DATE'] = pd.to_datetime(input_df['DATE'])
    vars_df_m = input_df.set_index('DATE').to_period("M").drop('GDPC1',
                                                               axis=1)
    vars_df_q = input_df.set_index('DATE').to_period("Q")

    endog_m_nyfed = vars_df_m.loc[start_date:, :]
    endog_q_nyfed = vars_df_q.loc[start_date:, 'GDPC1']
    # remove non-quarterly values for uniquely valued index
    endog_q_nyfed = endog_q_nyfed.dropna()

    return endog_m_nyfed, endog_q_nyfed


def gen_horizon_intervals(start_val=0, repeats=7, step=52):
    """
    Generates an array detailing weeks to number of days in the year

    Parameters
    ----------
    start_val: int
        Defaults to 0, i.e. first week in the year
    repeats: int
        Defaults to 52, i.e. 52 weeks in the year
    step: int
        Defaults to 7, i.e. 7 days in a week

    Returns
    -------
    horizon_date: np.array
        Array detailing weeks to number of days in the year

    """
    horizon_date = start_val + np.arange(repeats) * step

    return horizon_date


def year_quarter_to_ymd(q_string, date_sel='first'):
    """
    Converts current quarter to either first day of the last month in that 
    quarter (default), or last day of last month in that quarter

    Parameters
    ----------
    q_string : str
        String detailing current quarter
    date_sel: str
        Defaults to 'first', only used for selecting the first day of the 
        month

    Returns
    -------
    dt : datetime object

    """
    # convert quarterly string to datetime value representing first/last day
    # of last month of quarter
    parts = q_string.upper().split('Q')

    # for selecting first day of last month:
    if date_sel == 'first':
        dt = datetime(int(parts[0]), int(parts[1]) * 3, 1)

    # for selecting last day of last month:
    if date_sel == 'last':
        end_val = calendar.monthrange(int(parts[0]), int(parts[1]) * 3)[1]
        dt = datetime(int(parts[0]), int(parts[1]) * 3, end_val)

    return dt


def gen_end_of_q_datetime(input_horizon):
    """
    Generates end of quarter datetime value, depending on year_quarter_to_ymd
    arguments

    Parameters
    ----------
    input_horizon : datetime object
        Details current horizon

    Returns
    -------
    horizon_q : datetime object

    """
    # working with quarters is v.messy, using dfs and above string idea as a
    # workaround
    horizon_df = pd.DataFrame({'date': pd.to_datetime(input_horizon.strftime('%Y-%m'))}, index=[0])
    horizon_df['quarter'] = pd.PeriodIndex(horizon_df.date, freq='Q')

    horizon_q = horizon_df['quarter']
    horizon_q = f'{horizon_q[0]}'
    horizon_q = year_quarter_to_ymd(horizon_q)

    return horizon_q


def gen_data_for_horizons(input_df, input_pub_lags, input_start_date, input_horizon, df_vars_to_update):
    """
    For a given horizon, generates what data would be available at that point
    in time.

    Parameters
    ----------
    input_df: pd.DataFrame
        DataFrame containing all variables
    input_pub_lags: pd.DataFrame
        DataFrame containing publication lags for each variable
    input_start_date: datetime object
        Details start of sample
    input_horizon: datetime object
        Details current horizon
    df_vars_to_update: pd.DataFrame
        DataFrame that updates with available information for each horizon

    Returns
    -------
    obj : JSON object
        desc

    """
    # loop through to generate info available at horizon for each variable
    for column in input_df:

        # fetch publication lag specific to each variable
        pub_lag = input_pub_lags.loc[input_pub_lags['var'] == column,
                                     'pub_lag'].item()
        df_vars_to_update[column] = \
        input_df[column].loc[input_start_date:input_horizon]

        latest_release = df_vars_to_update.index[-1] + timedelta(int(pub_lag))

        if latest_release > input_horizon:

            # convert to str and drop all parts of str that aren't year/month
            latest_release_year_month = f'{latest_release}'[:-12]
            latest_release_year_month = \
            datetime.strptime(latest_release_year_month, '%Y-%m')
            horizon_year_month = f'{input_horizon}'[:-12]
            horizon_year_month = datetime.strptime(horizon_year_month,
                                                   '%Y-%m')

            # calc diff in terms of no. of months between horizon and last 
            # possible release
            time_diff = \
            int(pd.to_timedelta([latest_release_year_month - \
                                 horizon_year_month \
                                 + timedelta(1)]).astype('timedelta64[M]')[0])

            # if latest_release > horizon (i.e. not yet available for that 
            # month), force last months' value
            if time_diff == 0:
                time_diff = 1

            # insert NAs for that many months back to represent which series
            # is available at a given horizon
            # additional -1 to account for last set of horizons being outside
            # the range of available data
            df_vars_to_update[column].iloc[len(df_vars_to_update[column]) \
                                           - time_diff:] = np.nan

    return df_vars_to_update
