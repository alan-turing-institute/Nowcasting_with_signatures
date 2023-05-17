import numpy as np
import pandas as pd
from statsmodels.multivariate.pca import PCA

def factor_arrays_to_dict(input_factor_blocks, input_factor_loadings):
    """
    Converts array to list, flattens and then generates a dictionary that 
    maps the loading to the factor.

    Parameters
    ----------
    input_factor_blocks: np.array
        Array containing factor blocks
    input_factor_loadings: list
        List of np.arrays containing factor loadings

    Returns
    -------
    factor_dict: dict
        Dictionary mapping loading to factor group

    """
    # convert np.array to list, flatten, and put into dict
    input_factor_blocks = input_factor_blocks[:, [1]].tolist()
    input_factor_blocks = [i[0] for i in input_factor_blocks]

    factor_dict = dict(zip(input_factor_blocks, input_factor_loadings))

    return factor_dict


def generate_factor_loadings(input_df, factor_dict, input_factor_blocks):
    """
    Maps variables to their respective factors and generates the loading.

    Parameters
    ----------
    input_df: pd.DataFrame
        DataFrame containing all variables
    factor_dict: dict
        Dictionary mapping variables to factors
    input_factor_blocks: np.array
        Array containing factor blocks

    Returns
    -------
    factor_loadings: dict
        Dictionary mapping loading to factor group

    """
    factor_loadings = []

    for i in np.arange(len(input_factor_blocks)):

        # pull out required factor block
        current_factor_block = input_factor_blocks[i][1]

        # pull out variables from 'factors' dict that match specified factor
        # block and save into new dict
        select_vars = {}
        for k, v in factor_dict.items():
            values = [i for i in v if current_factor_block in i]
            if values:
                select_vars[k] = values

        # convert to list and intersect with explanatory variables df
        select_vars = list(select_vars.keys())
        input_df_select_vars = \
        input_df[input_df.columns.intersection(select_vars)]

        # run PCA on selected factors
        # interpolate a handful of missing values still present in the data
        # this exact line happens within sm.tsa.DynamicFactorMQ()
        # statsmodels.multivariate.pca has the option to fill in missing 
        # values via EM
        # following Chad Fulton, we use interpolation
        input_df_factors = pd.DataFrame(input_df_select_vars).interpolate()\
        .fillna(method='backfill').values
        # consider changing normalize = True - variables after transformation
        # are not similar in range
        res_pca = PCA(input_df_factors, ncomp=1, method='eig',
                      normalize=False)

        # append factor loadings into an array
        factor_loadings.append(res_pca.factors)

    factor_loadings = factor_arrays_to_dict(input_factor_blocks, 
                                            factor_loadings)

    return factor_loadings


def generate_global_factor(input_df):
    """
    Generates global factors of all variables

    Parameters
    ----------
    input_df: pd.DataFrame
        DataFrame containing all variables

    Returns
    -------
    global_dict: dict
        Dictionary containing global factors

    """
    input_df = input_df._get_numeric_data()
    input_df_factors = pd.DataFrame(input_df).interpolate()\
    .fillna(method='backfill').values
    # consider changing normalize = True - variables after transformation are
    # not similar in range
    res_pca_global = PCA(input_df_factors, ncomp=1, method='eig',
                         normalize=False)

    factor_loadings = res_pca_global.factors
    global_dict = {'Global': factor_loadings}

    return global_dict


def append_factor_loadings(input_df, factor_dict, input_factor_blocks_list):
    """
    Generates factor loadings for all variables and their respective groups,
    as detailed in factor_dict.

    Parameters
    ----------
    input_df: pd.DataFrame
        DataFrame containing all variables
    factor_dict: dict
        Dictionary mapping variables to factors
    input_factor_blocks_list: list
        Singular list containing array of factor blocks

    Returns
    -------
    factor_loadings_df: pd.DataFrame
        DataFrame containing loadings on each factor

    """
    factor_loadings_all = {}

    for input_factor_blocks in input_factor_blocks_list:
        factor_loadings = generate_factor_loadings(input_df, factor_dict, 
                                                   input_factor_blocks)
        factor_loadings_all.update(factor_loadings)

    # generate the global factor and update dict
    global_factor = generate_global_factor(input_df)
    factor_loadings_all.update(global_factor)

    # convert to pd.df for ease of use
    factor_loadings_df = pd.DataFrame()

    for i in np.arange(len(factor_loadings_all)):
        values = [*factor_loadings_all.values()][i].flatten()
        factor_loadings_df[f'{[*factor_loadings_all.keys()][i]}'] = values

    # append in dates as index
    factor_loadings_df = factor_loadings_df.set_index([input_df.index],
                                                      append=True)
    factor_loadings_df.drop("No factor block", axis=1, inplace=True)

    return factor_loadings_df
