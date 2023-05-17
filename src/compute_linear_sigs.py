import warnings
import numpy as np
import pandas as pd
import itertools
import iisignature as iisig
import esig
esig.set_backend("iisignature")

def linear_sig_keys(dim, level):
    '''
    Creates the keys of only the linear signatures which has fewer terms and
    hence signatures of higher depth can be computed
    The first element in sig keys represents the variable for the innermost 
    integral.
    Note this is currently indexed by 1 to start, in order to match with esig.
    If want to be 0 indexed as python, the indices can be adjusted afterwards.
    
    Parameters  dim: the dimension of the features
                level: truncation level of the signature
                
    Returns     keys: a list of signature keys/names (by an index)   

    '''

    keys = ['']

    if level >= 1:
        for i in range(1, dim+1):
            keys.append(str(i))

    for i in range(2, level+1):
        pre = []
        for k in range(2, dim+1):
            pre.append(str(k))
        
        keys_sub = keys[-(dim-1)*(i-1)-1:]
        post = ''
        for k in range(i-1):
            post+=',1'
            
        keys = keys + ['1,'+key for key in keys_sub] + [key+post for key in \
                                                        pre]

    keys = ['('+key+')' for key in keys]

    return keys


def all_sig_keys(dim, level):
    '''
    Create a list of names for signatures, with the first entry corresponding
    to the innermost integration dimension.
    
    Parameters  dim: the dimension of the features
                level: truncation level of signature terms
                
    Returns     keys: a list of names for signature terms
    
    '''
    keys = []
    for i in range(level+1):
        prod = list(itertools.product(np.arange(1, dim+1), repeat=i))
        prod = [str(x) for x in prod]
        keys.append(prod)
        
    keys = [i for s in keys for i in s]
        
    return keys


def find_single_sig_index(dim, index, level):
    '''
    Find the indices for signature of 1 particular variable. 
    This variable is in the dimension given by index on original path.
    
    Parameters  dim: the dimension of the features
                index: the position of the variable of interest (from 0 to
                       dim-1) 
                level: truncation level of the signature
                
    Returns     ind: a list of indices corresponding to signatures of the
                     variable of interest (including the constant 1 term at
                     the 0th order), e.g. if index=1, those corresponding to
                     signatures (), (1), (1,1), ...
    '''

    ind = [0]
    base_ind = 0
    for i in range(0, level):
        base_ind += dim**(i)
        increment = (index)*base_ind
        ind.append(base_ind + increment)
    return ind


def compute_linear_sigs(dim, level):
    '''
    Wrapper function to compute the indices of all linear signature terms 
    Note it is vital for time index to be at 0 as this is assumed here.
    
    Parameters  dim: the dimension of the features
                level: truncation level of the signature
                
    Returns     ind: a list of signature indices corresponding to ``linear''
                     signatures
    '''

    ind1 = find_innermost_sigs(dim, level)
    ind2 = find_outermost_sigs(dim, level)
    ind3 = find_middle_sigs(dim, level)
    
    # convert to set to remove duplicates caused by inner and outermost sig 
    # functions
    ind = list(set(ind1+ind2+ind3))
    ind.sort() 

    return ind
    

def find_innermost_sigs(dim, level):
    '''
    Find the indices corresponding to signature terms where only where the 
    innermost integral is with respect a variable which is not time 
    (index of the time variable is assumed to be 0)
    
    Parameters  dim: the dimension of the features
                level: truncation level of the (innermost) signature
                
    Returns     ind: a list of indices corresponding to signatures of the 
                     form S(x...)
    '''

    ind = [0]

    base_ind = 0
    for i in range(level):
        for j in range(dim):
            ind.append(base_ind + dim**i+j*dim**i)

        base_ind += dim**i
    return ind


def find_outermost_sigs(dim, level):
    '''
    Find the indices corresponding to signature terms where only where the 
    outermost integral is with respect a variable which is not time 
    (index of the time variable is assumed to be 0)
    
    Parameters  dim: the dimension of the features
                level: truncation level of the signature
                
    Returns     ind: a list of indices corresponding to signatures of the 
                     form S(...x)
    '''

    ind = [0]

    base_ind = 0
    for i in range(level):
        for j in range(dim):
            ind.append(base_ind + dim**i+j)

        base_ind += dim**i
    return ind


def find_middle_sigs(dim, level):
    '''
    Find the indices of the linear signature terms where the integral with 
    respect to variable (not time) is not the innermost or the outermost one 
    (where index of the time variable is assumed to be 0).
    
    Parameters  dim: the dimension of the features
                level: truncation level of the signature
                
    Returns     ind: a list of indices corresponding to ``linear'' signatures
                     of the form S(...x...) (which are not ``innermost'' or
                     ``outermost'' signatures).
    '''

    ind = []
    base = 1+dim+dim**2
    
    for i in range(3, level+1):
        for j in range(1, i-1):
            for k in range(1, dim):
                ind.append(base + k*dim**j)

        base += dim**i

    return ind


def find_linear_sig_features(dim, var_level, t_level=None, 
                             keep_sigs='innermost'):
    '''
    Filters the linear signatures so that the terms in purely time (t) can be
    of a different level to other variables. This assumes that the list of 
    signatures has been filtered down to linear signatures only from all the 
    signatures, and forms an additional filter.
    
    Parameters  dim: the dimension of the features
                var_level: truncation level of signature terms of features
                           (not time)
                t_level: a potentially different truncation level for 
                         signatures of time
                keep_sigs: whether to keep all the linear signatures or just
                           the innermost
                
    Returns     ind: a list of indices corresponding to the relevant 
                     signatures

    '''
    if keep_sigs == 'innermost':
        if not t_level:
            t_level = var_level

        ind = [0]

        base = 0
        for i in range(1, t_level+1):
            base += (dim-1)*(i-1)+1
            ind.append(base)
            
        base = 2-dim
        for i in range(1, var_level+1):
            base += (dim-1)*(i)+1
            for j in range(dim-1):
                ind.append(base+j)
                
        ind = list(set(ind))
        ind.sort()
        
    elif keep_sigs == 'all_linear':
        ind = list(range(int(1+dim*var_level+0.5*(dim-1)*var_level*\
                             (var_level-1))))
        if t_level < var_level:
            remove_ind = [1 + np.sum(dim + (dim-1)*np.arange(i+1)) for i in \
                          range(t_level-1, var_level-1)]            
            ind = list(set(ind).difference(remove_ind))
            
        elif t_level > var_level:
            add_ind = [1 + np.sum(dim + (dim-1)*np.arange(i+1)) for i in \
                          range(var_level-1, t_level-1)]            
            ind = list(set(ind).union(add_ind))
        
    else:
        warnings.warn("Linear signature set not specified")
        ind = []

    return ind


def multiplier_for_t_terms(sigs, multiplier, dim, t_level):
    '''
    Multiply signatures corresponding to time only by specified multiplier
    
    Parameters  sigs: the precomputed signature terms (where it is assumed 
                      that time is at index 0 for the original data)
                multiplier: specified multiplier which is typically `starting
                            value' of target variable 
                dim: the dimension of the feature set
                t_level: the truncation level of t_level
                
    Returns     sigs: modified signatures with the time terms multiplied by
                      the specified multiplier 
    '''
    
    # Create a logical mask for time terms and use it to set a scaling for
    # those times. 
    # Note that this is applied to a complete signature dataframe (i.e. level 
    # =t_level) and higher truncation orders of t filtered out later
    t_terms_index = find_single_sig_index(dim=dim, index=0, level=t_level)
    t_terms = [i in t_terms_index for i in np.arange(len(sigs))]

    # Apply multiplier for the linear case
    multipliers = [(multiplier-1)*a+1 for a in t_terms]
    sigs = [a*b for (a,b) in zip(sigs, multipliers)]

    return sigs


def rectilinear_interpolation(df):
    '''
    Find the rectilinear interpolated dataframe, assumng the time column is t
    
    Parameters  df: input dataframe
                
    Returns     df_filled: output dataframe filled by rectilinear 
                           interpolation
    '''
    
    df_filled = df.ffill()
    df2 = pd.concat([df_filled['t'].iloc[1:], 
                     df_filled.drop('t', axis=1).shift().iloc[1:]], axis=1)
    df_filled = pd.concat([df2, df_filled], 
                           ignore_index=True).\
    sort_values('t').reset_index(drop=True)
    
    return df_filled


def compute_sigs(df, level=3, fill_method='ffill', basepoint=True):
    
    '''
    Computes the signatures of a path to the required truncation level
    
    Parameters  df: the dataframe to compute signatures from
                level: truncation level of the signature
                fill_method: method to fill missing data
                basepoint: if basepoint is used, then append 0 to the end of 
                           the data
                
    Returns     sigs: an array of the signature values      
    '''

    if fill_method =='rectilinear':
        df_filled = rectilinear_interpolation(df)
    else: 
        df_filled = df.interpolate(method=fill_method)

    if basepoint:
        base_data = pd.DataFrame(np.zeros(len(df_filled.columns))\
                                 .reshape([1, -1]),
                                 columns=df_filled.columns)
        df_filled = pd.concat([base_data, 
                               df_filled.loc[:]]).reset_index(drop=True)

    sigs = esig.stream2sig(np.array(df_filled), level)
    return sigs


def rescale(df_train, df_test):
    '''
    Rescale data by the max absolute values in the train set
    
    Parameters  df_train: dataframe of the training data
                df_test: dataframe of the test data
                
    Returns     scaled df_train and scaled df_test      
    '''
    
    scale = np.max(abs(df_train))
    return 1/scale*df_train, 1/scale*df_test


def compute_subframe(df, l, configs):
    '''
    From the dataframe of the observed data, select a sub-dateframe based on 
    the current index for the rolling/expanding window
    
    Parameters  df: dataframe of the observed data
                l: the current index
                configs: the configurations for the model
                
    Returns     df2: a subset of the input dataframe, across the relevant 
                     horizon
    '''
    
    if configs['max_length'] and configs['max_length']!=np.infty:
        if configs['window_type']=='days':
            start_ind = max([df.index[l-1] \
                             -pd.Timedelta(configs['max_length'],'D'), 
                             df.index[0]])
        elif configs['window_type']=='ind':
            start_ind = df.index[max(0, l-int(configs['max_length']))]     
    else:
        start_ind = df.index[0]

    end_ind = df.index[l-1]

    df2 = df.loc[start_ind:end_ind, :].copy() 
    
    return df2


def subframe_sig_comp(df_sub, df_target, dim, t_level, configs):
    '''
    Compute signatures from a given sub-dataframe, with the necessary configs
    applied
    
    Parameters  df_sub: dataframe of the observed data
                df_target: dataframe of the outcome we want to predict
                dim: dimension of the observation data
                t_level: truncation level for the time terms
                configs: the configurations for the model
                
    Returns     sigs: signature computed on df_sub with given configs
    '''
    
    sigs = compute_sigs(df_sub, level=max([configs['level'], t_level]),
                            fill_method=configs['fill_method'],
                            basepoint=configs['basepoint'])

    if configs['use_multiplier']:
        start_ind = df_sub.index[0]
        multiplier = df_target.loc[start_ind, configs['target']]
        sigs = multiplier_for_t_terms(sigs=sigs, multiplier=multiplier, 
                                      dim=dim, t_level=t_level)
        
    return sigs


def make_sig_df(all_sigs, dim, t_level, configs):
    '''
    Compute signatures from a given sub-dataframe, with the necessary configs
    applied
    
    Parameters  all_sigs: a list of signatures
                dim: int, dimension of the observation data
                t_level: int, truncation level for the time terms
                configs: dictionary of the configurations for the model
                
    Returns     df_sigs: dataframe of signatures with sig keys as column names
    '''
    
    all_sig_names = all_sig_keys(dim, max([configs['level'], t_level]))
    if 'prefix' in configs:
        all_sig_names = [str(configs['prefix'])+n for n in all_sig_names]
            
    df_sigs = pd.DataFrame(all_sigs, columns = all_sig_names)
    
    return df_sigs


def select_signatures(df_sigs, dim, t_level, configs):
    '''
    From the dataframe of signatures, select those relevant to the specified
    config
    
    Parameters  df_sigs: dataframe of all signatures
                dim: dimension of features
                t_level: truncation level of the time (t) terms
                configs: the configurations for the model
                
    Returns     df_sigs: a subset of the input dataframe
    '''
    
    if configs['keep_sigs'] != 'all':
        # filter for sigs linear in the observed values
        ind = compute_linear_sigs(dim, configs['level'])
        df_sigs = df_sigs[df_sigs.columns[ind]]

        ind = find_linear_sig_features(dim=dim, var_level=configs['level'],
                                       t_level=t_level, 
                                       keep_sigs=configs['keep_sigs'])
        df_sigs = df_sigs[df_sigs.columns[ind]]

    else:
        t_terms_index = find_single_sig_index(dim=dim, index=0, level=t_level)
        t_terms_index2 = find_single_sig_index(dim=dim, index=0, 
                                               level=configs['level'])
        ind = np.arange(len(df_sigs.columns))
        if t_level < configs['level']:
            remove_ind = set(t_terms_index2).difference(set(t_terms_index))
            ind = \
            list(set(range(len(df_sigs.columns))).difference(remove_ind))
        elif t_level > configs['level']:
            add_ind = set(t_terms_index).difference(set(t_terms_index2))
            ind = list(set(range(np.sum(dim**np.arange(configs['level']))))\
                           .union(add_ind))
            
        df_sigs = df_sigs[df_sigs.columns[ind]]
            
    return df_sigs


def compute_sigs_dates(df, configs, df_target=None):    
    '''
    Computes path signatures from a timeseries/path. 
    Assumes that time is the zeroth index of the dataframe.
    
    Parameters  df: dataframe of the observation data
                configs: A dictionary of configurations.           
                         This may include the following keys:
                         - max_length: the maximum length of the lookback 
                                       window (expanding window if data is
                                       less than this length)
                         - level: truncation level of the signatures of 
                                  variables not including time
                         - t_level: the level of truncation of the time 
                                    parameter if different to other variables
                         - window_type: if we are choosing window based on 
                                        number of observations (ind) or 
                                        calendar date (days)
                         - keep_sigs: whether to keep only the innermost 
                                      signatures, all linear signatures or all
                                      signatures
                         - prefix: a prefix for the new columns of signature
                                   terms
                         - fill_method: method to fill the dataframe 
                         - basepoint: Boolean whether to add a basepoint to
                                      remove translation invariance of the 
                                      signature
                         - use_multiplier: Boolean whether to multiply the
                                           time terms by the (known) value of
                                           the target at the start 
                                           of the sliding window
                         - target: variable name of target (or similar/lagged
                                   variable used as a multiplication factor to
                                   the signature terms of time only
                                      
                df_target: dataframe of the outcome we want to predict
                           (needed for multiplying the t-terms in the 
                           signature if `use_multipliers' is True in the
                           config)
                
    Returns     df_sigs: a dataframe with the relevant signature features
    
    '''

    all_sigs = []

    if not configs['t_level']:
        t_level = configs['level']
    else:
        t_level = configs['t_level']

    dim = len(df.columns)

    for l in range(1, len(df)+1):
        df2 = compute_subframe(df, l, configs)
        sigs = subframe_sig_comp(df2, df_target, dim, t_level, configs)
        
        all_sigs.append(sigs)

    df_sigs = make_sig_df(all_sigs, dim, t_level, configs)
    df_sigs = select_signatures(df_sigs, dim, t_level, configs)
    
    return df_sigs.set_index(df.index.values)