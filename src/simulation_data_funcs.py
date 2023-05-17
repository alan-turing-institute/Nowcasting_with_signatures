import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(1234)

def find_inds(data, sampling_method, configs, preserve_start=True):
    '''
    A wrapper function to find the indices to keep, based on the sampling
    method and other configuration parameters
    
    Parameters    data: numpy array of the data to be sampled 
                  sampling_method: string, the method to sample: regular, 
                                   near_regular or random
                  configs: dictionary, contains all of the configurations
                           needed for each type of sampling
                  preserve_start: bool, whether to always keep the starting
                                  value (i.e. index 0)
    
    Returns       inds: list of indices which is kept in the sample
    '''
    
    t = 1    
    keep_inds = []
    for s in sampling_method:
        if s == 'regular':
            ind = regular_sample(freq=configs['freq'][t-1], length=len(data))
        elif s == 'near_regular':
            ind = near_regular_sample(freq=configs['freq'][t-1], 
                                      length=len(data),
                                      perturbation=configs['perturbation']\
                                      [t-1])
        elif s == 'random':
            ind = random_sample(configs['keep_proportion'][t-1], 
                                length=len(data))

        if preserve_start and (0 not in ind):
            ind = np.insert(ind, 0, 0)
        keep_inds.append(np.ravel_multi_index(np.array([ind,
                                                        np.repeat(t,\
                                                                  len(ind))]),
                                              data.shape))
        t += 1

    # Append all time values to keep
    keep_inds.append(np.ravel_multi_index(np.array([np.arange(data.shape[0]),
                                                    np.repeat\
                                                    (0, data.shape[0])]),
                                          data.shape))

    # Break down list of lists so that we have a flat list
    keep_inds = [item for sublist in keep_inds for item in sublist]
    inds = np.array(list(set(range(data.size)).difference(keep_inds)))
    
    return inds


def sample(xt, yt, configs, preserve_start=True, preserve_end=True):
    '''
    This function selects a sub-sample based on the sample rate/downsample 
    frequency
    
    Parameters    xt: the underlying process
                  yt: the noisy observed process
                  configs: a dictionary to hold keyword arguments, which can
                  include
                      sampling_method: one of regular/ near_regular/ random
                                       note that this can be a list/tuple
                                       input for each channel
                      freq: the approximate thinning frequency
                      perturbation: add small pertubation if we are sampling
                                    in a near regular way
                      keep_proportion: proportion of original data to keep if
                                       we are sampling randomly
                  preserve_start: if True, then the first value will always be
                                  kept (for random)
                  preserve_end: if True, then the last value will always be
                                kept
                  
    Returns       xt_sample: downsampled underlying data
                  yt_sample: downsampled observed data
                
    '''
    
    # Note if regular sampling of 30 days would not produce perfect months if 
    # we are indexing by calendar days
    
    # Sampling method can be a string or a list/tuple
    sampling_method = configs['sampling_method']
    
    if isinstance(sampling_method, str):
        if sampling_method == 'regular':
            ind = regular_sample(freq=configs['freq'], length=len(xt))
        elif sampling_method == 'near_regular':
            ind = near_regular_sample(freq=configs['freq'], length=len(xt),
                                      perturbation=configs['perturbation'])
        elif sampling_method == 'random':
            ind = random_sample(configs['keep_proportion'], length=len(xt))
        
        if preserve_start and (0 not in ind):
            ind = np.insert(ind, 0, 0)

        if preserve_end and (len(xt)-1 not in ind):
            ind = np.r_[ind, len(xt)-1]
    
        xt_sample = xt[ind]
        yt_sample = yt[ind]
    
    else:
        inds = find_inds(data=xt, sampling_method=sampling_method,
                         configs=configs, preserve_start=preserve_start)
        
        # Apply the indices mask and set values to nan
        # Alternatively can convert to Boolean mask
        np.put(xt, inds, np.nan)
        
        # check if separate config for y specified
        if 'configs_y' in configs.keys():
            print('separately sampling for observation points')
            inds_y = find_inds(data=yt, sampling_method=configs['configs_y']\
                               ['sampling_method'], 
                               configs=configs['configs_y'],
                               preserve_start=preserve_start)
            np.put(yt, inds_y, np.nan) 
        else:
            np.put(yt, inds, np.nan)

        # Note the original xt and yt have been altered (sampling occured) as
        # np.put is inplace 
        xt_sample = xt     
        yt_sample = yt

    return xt_sample, yt_sample


def random_sample(keep_proportion, length):
    '''
    A random sample of the indices based on the proportion of data we want to
    keep
    
    Parameters    keep_proportion: float, the fraction of data we want the
                                   sample to keep
                  length: integer, total length of series to sample
    
    Returns       ind: list, indices chosen in this sample
    
    '''
    
    ind = np.random.choice(length, int(np.floor(length*keep_proportion)),
                           replace=False)
    ind = np.sort(ind)
    return ind


def regular_sample(freq, length):
    '''
    A regular sample of the indices based on the proportion of data we want to
    keep
    
    Parameters    freq: float or int, frequency by which to sample
                  length: int, total length of series to sample
    
    Returns       ind: list, indices to keep according to the sampling 
                       frequency
    '''
    
    ind = np.arange(length)[::int(np.floor(freq))]
    return ind


def near_regular_sample(freq, length, perturbation=5):
    '''
    A regular sample of the indices based on the proportion of data we want to
    keep, perturbed slightly
    
    Parameters    freq: float or int, frequency that this sample approximates
                  length: int, total length of series to sample
                  perturbation: int, the maximum amount of perturbation
                                to make to each index
    
    Returns       ind: list, indices to keep according to the sampling 
                       frequency
    '''
    
    ind = np.arange(length)[::int(np.floor(freq))]
    ind += np.random.randint(-perturbation, perturbation, size=len(ind))
    ind = np.clip(ind, 0, length-1)
    ind = np.unique(ind)
    return ind


def save_df(data, time_index='integer', save_dir='./data', prefix='x',
            dropna=True, suffix=''):
    '''
    Save the data as a csv, changing the time to be dates if needed
    
    Parameters  data: the data to be saved
                time_index: whether the data should be indexed by int or
                            datetime
                save_dir: directory and name of the output file
                prefix: the prefix to the column names
                dropna: Boolean specifying whether empty rows will be droped
    '''
    _, dim = data.shape
    
    # Create column names
    cols = ['t']
    for i in range(1, dim):
        cols.append(f'{prefix}{i}')

    df = pd.DataFrame(data, columns=cols)
    
    # Change to datetime if desired with offset defined by the original column
    if time_index == 'dates':
        df['t'] = pd.to_datetime(df['t'], unit='D', 
                                 origin=pd.Timestamp('2000-01-01'))

    df.set_index('t', inplace=True)

    if dropna:
        df.dropna(how='all', inplace=True)
    
    df.to_csv(f'{save_dir}{suffix}.csv')

    
def generate_data_example(x0, var1=0.01, var2=0.01, length=1000, 
                          transition=None, dt=0.01, H=1):
    '''
    Generate random data 
    
    Parameters   x0 : the initial value of the system
                 var1: the variance of the additive noise in the system
                 var2: the variance of the measurement error
                 length: the length of the series to simulate
                 transition: the transition matrix
                 dt: time step size
                 H: parameter in model
    
    Returns      xt: the underlying process
                 yt: the noisy observed process
    '''

    if length == 'random':
        end_time = 0.1+0.9*np.random.rand()
        length = np.floor(end_time/dt)

    if type(length) != int:
        length = int(length)
        
    if transition is None:
        transition = -1.0
    
    xt = []
    xt.append(x0)
    
    for it in range(1, length):
        xt.append(np.array(xt[it-1]+transition*xt[it-1]*dt + \
                           (var1*dt)**0.5*np.random.randn()))

    noise_y = (var2*dt)**0.5*np.random.randn(length-1)
    noise_y = np.cumsum(noise_y)

    yt = np.insert(H*dt*np.cumsum(xt[1:])+noise_y, 0, 0)
    yt = yt + xt[0]
    yt = np.reshape(yt, (-1,1))
    
    t = np.expand_dims(np.arange(length), axis=0).T*dt

    xt = np.array(xt)
    xt = np.concatenate([t, xt], axis=1)
    yt = np.concatenate([t, yt], axis=1)
    
    return xt, yt

def transform_data(input_dir, configs):
    '''
    Transform existing data by specified transformation and save new data
    
    Parameters    input_dir: Path, the directory to read from (which is also 
                             where the transformed data is saved to)
                  configs: A dictionary of configs with the following keys
                           - num_samples: int, the number of total 
                                          trajectories
                           - data_transform: string, the transformation for 
                                             the data accepts sigmoid/log/exp
    '''

    def sigm(x):
        return 1/(1 + np.exp(-x))
    
    for index in range(configs['num_samples']):
        df = pd.read_csv(input_dir/f'yt_sample{index}.csv')
        df_x = pd.read_csv(input_dir/f'xt_sample{index}.csv')

        if configs['data_transform'] == 'sigmoid':
            df['y1'] = sigm(df['y1'])
            
        elif configs['data_transform'] == 'exp':
            df['y1'] = np.exp(df['y1'])

        elif configs['data_transform'] == 'log':
            df['y1'] = np.log(df['y1'])

        Path(input_dir/configs['data_transform']).mkdir(parents=True,
                                                        exist_ok=True)
        
        df.to_csv(input_dir/configs['data_transform']/f'yt_sample{index}.csv',
                  index=False)
        df_x.to_csv(input_dir/configs['data_transform']\
                    /f'xt_sample{index}.csv', index=False)
