import sys
import numpy as np
import pandas as pd
from pathlib import Path
import yaml
import warnings

import simulation_data_funcs as d_funcs
import simulation_analysis_funcs as s_funcs

np.random.seed(1234)


def define_analysis_folders(configs):
    '''
    Define input and output folders based on the config settings
    
    Parameters    configs: a dictionary to hold keyword arguments, which can
                  include
                      data_folder: str, folder name of the experiment
                      downsample: bool, whether to downsample
                      nonlinear_transform: bool, whether to transform the data
                      data_transform: str, the type of nonlinear transformation
    
    Returns       input_dir: Path, where the data is stored and read from
                  output_dir: Path, where the results is saved to
    '''

    if configs['nonlinear_transform'] and configs['downsample']:
        input_dir = Path(__file__).resolve().parent.parent/'simulation_data'\
        /configs['data_folder']/'downsample'/configs['data_transform']
        output_dir = Path(__file__).resolve().parent.parent/'results'\
        /configs['save_folder']/'downsample'/configs['data_transform'] 
       
    elif configs['nonlinear_transform']:
        input_dir = Path(__file__).resolve().parent.parent/'simulation_data'\
        /configs['data_folder']/configs['data_transform']
        output_dir = Path(__file__).resolve().parent.parent/'results'\
        /configs['save_folder']/configs['data_transform']            
    elif configs['downsample']:
        input_dir = Path(__file__).resolve().parent.parent/'simulation_data'\
        /configs['data_folder']/'downsample'
        output_dir = Path(__file__).resolve().parent.parent/'results'\
        /configs['save_folder']/'downsample'    
    else:
        input_dir = Path(__file__).resolve().parent.parent/'simulation_data'\
        /configs['data_folder']
        output_dir = Path(__file__).resolve().parent.parent/'results'\
        /configs['save_folder']
    if output_dir.exists():
        warnings.warn('Results folder already exists, results may be '
                      'overwritten', stacklevel=2)

    return input_dir, output_dir


def generate_data(configs):
    '''
    Generates simulated data
    
    Parameters    configs: a dictionary to hold keyword arguments, which can
                  include
                      save_folder: str, name of folder to save to
                      num_samples: int, number of paths to simulate
                      x0: float, the inital value of the hidden variable
                      length: int, length of series to simulate 
                      var1: the variance of the additive noise in the system
                      var2: the variance of the measurement error
                      dt: float, time step size
                      H: float, parameter
                      downsample: bool, whether to downsample the full path
                      time_index: str, how to index the data (integer/dates)
    
    '''
    
    output_dir = \
    Path(__file__).resolve().parent.parent/'simulation_data'\
    /configs['save_folder']
    if output_dir.exists():
        warnings.warn('Data folder already exists, '\
                      'results may be overwritten', stacklevel=2)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f'Saving data to simulation_data {configs["save_folder"]}')

    for i in range(configs['num_samples']):

        if isinstance(configs['x0'], str):
            x0 = np.array([np.random.randn()])
        else:
            x0 = np.array([configs['x0']])

        xt, yt = d_funcs.generate_data_example(x0=x0,
                                               length=configs['length'],
                                               var1=configs['var1'],
                                               var2=configs['var2'],
                                               dt=configs['dt'], 
                                               H=configs['H'])

        if configs['downsample']:
            xt_sample, yt_sample = d_funcs.sample(xt, yt, configs)
        else:
            xt_sample, yt_sample = xt, yt

        d_funcs.save_df(xt_sample, time_index=configs['time_index'],
                        save_dir=f'{output_dir}/xt_sample', prefix='x',
                        suffix=i)
        d_funcs.save_df(yt_sample, time_index=configs['time_index'],
                        save_dir=f'{output_dir}/yt_sample', prefix='y',
                        suffix=i)

        
def main(configs):
    
    '''
    Main function for one experiment. This takes in the (pre-)generated data,
    performs any downsampling and data transformation required, then the
    analysis. All results and plots are saved to the folder specified within
    the configs.
    
    Parameters    configs: a dictionary to hold keyword arguments for this
                           experiment with simulated data
    '''
    
    if configs['downsample']:
        save_folder = Path(__file__).resolve().parent.parent/\
        'simulation_data'/configs['save_folder']
        Path(save_folder/'downsample').mkdir(parents=True, exist_ok=True)

        for i in range(configs['num_samples']):

            xt = pd.read_csv(save_folder/f'xt_sample{str(i)}.csv').to_numpy()
            yt = pd.read_csv(save_folder/f'yt_sample{str(i)}.csv').to_numpy()

            xt_sample, yt_sample = d_funcs.sample(xt, yt, configs)
                
            d_funcs.save_df(xt_sample, time_index=configs['time_index'], 
                    save_dir=f'{save_folder}/downsample/xt_sample',
                    prefix='x', suffix=i)
            d_funcs.save_df(yt_sample, time_index=configs['time_index'],
                    save_dir=f'{save_folder}/downsample/yt_sample',
                    prefix='y', suffix=i)
    

    if configs['nonlinear_transform']:
        print(f"Applying {configs['data_transform']} transformation to data")
        if configs['downsample']:  
            output_dir = \
            Path(__file__).resolve().parent.parent/'simulation_data'\
            /configs['save_folder']/'downsample'
        else:
            output_dir = \
            Path(__file__).resolve().parent.parent/'simulation_data'\
            /configs['save_folder']
            
        d_funcs.transform_data(input_dir=output_dir, configs=configs)

    if configs['analyse']:
        
        input_dir, output_dir = define_analysis_folders(configs)

        Path(output_dir).mkdir(parents=True, exist_ok=True)

        #print(f'Data folder is {input_dir}')
        #print(f'Results folder is {output_dir}')

        if configs['kf']:
            print('KF analysis')
            s_funcs.filter_results(input_dir, output_dir, configs)

        if configs['sig']:
            print('signature analysis')
            s_funcs.analysis(input_dir,
                             configs=configs, 
                             save_dir=output_dir)

            s_funcs.compare_methods(output_dir, configs['train_proportion'], 
                            save=output_dir)

        if configs['plot_example']:
            
            if configs['plot_ind']:
                plot_ind = configs['plot_ind']
            else:   
                plot_ind = \
                int(configs['train_proportion']*configs['num_samples']+1)
            
            s_funcs.filter_example(input_dir, plot_ind, output_dir, configs)
            s_funcs.plot_example(input_dir, plot_ind, output_dir, configs)
    
        # save config with data for future reference
        with open(output_dir/'configs_used.yaml', 'w') as file:
            yaml.dump(configs, file)


def generate_all_results(configs):
    '''
    A wrapper function that changes the configs as required and runs the 
    analysis multiple times.
    
    Parameters    configs:  a dictionary to hold keyword arguments for this
                           experiment with simulated data
    '''
    
    main(configs)
    configs.update({'downsample': True})
    main(configs)
    configs.update({'nonlinear_transform': True})
    configs.update({'keep_sigs': 'all'})
    configs.update({'level': 3})
    configs.update({'t_level': 3})
    main(configs)
    configs.update({'downsample': False})
    main(configs)


def reduce_level(yaml_name):
    
    '''
    Reduce the level of the parameters so that the model can be used
    in comparing with true parameters
    
    Parameters    yaml_name: str, name of the config yaml file
    '''

    with open(f'../src/{yaml_name}', 'r') as stream:
        configs = yaml.safe_load(stream)

    configs['level'] = 3
    configs['t_level'] = 1
    configs['save_folder'] = 'simulation_reduce'

    main(configs)
    
    
if __name__ == '__main__':

    # check for custom config
    if len(sys.argv) > 1:
        yaml_name = sys.argv[1]
    else:
        yaml_name = 'simulation_example_config.yaml'
    
    with open(Path(__file__).resolve().parent/yaml_name, 'r') as stream:
        configs = yaml.safe_load(stream)

    if configs['gen_data']:    
        generate_data(configs)
        
    generate_all_results(configs)
    
    if configs['compare_coeffs']: 
        print('Rerun with lower truncation levels')
        reduce_level(yaml_name)


