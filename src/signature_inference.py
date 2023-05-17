import pandas as pd
import signature_helper_funcs as helper_funcs
from pathlib import Path
import pickle

class SignatureInference(object):
    
    def __init__(self, df, target, configs, factor_groups=None):
        '''
        Initialise the class
        
        Parameters  df: DataFrame of indicators and target
                    target: str, name of target variable
                    configs: dict, the configurations
                    factor_groups: DataFrame, with column 'factor_group'
                                   indicting which group a variable belongs to
        
        '''
        
        self.set_data(df)
        self.target = target
        self.target_data = df[target]
        self.configs = configs
        
        if factor_groups:
            self.set_factor_structure(factor_groups)
        else: 
            self.factor_structure = None
    
    
    def set_data(self, df):
        '''
        Set the working data for the inference problem 

        Parameters  df: DataFrame of indicators and target
        '''
        
        self.df = df.copy(deep=True)
        self.set_working_data()
        
        
    def set_working_data(self):
        '''
        Set the working data as a copy of the input data so as to not alter
        the original data.
        '''
        
        self.df_current = self.df.copy(deep=True)
     
        
    def set_factor_structure(self, factor_groups):
        '''
        Define a factor structure based on factor_groups

        Parameters  factor_groups: DataFrame, with column 'factor_group'
                                   indicting which group a variable belongs to
        '''
        
        self.factor_structure = {}
        for group in factor_groups['factor_group'].unique():
            new_group = {str(group): \
                         list(factor_groups[factor_groups\
                                            ['factor_group']==\
                                            group].factor_name)}
            self.factor_structure.update(new_group)


    def update_configs(self, new_configs):
        '''
        Update the configs attribute

        Parameters  new_configs: dict, new configs to be updated to 
                                 self.configs

        '''
        
        self.configs.update(new_configs)

               
    def apply_signature(self):
        '''
        Main class function to apply the signature method
        '''   
        
        target_full = self.target_data.bfill()

        if self.configs['reduce_dim']:
            self.df_current = helper_funcs.reduce_dim\
            (self.df_current.drop(self.target, axis=1),
             self.configs['k'], factor_structure=self.factor_structure, 
             fill_method=self.configs['pca_fill_method'])
            
            self.df_current[self.target] = target_full

        if self.configs['window_type'] == 'days':
            self.df_current['t'] = pd.to_datetime(self.df_current['t'])
            self.df_current.index = self.df_current['t']

        observation_sigs = helper_funcs.find_signatures(self.df_current,
                                                        self.configs)
        
        self.observation_sigs = observation_sigs
        
        if self.configs['use_prev_value']:
            # Obtain shifted data
            observation_sigs = \
            helper_funcs.append_previous_target(self.df_current,
                                                observation_sigs, 
                                                self.configs)

        target_train, observation_train, target_test, observation_test = \
        helper_funcs.split_data(target_full, observation_sigs, self.configs)
        
        # We remove first row as esig sometimes has memory errors
        if not self.configs['basepoint']:
            target_train = target_train[1:]
            observation_train = observation_train[1:]

        results_list = []
        
        if self.configs['save_models']:
            results_list, predictions = helper_funcs.regress(observation_train,
                                                             target_train,
                                                             observation_test,
                                                             target_test,
                                                             self.configs,
                                                             results_list,
                                                             self.configs\
                                                             ['save'])
            self.set_model(self.configs['save'])
            
        else:
            results_list, predictions = helper_funcs.regress(observation_train,
                                                             target_train,
                                                             observation_test,
                                                             target_test,
                                                             self.configs,
                                                             results_list)

        return predictions
    
    
    def set_model(self, foldername, filename='signature_model.pkl'):
        
        '''
        Set a saved model as a class attribute

        Parameters  foldername: str, name of the folder that the model is 
                                saved in
                    filename: str, name of the signature model

        '''
        
        save_dir = Path(__file__).resolve().parent.parent/'results'/foldername
        self.trained_model = pickle.load(open(f'{save_dir}/{filename}', 'rb'))
