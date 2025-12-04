"""
author: Oliver Löthgren

This file contains all configs to run the experiments from the first paper
"""
import numpy as np

from configs.config_utils import get_parameter_array, get_dataset_overview, \
    makedirs, data_path, training_data_path



# ==============================================================================
# -------------------- ONJODE 1 - Dataset Dicts ---------------------------------
# ==============================================================================

# ------------------------------------------------------------------------------
# default hp dict for generating BrownianCosine, BrownianExpDecay (1d in space, 1d in output)

# standard: nb_paths, nb_time_steps, nb_space_steps, X0, maturity, space_limits, space_dimension, dimension
# extras: J_obs_perc, obs_perc
# customize: masked, J_distribution, space_distribution, obs_scheme 
hyperparam_1d1d_default = {
    'nb_paths': 20000, 'nb_time_steps': 100,
    'nb_space_steps': 100,
    'maturity': 1., 'obs_perc': 0.1,
    'space_limits': (-5, 15), 'space_dimension': 1,
    'dimension': 1,
    'J_obs_perc': 0.6
}


# ------------------------------------------------------------------------------
# default hp dict for BrownianCosSine, ... (1d in space, 2d in output)
hyperparam_1d2d_default = {
    'nb_paths': 20000, 'nb_time_steps': 100,
    'nb_space_steps': 100,
    'maturity': 1., 'obs_perc': 0.1,
    'space_limits': (-10, 10), 'space_dimension': 1,
    'dimension': 2,
    'J_obs_perc': 0.4
}


# ------------------------------------------------------------------------------
# default hp dict for BrownianCosSum, ...  (2d in space, 1d in output)
hyperparanm_2d1d_default = {
    'nb_paths': 20000, 'nb_time_steps': 100,
    'nb_space_steps': 100,
    'maturity': 1., 'obs_perc': 0.1,
    'space_limits': [(-10, 10), [-5, 5]], 'space_dimension': 2,
    'dimension': 1,
    'J_obs_perc': 0.4
}

#add a 2d2d default hyperparameter dict if needed in the future

# ==============================================================================
# -------------------- ONJODE - Training Dicts --------------------------------
# ==============================================================================

ode_nn = ((50, 'relu'), (50, 'relu'))
readout_nn = ((50, 'tanh'), (50, 'tanh'))#, (32, 'lrelu'))
enc_nn = ((50, 'relu'), (50, 'relu'))#, (32, 'lrelu'))
genk_nn = ((32, 'relu'), (32, 'relu'))#, (16, 'lrelu')) # placeholder for now

# ------------------------------------------------------------------------------
# --- BrownianCosine, BrownianExpDecay
error_dist_dict_1d1d = {
    'eval_times': ["mid", "end"], # times at which to evaluate the error distribution
    'model_names': None,
}

brownian_linear_dict = {
    'model_name': "BrownianLinear",
    'nb_paths': 10000, 'nb_time_steps': 50, 
    'nb_space_steps': 50,
    'maturity': 1., 'obs_perc': 0.1,
    'space_limits': (1, 5), 'space_dimension': 1,
    'dimension': 1,
    'J_obs_perc': 0.4
    } 

brownian_cos_dict = {
    'model_name': "BrownianCosine",
    'nb_paths': 20000, 'nb_time_steps': 50, 
    'nb_space_steps': 50,
    'maturity': 1., 'obs_perc': 0.1,
    'space_limits': (-5, 5), 'space_dimension': 1,
    'dimension': 1,
    'J_obs_perc': 0.1
    } 

brownian_expdec_dict = {
    'model_name': "BrownianExpDecay",
    'alpha': 1,
    'nb_paths': 20000, 'nb_time_steps': 50,
    'nb_space_steps': 50,
    'maturity': 1., 'obs_perc': 0.1,
    'space_limits': (-5, 5), 'space_dimension': 1,
    'dimension': 1,
    'J_obs_perc': 0.1
}


param_dict_1d1d = {
    # training parameters
    'epochs': [200], 
    'batch_size': [200],
    'save_every': [5],
    'learning_rate': [0.001],
    'test_size': [0.2],
    'seed': [398],
    'bias': [True],
    'solver': ["euler"],  
    # networks
    'hidden_size': [100], 
    'context_size': [100], 
    'space_dimension': [1], 
    'dim': [1], 
    'genk_nn': [genk_nn], 
    'ode_nn': [ode_nn], 'readout_nn': [readout_nn], 'enc_nn': [enc_nn],
    'use_rnn': [True], 
    # dataset
    'dataset': ["BrownianCosine"], #, "BrownianExpDecay"], For parallel runs, use different datasets in the same list
    #'dataset_id': [None]
    'data_dict': [None], 
    # extras
    'weight': [0.5],
    'weight_decay': [1.],
    'plot': [True], 
    'evaluate': [True],
    'paths_to_plot': [(0,1,2,3)],
    #'residual_ode': [True], # optionally residual networks
    'gradient_clip': [50.], # gradient clipping by norm
    'input_scaling_func': ["tanh"], # input scaling key for Neural ODE
    'layer_norm': [True],
    'enc_input_norm': [False],
    #"clamp": [10], 
    #'plot_error_dist': error_dist_dict_1d1d, # to plot errors through space grid
}

param_dict_1d1d = get_parameter_array(param_dict=param_dict_1d1d)


# ------------------------------------------------------------------------------
# --- BrownianCosSum and BrownianCosSine may need separate dicts for different dimensions in space/output -- or the same works

