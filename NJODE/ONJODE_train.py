"""
author: Oliver Löthgren

implementation of the training (and evaluation) of O-NJ-ODE

NOTE: pandas version updated from 1.0.4 --> 1.3.5 due to empty compatibility issues regarding empty dataframes
"""

# =====================================================================================================================
from typing import List

import torch  # machine learning
import torch.nn as nn
import tqdm  # process bar for iterations
import numpy as np  # large arrays and matrices, functions
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import os, sys
import pandas as pd  # data analysis and manipulation
import json  # storing and exchanging data
import time
import socket
import matplotlib  # plots
import matplotlib.colors
from torch.backends import cudnn
import gc
import scipy.stats as stats
import warnings

from configs import config
import models
import data_utils
sys.path.append("../")
import baselines.GRU_ODE_Bayes.models_gru_ode_bayes as models_gru_ode_bayes

try:
    from telegram_notifications import send_bot_message as SBM
except Exception:
    from configs.config import SendBotMessage as SBM


# =====================================================================================================================
# check whether running on computer or server
if 'ada-' not in socket.gethostname():
    SERVER = False
    N_CPUS = 1
    SEND = False
else:
    SERVER = True
    N_CPUS = 1
    SEND = True
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
print(socket.gethostname())
print('SERVER={}'.format(SERVER))


# ==============================================================================
# Global variables
CHAT_ID = config.CHAT_ID
ERROR_CHAT_ID = config.ERROR_CHAT_ID

data_path = config.data_path
saved_models_path = config.saved_models_path
flagfile = config.flagfile

METR_COLUMNS: List[str] = [
    'epoch', 'train_time', 'val_time', 'train_loss', 'val_loss',
    'optimal_val_loss', 'test_loss', 'optimal_test_loss', 'evaluation_mean_diff']
default_ode_nn = ((50, 'tanh'), (50, 'tanh'))
default_readout_nn = ((50, 'tanh'), (50, 'tanh'))
default_enc_nn = ((50, 'tanh'), (50, 'tanh'))
default_genk_nn = ((50, 'tanh'), (50, 'tanh'))  # ADDED may need modification

ANOMALY_DETECTION = False 
N_DATASET_WORKERS = 0
USE_GPU = False 

# =====================================================================================================================
# Functions
makedirs = config.makedirs

def update_metric_df_to_new_version(df, path):
    """
    update the metric file to the new version, using the updated column names
    """
    if 'val_loss' not in df.columns:
        df = df.rename(columns={
            'eval_time': 'val_time', 'eval_loss': 'val_loss',
            'optimal_eval_loss': 'optimal_val_loss'})
        df.to_csv(path)
    return df


def train(
        anomaly_detection=None, n_dataset_workers=None, use_gpu=None,
        nb_cpus=None, send=None, gpu_num=0,
        model_id=None, epochs=100, batch_size=100, save_every=1, 
        learning_rate=0.001, test_size=0.2, seed=398,
        hidden_size=10, context_size=10,  
        bias=True, dropout_rate=0.,
        ode_nn=default_ode_nn, readout_nn=default_readout_nn,
        enc_nn=default_enc_nn, genk_nn=default_genk_nn, 
        solver="euler", weight=0.5, weight_decay=1.,
        dataset='BrownianCosine', dataset_id=None, data_dict=None,
        plot=True, paths_to_plot=(0,),
        saved_models_path=saved_models_path,
        DEBUG=0,
        **options
):
    print(torch.cuda.is_available())
    print(use_gpu)
    print(torch.version.cuda)
    print(torch.cuda.get_device_name(0))
    
    """
    training function for ONJODE model (models.ONJODE),
    the model is automatically saved in the model-save-path with the given
    model id, also all evaluations of the model are saved there

    :param anomaly_detection: used to pass on FLAG from parallel_train
    :param n_dataset_workers: used to pass on FLAG from parallel_train
    :param use_gpu: used to pass on FLAG from parallel_train
    :param nb_cpus: used to pass on FLAG from parallel_train
    :param send: used to pass on FLAG from parallel_train
    :param model_id: None or int, the id to save (or load if it already exists)
            the model, if None: next biggest unused id will be used
    :param epochs: int, number of total epochs to train, each epoch is one cycle
            through all (random) batches of the training data
    :param batch_size: int
    :param save_every: int, defined number of epochs after each of which the
            model is saved and plotted if wanted. whenever the model has a new
            best eval-loss it is also saved, independent of this number (but not
            plotted)
    :param learning_rate: float
    :param test_size: float in (0,1), the percentage of samples to use for the
            test set (here there exists only a test set, since there is no extra
            evaluation)
    :param seed: int, seed for the random splitting of the dataset into train
            and test
    :param hidden_size: see models.ONJODE
    :param context_size: see models.ONJODE
    :param bias: see models.ONJODE
    :param dropout_rate: float
    :param ode_nn: see models.ONJODE
    :param readout_nn: see models.ONJODE
    :param genk_nn: see models.ONJODE
    :param enc_nn: see models.ONJODE
    :param solver: see models.ONJODE
    :param weight: see models.ONJODE
    :param weight_decay: see models.ONJODE
    :param dataset: str, which dataset to use, supported: {'BrownianCosine',
    'BrownianExpDecay', 'BrownianCosSine'}. The corresponding dataset already
            needs to exist (create it first using data_utils.create_dataset)
    :param dataset_id: int or None, the id of the dataset to be used, if None,
            the latest generated dataset of the given name will be used
    :param data_dict: None, str or dict, if not None, the inputs dataset and
            dataset_id are overwritten. if str, the dict from config.py with the
            given name is loaded.
            from the dataset_overview.csv file, the
            dataset with a matching description is loaded.
    :param plot: bool, whether to plot
    :param paths_to_plot: list of ints, which paths of the test-set should be
            plotted. Default to first path only
    :param saved_models_path: str, where to save the models
    :param DEBUG: int, if >0, then the model is in debug mode
    :param options: kwargs, used keywords:
        'test_data_dict'    None, str or dict, if no None, this data_dict is
                        used to define the dataset for plot_only and
                        evaluation (if evaluate=True)
        'masked'        bool, whether the data is masked (i.e. has                      USED
                        incomplete observations)
        'save_extras'   bool, dict of options for saving the plots
                        when plotting the variance
        'parallel'      bool, used by parallel_train.parallel_training
        'resume_training'   bool, used by parallel_train.parallel_training
        'plot_only'     bool, whether the model is used only to plot after
                        initiating or loading (i.e. no training) and exit
                        afterwards (used by demo)
        'ylabels'       list of str, see plot_one_path_with_pred()
        'legendlabels'  list of str, see plot_one_path_with_pred()
        'plot_same_yaxis'   bool, whether to plot the same range on y axis
                        for all dimensions
        'plot_obs_prob' bool, whether to plot the observation probability
        'residual_enc_dec'  bool, whether resNNs are used for encoder and
                        readout NN, used by models.NJODE. the provided value
                        is overwritten by 'residual_enc' & 'residual_dec' if
                        they are provided. default: False
        'residual_enc'  bool, whether resNNs are used for encoder NN,
                        used by models.NJODE.
                        default: False (this is
                        for backward compatibility)
        'residual_dec'  bool, whether resNNs are used for readout NN,
                        used by models.NJODE. default: True
        'use_y_for_ode' bool, whether to use y (after jump) or x_impute for
                        the ODE as input, only in masked case, default: True
        'use_current_y_for_ode' bool, whether to use the current y as input
                        to the ode. this should make the training more
                        stable in the case of long windows without
                        observations (cf. literature about stable output
                        feedback). default: False
        'coord_wise_tau'    bool, whether to use a coordinate wise tau
        'input_current_t'   bool, whether to additionally input current time
                        to the ODE function f, default: False
        'enc_input_t'   bool, whether to use the time as input for the
                        encoder network. default: False
        'train_readout_only'    bool, whether to only train the readout
                        network
        'training_size' int, if given and smaller than
                        dataset_size*(1-test_size), then this is the number
                        of samples used for the training set (randomly
                        selected out of original training set)
        'evaluate'      bool, whether to evaluate the model in the test set
                        (i.e. not only compute the val_loss, but also
                        compute the mean difference between the true and the
                        predicted paths comparing at each time point)
        'load_best'     bool, whether to load the best checkpoint instead of
                        the last checkpoint when loading the model. Mainly
                        used for evaluating model at the best checkpoint.
        'gradient_clip' float, if provided, then gradient values are clipped
                        by the given value
        'clamp'         float, if provided, then output of model is clamped
                        to +/- the given value
        'use_observation_as_input'  bool, whether to use the observations as
                        input to the model or whether to only use them for
                        the loss function (this can be used to make the
                        model learn to predict well far into the future).
                        can be a float in (0,1) to use an observation with
                        this probability as input. can also be a string
                        defining a function (when evaluated) that takes the
                        current epoch as input and returns a bool whether to
                        use the current observation as input (this can be a
                        random function, i.e. the output can depend on
                        sampling a random variable). default: true
        'val_use_observation_as_input'  bool, None, float or str, same as
                        'use_observation_as_input', but for the validation
                        set. default: None, i.e. same as for training set
        'ode_input_scaling_func'    None or str in {'id', 'tanh'}, the
                        function used to scale inputs to the neuralODE.
                        default: tanh
        'use_cond_exp'  bool, whether to use the conditional expectation
                        as reference for model evaluation, default: True
        'eval_use_true_paths'   bool, whether to use the true paths for
                        evaluation (instead of the conditional expectation)
                        default: False
        'input_coords'  list of int or None, which coordinates to use as                             USED
                        input. overwrites the setting from dataset_metadata.
                        if None, then all coordinates are used.
        'output_coords' list of int or None, which coordinates to use as                             USED
                        output. overwrites the setting from
                        dataset_metadata. if None, then all coordinates are
                        used.
        'plot_only_evaluate' bool, whether to evaluate the model when in
                        plot_only mode
        'plot_error_dist'   None or dict, if not None, then the kwargs for
                        the plot_error_distribution function

            -> Following extra options from the ONJODE framework                         NOT
                names 'ONJODE'+<option_name>, for the following list
                of possible choices for <options_name>:
                '-TEXT'   type, default: TEXT
    """

    global ANOMALY_DETECTION, USE_GPU, SEND, N_CPUS, N_DATASET_WORKERS
    if anomaly_detection is not None:
        ANOMALY_DETECTION = anomaly_detection
    if use_gpu is not None:
        USE_GPU = use_gpu
    if send is not None:
        SEND = send
    if nb_cpus is not None:
        N_CPUS = nb_cpus
    if n_dataset_workers is not None:
        N_DATASET_WORKERS = n_dataset_workers

    initial_print = "model-id: {}\n".format(model_id)

    use_cond_exp = True # whether to use the conditional expectation as reference for model evaluation
    if 'use_cond_exp' in options:
        use_cond_exp = options['use_cond_exp']
    eval_use_true_paths = False
    if 'eval_use_true_paths' in options: # whether to use true paths for evaluation or the conditional expectation (defaulted to use conditional expectation: false)
        eval_use_true_paths = options['eval_use_true_paths']

    masked = False # whether the output data dimension is masked (i.e. has incomplete observations)
    if 'masked' in options: 
        masked = options['masked']

    if ANOMALY_DETECTION:
        # allow backward pass to print the traceback of the forward operation
        #   if it fails, "nan" backward computation produces error
        torch.autograd.set_detect_anomaly(True)
        torch.manual_seed(0)
        np.random.seed(0)
        # set seed and deterministic to make reproducible
        cudnn.deterministic = True

    # set number of CPUs
    torch.set_num_threads(N_CPUS)

    # get the device for torch
    if USE_GPU and torch.cuda.is_available():
        device = torch.device("cuda:{}".format(gpu_num))
        torch.cuda.set_device(gpu_num)
        initial_print += '\nusing GPU'
    else:
        device = torch.device("cpu")
        initial_print += '\nusing CPU'

    # load dataset-metadata --> this includes all the specs and parameters of the dataset
    if data_dict is not None:
        dataset, dataset_id = data_utils._get_dataset_name_id_from_dict(
            data_dict=data_dict)
        dataset_id = int(dataset_id)
    else:
        if dataset is None:
            dataset = data_utils._get_datasetname(time_id=dataset_id)
        dataset_id = int(data_utils._get_time_id(stock_model_name=dataset,
                                                 time_id=dataset_id))
    dataset_metadata = data_utils.load_metadata(stock_model_name=dataset,
                                                time_id=dataset_id)

    # get input and output coordinates of the dataset
    input_coords = None
    output_coords = None
    if "input_coords" in options:
        input_coords = options["input_coords"]
    elif "input_coords" in dataset_metadata:
        input_coords = dataset_metadata["input_coords"]
    if "output_coords" in options:
        output_coords = options["output_coords"]
    elif "output_coords" in dataset_metadata:
        output_coords = dataset_metadata["output_coords"]
    if input_coords is None:
        input_size = dataset_metadata['dimension']                              # input_size is the number of coordinates of X used as input
        input_coords = np.arange(input_size)                                    # input_chords is from 0->dim-1
    else:
        input_size = len(input_coords)
    if output_coords is None:
        output_size = dataset_metadata['dimension']                           # output_size is similar (of the conditional expectation)
        output_coords = np.arange(output_size)
    else:
        output_size = len(output_coords)
    
    initial_print += '\ninput_coords: {}\noutput_coords: {}'.format(
        input_coords, output_coords)          # prints which coordinates from the dataset are used as input and output (overwrites metadata)
    initial_print += '\ninput_size: {}\noutput_size: {}'.format(
        input_size, output_size)

    # gather metadata
    dimension = dataset_metadata['dimension']
    T = dataset_metadata['maturity']
    delta_t = dataset_metadata['dt']  
    space_dimension = dataset_metadata['space_dimension'] 

    # load raw data into train and split by path index
    train_idx, val_idx = train_test_split(
        np.arange(dataset_metadata["nb_paths"]), test_size=test_size,
        random_state=seed) # test_size = percentage of samples to use in training set
    if 'training_size' in options:
        train_set_size = options['training_size']
        if train_set_size < len(train_idx):
            train_idx = np.random.choice(
                train_idx, train_set_size, replace=False
            )
    # define the training, validation, and test datasets
    data_train = data_utils.OperatorDataset(
        model_name=dataset, time_id=dataset_id, idx=train_idx) 
    data_val = data_utils.OperatorDataset(
        model_name=dataset, time_id=dataset_id, idx=val_idx) 
    test_data_dict = None
    if 'test_data_dict' in options:
        test_data_dict = options['test_data_dict'] 
    if test_data_dict is not None:
        test_ds, test_ds_id = data_utils._get_dataset_name_id_from_dict(
            data_dict=test_data_dict)
        test_ds_id = int(test_ds_id)
        data_test = data_utils.OperatorDataset(
            model_name=test_ds, time_id=test_ds_id, idx=None) 

    functions = None  # no additional functions applied to X
    collate_fn, mult = data_utils.OperatorCollateFnGen(None) 
    mult = 1

    # get data-loader for training
    # train data loader
    dl = DataLoader(  # class to iterate over training data
        dataset=data_train, collate_fn=collate_fn,
        shuffle=True, batch_size=batch_size, num_workers=N_DATASET_WORKERS)
    # val data loader
    dl_val = DataLoader( # class to iterate over validation data
        dataset=data_val, collate_fn=collate_fn,
        shuffle=False, batch_size=int(len(data_val)/10), num_workers=N_DATASET_WORKERS)
    stockmodel = data_utils._OPERATOR_MODELS[
        dataset_metadata['model_name']](**dataset_metadata)                       
    if test_data_dict is not None: # including a test dataset (on top of train and val) generated from the data_dict
        dl_test = DataLoader(  # class to iterate over test data
            dataset=data_test, collate_fn=collate_fn,
            shuffle=False, batch_size=int(len(data_test)/10),
            num_workers=N_DATASET_WORKERS)
        testset_metadata = data_utils.load_metadata(
            stock_model_name=test_ds, time_id=test_ds_id)
        stockmodel_test = data_utils._OPERATOR_MODELS[
            testset_metadata['model_name']](**testset_metadata)
    else:
        dl_test = dl_val # No test data --> stick with train, val & same model/params
        stockmodel_test = stockmodel
        testset_metadata = dataset_metadata
    
    ylabels = None
    if 'ylabels' in options:
        ylabels = options['ylabels']
    legendlabels = None
    if 'legendlabels' in options:
        legendlabels = options['legendlabels']
    plot_same_yaxis = False
    if 'plot_same_yaxis' in options:
        plot_same_yaxis = options['plot_same_yaxis']
    plot_obs_prob = False
    if 'plot_obs_prob' in options:
        plot_obs_prob = options["plot_obs_prob"]

    # validation loss function
    which_val_loss = 'operator'                                 # defaults to the io loss (operator loss function under consideration)
    
    print(dataset_metadata)
    plot_only = False
    if 'plot_only' in options:
        plot_only = options['plot_only']
    # // Calculates the optimal loss (conditional expectation loss) for validation set
    if use_cond_exp and not plot_only:
        store_cond_exp = True
        if dl_val != dl_test:                  # when there is a separate test set, then do not store the loss, path_t, or path_X as self.objects in the stockmodel class
            store_cond_exp = False
        
        opt_val_loss = compute_optimal_val_loss(
            dl_val, stockmodel, delta_t, T, mult=mult,
            store_cond_exp=store_cond_exp, 
            which_loss=which_val_loss) # returns loss, path_t, path_X
        initial_print += '\noptimal {}val-loss (achieved by true cond exp): ' \
                     '{:.5f}'.format("", opt_val_loss)               
    else:
        opt_val_loss = np.nan
    
    # get params_dict
    params_dict = {  # create a dictionary of the model parameters 
        'input_size':input_size, 'output_size': output_size, 'epochs': epochs, 'batch_size': batch_size,
        'hidden_size': hidden_size, 'context_size': context_size, 
        'space_dimension': space_dimension, 'dim': dimension, 'bias': bias,
        'ode_nn': ode_nn, 'readout_nn': readout_nn, 'enc_nn': enc_nn,
        'genk_nn': genk_nn, 
        'dropout_rate': dropout_rate,
        'solver': solver, 'dataset': dataset, 'dataset_id': dataset_id,
        'data_dict': data_dict,
        'learning_rate': learning_rate, 'test_size': test_size, 'seed': seed,
        'weight': weight, 'weight_decay': weight_decay,
        'optimal_val_loss': opt_val_loss, 'options': options}
    desc = json.dumps(params_dict, sort_keys=True) # description of the trained model (for saving it)

    # add additional values to params_dict (not to be shown in the description)
    params_dict['input_coords'] = input_coords
    params_dict['output_coords'] = output_coords

    # // Get an existing model (if inputed) or add a new one to be trained (doesnt make sense to plot with this one)
    # get overview file
    resume_training = False
    if ('parallel' in options and options['parallel'] is False) or \
            ('parallel' not in options): # no parallel training, so we can save the model overview
        model_overview_file_name = '{}model_overview.csv'.format(
            saved_models_path
        ) # if there are other saved models, load an overview of these
        makedirs(saved_models_path) 
        if not os.path.exists(model_overview_file_name): 
            df_overview = pd.DataFrame(columns=['id', 'description'])
            max_id = 0
        else:
            df_overview = pd.read_csv(model_overview_file_name, index_col=0)  # read model overview csv file
            max_id = np.max(df_overview['id'].values)

        # get model_id, model params etc.
        if model_id is None: # none given as input
            model_id = max_id + 1 # train a new one
        if model_id not in df_overview['id'].values:  # create new model ID
            initial_print += '\nnew model_id={}'.format(model_id)
            df_ov_app = pd.DataFrame([[model_id, desc]],
                                     columns=['id', 'description'])
            df_overview = pd.concat([df_overview, df_ov_app], ignore_index=True)
            df_overview.to_csv(model_overview_file_name)
        else:
            initial_print += '\nmodel_id already exists -> resume training'  # resume training if model already exists, take its param_dict
            resume_training = True
            desc = (df_overview['description'].loc[
                df_overview['id'] == model_id]).values[0]
            params_dict = json.loads(desc) # overwrites params_dict for ONJODE model
    initial_print += '\nmodel params:\n{}'.format(desc)
    if 'resume_training' in options and options['resume_training'] is True: # continue training the model, otherwise it just pulls the model and its specs
        resume_training = True

    # // get all needed paths: directory for model, inside there is last_checkpoint, best_checkpoint, metric_id-, and plots folders.
    model_path = '{}id-{}/'.format(saved_models_path, model_id)
    makedirs(model_path)
    model_path_save_last = '{}last_checkpoint/'.format(model_path)
    model_path_save_best = '{}best_checkpoint/'.format(model_path)
    makedirs(model_path_save_last)
    makedirs(model_path_save_best)
    model_metric_file = '{}metric_id-{}.csv'.format(model_path, model_id)
    plot_save_path = '{}plots/'.format(model_path)
    if 'save_extras' in options:
        save_extras = options['save_extras'] # dict of options for saving plots
    else:
        save_extras = {}

    # // get the model & optimizer
    model = models.ONJODE(**params_dict) 
    model_name = 'ONJODE'
    print("train: created model instance")
    
    train_readout_only = False
    if 'train_readout_only' in options:
        train_readout_only = options['train_readout_only']
    model.to(device)  # pass model to CPU/GPU
    if not train_readout_only: #all parameters
       optimizer = torch.optim.Adam(
           model.parameters(), lr=learning_rate, weight_decay=0.0005)
       scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, min_lr = 1e-6, threshold=0.01)
    else: # just readout_map parameters
        optimizer = torch.optim.Adam(
            model.readout_map.parameters(), lr=learning_rate,
            weight_decay=0.0005)
    gradient_clip = None
    if 'gradient_clip' in options:
        gradient_clip = options["gradient_clip"]
    print("train: finished model & optimizer setup") # TODO: remove
    
    # // load saved model if wanted/possible OR initiate a new one
    best_val_loss = np.infty
    metr_columns = METR_COLUMNS # METR_COLUMNS contains 'epoch', 'train_time', 'val_time', 'train_loss', 'val_loss', 'optimal_val_loss', 'test_loss', 'optimal_test_loss', 'evaluation_mean_diff'
    print(metr_columns)
    if resume_training:
        print("resume training ...")
        initial_print += '\nload saved model ...'
        try:
            if 'load_best' in options and options['load_best']: # sends these models the saved model specs to device
                models.get_ckpt_model(model_path_save_best, model, optimizer,
                                      device)
            else:
                models.get_ckpt_model(model_path_save_last, model, optimizer,
                                      device)
            df_metric = pd.read_csv(model_metric_file, index_col=0)
            df_metric = update_metric_df_to_new_version(
                df_metric, model_metric_file)
            best_val_loss = np.min(df_metric['val_loss'].values)
            model.epoch += 1
            model.weight_decay_step()
            initial_print += '\nepoch: {}, weight: {}'.format(
                model.epoch, model.weight)
        except Exception as e:
            initial_print += '\nloading model failed -> initiate new model'
            initial_print += '\nException:\n{}'.format(e)
            resume_training = False
    if not resume_training:
        print("initiate new model ...")
        initial_print += '\ninitiate new model ...'
        df_metric = pd.DataFrame(columns=metr_columns) # df_metric is a dataframe storing all the METR_COLUMNS info relating to training and evaluation      # NOTE: pandas 1.0.4 --> 1.3.5 updated

    # ---------- plot only option ------------
    if plot_only:
        print("!! plot only mode !!") # initiates or loads a model for plotting, then stops before any more training is done
        batch = next(iter(dl_test)) # first batch of data in test set
        model.epoch -= 1
        initial_print += '\nplotting ...'
        plot_filename = 'demo-plot_epoch-{}'.format(model.epoch)
        plot_filename = plot_filename + '_path-{}.pdf'
        plot_error_dist = None
        if "plot_error_dist" in options:
            plot_error_dist = options["plot_error_dist"] # kwargs for plot_error_distribution function
        ref_model_to_use = None
        if "ref_model_to_use" in options:
            ref_model_to_use = options["ref_model_to_use"] # reference model in plots (None implies conditional expectation) str
        curr_opt_loss, c_model_test_loss, ed_paths = plot_one_path_with_pred(
            device, model, batch, stockmodel_test,
            testset_metadata['dt'], testset_metadata['maturity'],
            path_to_plot=paths_to_plot, save_path=plot_save_path,
            filename=plot_filename, 
            functions=functions,
            use_cond_exp=use_cond_exp, output_coords=output_coords,
            model_name=model_name, save_extras=save_extras, ylabels=ylabels,
            legendlabels=legendlabels, input_coords=input_coords,
            same_yaxis=plot_same_yaxis, plot_obs_prob=plot_obs_prob,
            dataset_metadata=testset_metadata, reuse_cond_exp=True,
            which_loss=which_val_loss,
            plot_error_dist=plot_error_dist, ref_model_to_use=ref_model_to_use)
        eval_msd = None
        if "plot_only_evaluate" in options and options["plot_only_evaluate"]: # also evaluate model in plot
            print("evaluate model ...")
            eval_msd = evaluate_model(
                model=model, dl_test=dl_test, device=device,
                stockmodel_test=stockmodel_test,
                testset_metadata=testset_metadata,
                mult=mult, use_cond_exp=use_cond_exp,
                eval_use_true_paths=eval_use_true_paths, return_paths=False)
        if SEND:
            files_to_send = []
            caption = "{} - id={}".format(model_name, model_id)
            for i in paths_to_plot:
                files_to_send.append(
                    os.path.join(plot_save_path, plot_filename.format(i)))
            if ed_paths is not None:
                files_to_send += ed_paths
            SBM.send_notification(
                text='finished plot-only: {}, id={}\n'
                     'optimal test loss: {}\n'
                     'current test loss: {}\n'
                     'evaluation metric (test set): {}, epoch: {}\n\n{}'.format(
                    model_name, model_id, curr_opt_loss,
                    c_model_test_loss, eval_msd, model.epoch, desc),
                chat_id=config.CHAT_ID,
                files=files_to_send,
                text_for_files=caption
            )
        initial_print += '\noptimal test-loss (with current weight={:.5f}): ' \
                         '{:.5f}'.format(model.weight, curr_opt_loss) # optimal is with conditional expectation
        initial_print += '\nmodel test-loss (with current weight={:.5f}): ' \
                         '{:.5f}'.format(model.weight, c_model_test_loss) # from model
        if eval_msd is not None:
            initial_print += '\nevaluation metric (test set): {:.3e}'.format(
                eval_msd)
        print(initial_print)
        return 0

    # ---------------- TRAINING ----------------
    skip_training = True
    if model.epoch <= epochs:  
        skip_training = False

        # send telegram notification
        if SEND:
            SBM.send_notification(
                text='start training - model id={}'.format(model_id),
                chat_id=config.CHAT_ID)
        initial_print += '\n\nmodel overview:'
        print(initial_print)
        print(model, '\n')

        # compute number of parameters
        nr_params = 0
        for name, param in model.named_parameters():
            skip = False
            for p_name in ['gru_debug', 'classification_model']:
                if p_name in name:
                    skip = True
            if not skip:
                nr_params += param.nelement()  # count number of parameters
        print('# parameters={}\n'.format(nr_params))

        # compute number of trainable params
        nr_trainable_params = 0
        for pg in optimizer.param_groups:
            for p in pg['params']:
                nr_trainable_params += p.nelement()
        print('# trainable parameters={}\n'.format(nr_trainable_params))
        print('start training ...')

    metric_app = []
    # // training
    while model.epoch <= epochs:
        print("start epoch: {}".format(model.epoch))
        t = time.time()  
        model.train()  
        for i, b in tqdm.tqdm(enumerate(dl)):
            print("training batch: {}".format(i))
            optimizer.zero_grad() 

            times = b["times"]  
            time_ptr = b["time_ptr"]  
            X = b["X"].to(device)
            M = b["M"]
            if M is not None:
                M = M.to(device)
            start_M = b["start_M"]
            if start_M is not None:
                start_M = start_M.to(device)

            obs_idx = b["obs_idx"]
            n_obs_ot = b["n_obs_ot"].to(device)
            n_space_ot = b["n_space_ot"].to(device) 
            space_points = b["space_points"].to(device) 
            batch_ptr = b["batch_ptr"] 
            eval_points = b["eval_points"].to(device)
            eval_ptr = b["eval_ptr"]

            # // final time hidden state and loss calculation
            _, loss = model(eval_points=eval_points, eval_ptr=eval_ptr, times=times, time_ptr=time_ptr, X=X, obs_idx=obs_idx,
                             space_points=space_points, n_space_ot=n_space_ot, batch_ptr=batch_ptr, 
                             delta_t=delta_t, T=T, n_obs_ot=n_obs_ot, M=M) 

            loss.backward()     
            
            if gradient_clip is not None:
                print(gradient_clip)
                nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=gradient_clip)
            # diagnostics
            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5 
            print(f'Gradient norm: {total_norm}')
            print(f"ODE gradients: {sum(p.grad.data.norm(2)**2 for p in model.ode_f.parameters() if p.grad is not None) ** 0.5:.4f}")
            print(f"GenK gradients: {sum(p.grad.data.norm(2)**2 for p in model.gen_kernel.parameters() if p.grad is not None) ** 0.5:.4f}")
            print(f"Encoder gradients: {sum(p.grad.data.norm(2)**2 for p in model.encoder_map.parameters() if p.grad is not None) ** 0.5:.4f}")
            print(f"Readout gradients: {sum(p.grad.data.norm(2)**2 for p in model.readout_map.parameters() if p.grad is not None) ** 0.5:.4f}")
            print("Loss value (batch): {}".format(loss.detach().cpu().numpy())) 

            optimizer.step()  # update weights by ADAM optimizer

            if ANOMALY_DETECTION:
                print(r"current loss: {}".format(loss.detach().cpu().numpy()))
            if DEBUG:
                print("DEBUG MODE: stop training after first batch")
                break
        
        train_loss = loss.detach().cpu().item()  # get loss as float number
        print("TRAIN LOSS EPOCH {}: {:.5f}".format(model.epoch, train_loss))

        train_time = time.time() - t  
        # -------- evaluation --------
        print("evaluating ...")
        t = time.time()
        batch = None
        with torch.no_grad():  
            loss_val = 0
            num_obs = 0
            eval_msd = 0
            model.eval() 
            for i, b in enumerate(dl_val): # VALIDATION LOSS
                print("in batch {}".format(i))
                times = b["times"] 
                time_ptr = b["time_ptr"]
                space_points = b["space_points"] 
                X = b["X"].to(device)
                M = b["M"]
                if M is not None:
                    M = M.to(device)
                
                obs_idx = b["obs_idx"] 
                n_obs_ot = b["n_obs_ot"].to(device)
                n_space_ot = b["n_space_ot"].to(device)
                space_points = b["space_points"].to(device)
                batch_ptr = b["batch_ptr"]
                eval_points = b["eval_points"].to(device)
                eval_ptr = b["eval_ptr"]

                _, c_loss = model(eval_points=eval_points, eval_ptr=eval_ptr, times=times, time_ptr=time_ptr, X=X, obs_idx=obs_idx,
                                   space_points=space_points, n_space_ot=n_space_ot, batch_ptr=batch_ptr, 
                                   delta_t=delta_t, T=T, n_obs_ot=n_obs_ot, M=M)

                loss_val += c_loss
                num_obs += 1  # count number of batches

            loss_val = (loss_val / num_obs).detach().cpu().item() # averages the losses over batches
            scheduler.step(loss_val)
            print("VAL LOSS EPOCH {}: {:.5f}".format(model.epoch, loss_val))

            # mean squared difference evaluation ON TEST SET
            if 'evaluate' in options and options['evaluate']:
                print("Calculate msq")
                eval_msd = evaluate_model(
                    model=model, dl_test=dl_test, device=device, 
                    stockmodel_test=stockmodel_test,
                    testset_metadata=testset_metadata,
                    mult=mult, use_cond_exp=use_cond_exp,
                    eval_use_true_paths=eval_use_true_paths, return_paths=False) 

            val_time = time.time() - t
            eval_msd = eval_msd / num_obs
            print_str = "epoch {}, weight={:.5f}, train-loss={:.5f}, " \
                        "optimal-val-loss={:.5f}, val-loss={:.5f}, ".format(
                model.epoch, model.weight, train_loss, opt_val_loss, loss_val)
            print(print_str)

        curr_metric = [model.epoch, train_time, val_time, train_loss,
                               loss_val, opt_val_loss, None, None]
        if 'evaluate' in options and options['evaluate']:
            curr_metric.append(eval_msd)
            print("evaluation mean square difference (test set): {:.5f}".format(
                eval_msd))
        else:
            curr_metric.append(None)
        metric_app.append(curr_metric)

        print("finished evaluation")

        # save model
        save_every = 20
        if model.epoch % save_every == 0: 
            if plot:
                print("in plot")
                batch = next(iter(dl_test)) # test batch
                print('plotting ...')
                plot_error_dist = None
                if "plot_error_dist" in options:
                    plot_error_dist = options["plot_error_dist"] # for plot_error_distribution function
                print('-'*50)
                print(options)
                print(plot_error_dist)
                print('-'*50)
                ref_model_to_use = None
                if "ref_model_to_use" in options:
                    ref_model_to_use = options["ref_model_to_use"] # reference model in plots (None implemented for ONJODE)
                plot_filename = 'epoch-{}'.format(model.epoch)
                plot_filename = plot_filename + '_path-{}.pdf'
                curr_opt_test_loss, c_model_test_loss, _ = plot_one_path_with_pred(
                    device=device, model=model, batch=batch,
                    stockmodel=stockmodel, delta_t=delta_t, T=T, 
                    path_to_plot=paths_to_plot, save_path=plot_save_path,
                    filename=plot_filename, 
                    model_name=model_name, save_extras=save_extras,
                    ylabels=ylabels, use_cond_exp=use_cond_exp,
                    legendlabels=legendlabels, reuse_cond_exp=True,
                    output_coords=output_coords, input_coords=input_coords,
                    same_yaxis=plot_same_yaxis, plot_obs_prob=plot_obs_prob,
                    dataset_metadata=dataset_metadata,
                    which_loss=which_val_loss, plot_error_dist=plot_error_dist, ref_model_to_use=ref_model_to_use)
                print('optimal test-loss (with current weight={:.5f}): '
                      '{:.5f}'.format(model.weight, curr_opt_test_loss))
                print('model test-loss (with current weight={:.5f}): '
                      '{:.5f}'.format(model.weight, c_model_test_loss))
                curr_metric[-2] = curr_opt_test_loss
                curr_metric[-3] = c_model_test_loss
            
            print('save model ...')
            df_m_app = pd.DataFrame(data=metric_app, columns=metr_columns) 
            df_metric = pd.concat([df_metric, df_m_app], ignore_index=True) # saves all metrics for the epoch 
            df_metric.to_csv(model_metric_file)
            models.save_checkpoint(model, optimizer, model_path_save_last,
                                   model.epoch)
            metric_app = []
            print('saved!')
        if loss_val < best_val_loss: # if after any epoch the validation loss is better than all previous models, save as best model
            print('save new best model: last-best-loss: {:.5f}, '
                  'new-best-loss: {:.5f}, epoch: {}'.format(
                best_val_loss, loss_val, model.epoch))
            df_m_app = pd.DataFrame(data=metric_app, columns=metr_columns)
            df_metric = pd.concat([df_metric, df_m_app], ignore_index=True)
            df_metric.to_csv(model_metric_file)
            models.save_checkpoint(model, optimizer, model_path_save_last,
                                   model.epoch)
            models.save_checkpoint(model, optimizer, model_path_save_best,
                                   model.epoch)
            metric_app = []
            best_val_loss = loss_val
            print('saved!')
        print("-"*100)

        model.epoch += 1
        model.weight_decay_step()

    # send notification
    if SEND and not skip_training:
        files_to_send = [model_metric_file]
        caption = "{} - id={}".format(model_name, model_id)
        if plot:
            for i in paths_to_plot:
                files_to_send.append(
                    os.path.join(plot_save_path, plot_filename.format(i)))
        SBM.send_notification(
            text='finished training: {}, id={}\n\n{}'.format(
                model_name, model_id, desc),
            chat_id=config.CHAT_ID,
            files=files_to_send,
            text_for_files=caption)

    return 0

def compute_optimal_val_loss(
        dl_val, stockmodel, delta_t, T, mult=None,
        store_cond_exp=False, 
        which_loss='operator'): 
    """
    compute optimal evaluation loss (with the true cond. exp.) on the
    test-dataset
    :param dl_val: torch.DataLoader, used for the validation dataset
    :param stockmodel: stock_model.StockModel instance
    :param delta_t: float, the time_delta
    :param T: float, the terminal time
    :param mult: None or int, the factor by which the dimension is multiplied
    :param store_cond_exp: bool, whether to store the conditional expectation
    :return: float (optimal loss)
    """
    opt_loss = 0
    num_obs = 0
    print("1")
    for i, b in enumerate(dl_val):
        times = b["times"]
        time_ptr = b["time_ptr"]
        X = b["X"].detach().cpu().numpy()
        start_X = b["start_X"].detach().cpu().numpy()
        obs_idx = b["obs_idx"]
        n_obs_ot = b["n_obs_ot"].detach().cpu().numpy()
        space_points = b["space_points"].detach().cpu().numpy() 
        observed_dates = b["observed_dates"] 
        n_space_ot = b["n_space_ot"].detach().cpu().numpy() 
        batch_ptr = b["batch_ptr"] 
        time_idx = b["time_idx"]
        space_idx = b["space_idx"]

        M = b["M"]
        start_M = b["start_M"]
        if M is not None:
            M = M.detach().cpu().numpy()
            start_M = start_M.detach().cpu().numpy()
        num_obs += 1
        print("made it to batch {}".format(i))
        opt_loss += stockmodel.get_optimal_loss( 
            times, time_ptr, X, obs_idx, delta_t, T, start_X, n_obs_ot,
            space_points, observed_dates, n_space_ot, batch_ptr, time_idx, space_idx,
            M=M, start_M=start_M, 
            mult=mult, store_and_use_stored=store_cond_exp,
            which_loss=which_loss) # goes to get_optimal_loss of synthetic datasets to compute_cond_exp of the synthetic dataset, which computes the data's known cond. exp., stepwise, and computes its overall loss.
        
    return opt_loss / num_obs


def evaluate_model(
        model, dl_test, device, stockmodel_test, testset_metadata,
        mult, use_cond_exp, eval_use_true_paths, return_paths=False): 
    """
    evaluate the model on the test set

    Args:
        model:
        dl_test:
        device:
        stockmodel_test:
        testset_metadata:
        mult:
        use_cond_exp:
        eval_use_true_paths:

    Returns: evaluation metric

    """
    eval_msd = 0.
    for i, b in enumerate(dl_test):
        times = b["times"]
        time_ptr = b["time_ptr"]
        X = b["X"].to(device)
        M = b["M"]
        if M is not None:
            M = M.to(device)
        start_M = b["start_M"]
        if start_M is not None:
            start_M = start_M.to(device)
        start_X = b["start_X"].to(device)
        obs_idx = b["obs_idx"]
        n_obs_ot = b["n_obs_ot"].to(device)
        space_points = b["space_points"].to(device) 
        observed_dates = b["observed_dates"] 
        n_space_ot = b["n_space_ot"].to(device) 
        batch_ptr = b["batch_ptr"] 
        time_idx = b["time_idx"] 
        space_idx = b["space_idx"] 
        eval_points = b["eval_points"].to(device)
        eval_ptr = b["eval_ptr"]
        space_coords = b["space_coords"].to(device) 

        true_paths = b["true_paths"]
        true_mask = b["true_mask"]


        if use_cond_exp and not eval_use_true_paths:
            true_paths = None
            true_mask = None
        _eval_msd = model.evaluate( # does forward pass 
            eval_points=eval_points, eval_ptr=eval_ptr, times=times, time_ptr=time_ptr, X=X, obs_idx=obs_idx,
            space_points=space_points, observed_dates=observed_dates, n_space_ot=n_space_ot, batch_ptr=batch_ptr, time_idx=time_idx, space_idx=space_idx,
            space_coords=space_coords,
            delta_t=testset_metadata["dt"],
            T=testset_metadata["maturity"],
            start_X=start_X, n_obs_ot=n_obs_ot,
            stockmodel=stockmodel_test, M=M,
            start_M=start_M, true_paths=true_paths,
            true_mask=true_mask, mult=mult,
            use_stored_cond_exp=True, ) 
        
        eval_msd += _eval_msd

    return eval_msd


# ADDING: modifying this function for the ONJODE setting --> errors now add over time and for each time over space points
def compute_prediction_errors(
        model_pred, model_t, true_paths, true_times, output_coords,
        original_out_dim, eval_times):
    """
    compute the error distribution of the model predictions
    Args:
        model_pred: np.array, the model predictions,
            shape: (time_steps, bs, dim)                                                  --> either single spatial point included, or several retrained model predictions at different spatial points
        model_t: np.array, the time points of the model predictions,
            shape: (time_steps,)
        true_paths: np.array, the true paths, shape: (bs, dim, time_steps, # ADDED: now also spatial_grid..)
        true_times: np.array, the time points of the true paths,
            shape: (time_steps,)
        output_coords: list of int, the coordinates corresponding to the model            --> do not need with function applications
            output in the extended X (after function applications)
        original_out_dim: int, the original dimension of the output (i.e.,
            without function applications)
        eval_times: list of float, the times where to evaluate the error

        # will also need an eval point in space since the model learns the conditional exp at a specific point in space.
        # can construct the surface at all points in space ---> must retrain the model for each spatial point

    Returns:
    """
    if len(output_coords) > original_out_dim:
        output_coords = output_coords[:original_out_dim]
    eval_times = sorted(eval_times)
    # get the model predictions at the evaluation times --> ADDED: spatial dimension
    ind = np.searchsorted(model_t, eval_times, side='right') - 1
    model_pred_eval = model_pred[:, :, :original_out_dim, :][ind]
    # get the true paths at the evaluation times
    ind = np.searchsorted(true_times, eval_times, side='right') - 1
    true_paths = np.transpose(true_paths, (2, 0, 1, 3)) # (bs, dim, time, space) -> (time, bs, dim, space)
    true_paths_eval = true_paths[:, :, output_coords, :][ind]

    # compute the errors
    error = model_pred_eval - true_paths_eval # shape : (eval_times, bs, dim, space)

    return error

def plot_error_distribution(
        true_paths, true_times, model_preds, model_ts, output_coords, 
        original_out_dim, colors, eval_times=None, model_names=None, save_path='',
        filename='error_dist_model_{}_time_{}.pdf', coord_names=None, space_grids=None):
    print("in plot error distribution ...")
    # replace special times ("mid", "end") by the actual times
    T = true_times[-1]
    _evl_times = []
    for t in eval_times:
        if t == "mid":
            _evl_times.append(T/2)
        elif t in ["end", "last"]:         
            _evl_times.append(T)                              
        else:
            _evl_times.append(t)
    eval_times = _evl_times

    # compute the errors for each model, shape: (model, eval_times, bs, dim, space)
    errors = [compute_prediction_errors(
        model_pred, model_t, true_paths, true_times, output_coords,
        original_out_dim, eval_times) for model_pred, model_t in
              zip(model_preds, model_ts)]

    # get error statistics
    # first across batches: since the space distribution is the same for all of them, expect the same performance relative to space grid
    nperrors = np.array(errors)
    mean_errors = np.mean(nperrors, axis=2) # mean over batches, implies dimension : (model, eval_times, dim, space)
    std_errors = np.std(nperrors, axis=2)
    cols = ["model", "eval_time", "coord", "mean in space", "std in space"]
    dat = []
    for i, model_name in enumerate(model_names):
        for j, eval_time in enumerate(eval_times): 
            for k, coord in enumerate(output_coords):
                dat.append([model_name, eval_time, coord, mean_errors[i, j, k, :],
                            std_errors[i, j, k, :]]) # store vectors across space of the mean and std errors in batches

    print("saving error statistics ...")
    df = pd.DataFrame(columns=cols, data=dat)
    err_stats_file = save_path + "error_stats.csv"
    df.to_csv(err_stats_file)
    paths = [err_stats_file]

    print("plotting error distributions at eval times, over space")
    # change this to make plots for each dimension, at each eval_time, over space points (i.e. errors vs space for model i at eval time t)
    for i, error in enumerate(errors): # for each model, it has the calculated errors. error is dim (eval_times, bs, dim, space)
        for j, eval_time in enumerate(eval_times):
            plt.figure()
            mean_error_ij = mean_errors[i, j, :, :] # shape (dim, space)
            std_error_ij = std_errors[i, j, :, :] # shape (dim, space)
            for d in range(original_out_dim):
                mean_err_ijd = mean_error_ij[d] # shape (space,)
                std_error_ijd = std_error_ij[d]
                if space_grids is None:
                    x_space = np.arange(len(mean_err_ijd)) # assuming space points are indexed from 0 to spatial_grid-1
                # instead of indices, plot the actual points --> they are in model.space_grid_1d or
                else:
                    x_space = space_grids # for now just connsidering 1 space dimension, otherwise would be several [spacegrids, spacegrids, ...]
                plt.errorbar(
                    x_space, mean_err_ijd, yerr=std_error_ijd, fmt='-o', label='Coord {}'.format(d))
            plt.xlabel('Space Point')
            plt.ylabel('Prediction Error')
            plt.title('Error distribution at time {} for model {}'.format(eval_time, model_names[i]))
            plt.legend()
            plt.tight_layout()
            plt.savefig(filename.format(i, eval_time))
            plt.close()
            paths.append(filename.format(i, eval_time))

    return paths


def plot_one_path_with_pred(
        device, model, batch, stockmodel, delta_t, T,
        path_to_plot=(0,), save_path='', filename='plot_{}.pdf',
        model_name=None, ylabels=None,
        legendlabels=None,
        save_extras={'bbox_inches': 'tight', 'pad_inches': 0.01},
        use_cond_exp=True, same_yaxis=False,
        plot_obs_prob=False, dataset_metadata=None,
        reuse_cond_exp=True, output_coords=None, input_coords=None,
        which_loss='operator',
        plot_error_dist=None, ref_model_to_use=None,
):
    """
    plot one path of the stockmodel together with optimal cond. exp. and its
    prediction by the model
    :param device: torch.device, to use for computations
    :param model: models.ONJODE instance
    :param batch: the batch from where to take the paths
    :param stockmodel: stock_model.StockModel instance, used to compute true
            cond. exp.
    :param delta_t: float
    :param T: float
    :param path_to_plot: list of ints, which paths to plot (i.e. which elements
            of the batch)
    :param save_path: str, the path where to save the plot
    :param filename: str, the filename for the plot, should have TWO insertion      # NOW TWO INSERTIONS: path number and dim
            possibilitIES to put the path number and dimension of X
    :param model_name: str or None, name used for model in plots
    :param ylabels: None or list of str of same length as dimension of X
    :param legendlabels: None or list of str of length 4 or 5 (labels of
        i) true path, ii) our model, iii) true cond. exp., iv) observed values,
        v) true values at observation times (only if noisy observations are
        used))
    :param save_extras: dict with extra options for saving plot
    :param use_cond_exp: bool, whether to plot the conditional expectation
    :param same_yaxis: bool, whether to plot all coordinates with same range on
        y-axis
    :param plot_obs_prob: bool, whether to plot the probability of an
        observation for all times
    :param dataset_metadata: needed if plot_obs_prob=true, the metadata of the
        used dataset to extract the observation probability
    :param reuse_cond_exp: bool, whether to reuse the conditional expectation
        from the last computation
    :param output_coords: None or list of ints, the coordinates corresponding to
        the model output
    :param loss_quantiles: None or list of floats, the quantiles to plot for the
        loss
    :param input_coords: None or list of ints, the coordinates corresponding to
        the model input
    :param which_loss: str, the loss function to use for the computation
    :param plot_error_dist: None or dict, the kwargs for plotting the error
        distribution. Inluding eval_times: list of times (floats) to evaluate errors over space    # added detail
    :param ref_model_to_use: None or str, the reference models to use in the
        plots, None uses the standard reference model, usually the conditional
        expectation

    :return: tuple of (optimal loss, model loss, paths to the plots)


    :return: optimal loss
    """
    print("in plot_one_path_with_pred..") # TODO: remove for debugging
    if model_name is None or model_name == "ONJODE":
        model_name = 'Our model'

    prop_cycle = plt.rcParams['axes.prop_cycle']  # change style of plot?
    colors = prop_cycle.by_key()['color']

    makedirs(save_path)  # create a directory

    times = batch["times"] # observation times of the batch
    time_ptr = batch["time_ptr"]
    X = batch["X"].to(device)
    M = batch["M"]
    if M is not None:
        M = M.to(device)
    start_X = batch["start_X"].to(device)
    start_M = batch["start_M"]
    if start_M is not None:
        start_M = start_M.to(device)
    obs_idx = batch["obs_idx"]
    n_obs_ot = batch["n_obs_ot"].to(device)
    space_points = batch["space_points"].to(device) 
    n_space_ot = batch["n_space_ot"].to(device) 
    batch_ptr = batch["batch_ptr"] 
    time_idx = batch["time_idx"] 
    space_idx = batch["space_idx"] 
    true_X = batch["true_paths"]
    eval_points = batch["eval_points"].to(device)
    eval_ptr = batch["eval_ptr"]
    space_coords = batch["space_coords"].to(device)

    # dim does not take the function applications into account
    batch_size, dim, time_points, space_grid_size = true_X.shape
    if output_coords is None:
        output_coords = list(range(dim))
        out_dim = dim
    else:
        # if output_coords is given, then they also include the function -- for i/o code
        # #   applications
        # mult = len(functions)+1 if functions is not None else 1
        # out_dim = int(len(output_coords)/mult)
        out_dim = int(len(output_coords)) # no several functions
    true_M = batch["true_mask"]
    observed_dates = batch['observed_dates'] # (bs, timesteps, spacegrid)
    observed_times = observed_dates.sum(axis=2) > 0 # (bs, timesteps)
    # if "obs_noise" in batch:
    #     obs_noise = batch["obs_noise"]
    # else:
    #     obs_noise = None
    path_t_true_X = np.linspace(0., T, int(np.round(T / delta_t)) + 1) # length time_steps+1

    model.eval()  # put model in evaluation mode
    # model predictions
    res = model.get_pred(
        eval_points=eval_points, eval_ptr=eval_ptr,times=times, time_ptr=time_ptr, X=X, obs_idx=obs_idx, 
        delta_t=delta_t, T=T, n_obs_ot=n_obs_ot,
        space_points=space_points, n_space_ot=n_space_ot, batch_ptr=batch_ptr,
        space_coords=space_coords, paths_to_plot=path_to_plot, 
        M=M, which_loss=which_loss)
    path_y_pred = res['pred'].detach().cpu().numpy() 
    path_t_pred = res['pred_t']
    current_model_loss = res['loss'].detach().cpu().item()
    
    if use_cond_exp: # not actual X paths, now calculating the TRUE conditional expectation to
        if M is not None:
            M = M.detach().cpu().numpy() 
            start_M = start_M.detach().cpu().numpy()#[:, :dim, :] # see what dim is here
        X_ = X.detach().cpu().numpy()#[:, :dim]
        start_X_ = start_X.detach().cpu().numpy()#[:, :dim, :] # added space dim
        if M is not None:
            M = M#[:, :dim]            # REMOVED: all the :dim as these are for the case of io function applications
        # true conditional expectation
        res_sm = stockmodel.compute_cond_exp(
            times, time_ptr, X_, obs_idx,
            delta_t, T, start_X_, n_obs_ot.detach().cpu().numpy(), space_points.detach().cpu().numpy(), 
            observed_dates, n_space_ot.detach().cpu().numpy(), batch_ptr, time_idx, space_idx, M=M, start_M=start_M, 
            return_path=True, get_loss=True, weight=model.weight,
            store_and_use_stored=reuse_cond_exp,
            which_loss=which_loss, ref_model=ref_model_to_use) 

        opt_loss, path_t_true, path_s_true, path_y_true = res_sm[:4] # optimal loss is that calculated with true conditional expectation

        # get reference model predictions - NOT IMPLEMENTED for ONJODE
        if (plot_error_dist is not None
                and "additional_ref_models" in plot_error_dist):
            print("plot_error_dist=True, getting data for additional ref models")
            add_model_preds = []
            add_model_ts = []
            for ref_model in plot_error_dist["additional_ref_models"]:
                res_sm_add = stockmodel.compute_cond_exp(
                    times, time_ptr, X_, obs_idx.detach().cpu().numpy(),
                    delta_t, T, start_X_, n_obs_ot.detach().cpu().numpy(),
                    space_points, observed_dates, n_space_ot, batch_ptr, time_idx, space_idx, start_M=start_M.detach().cpu().numpy(), # NEW
                    return_path=True, get_loss=True, weight=model.weight,
                    M=M, store_and_use_stored=False,
                    which_loss=which_loss, ref_model=ref_model)
                _, t, pred = res_sm_add[:3]
                add_model_preds.append(pred)
                add_model_ts.append(t)
    else: # not considering conditional expectation, rather the true paths, which means the optimal loss is zero
        opt_loss = 0
    
    # plot the error distribution
    space_coords = space_coords.detach().cpu().numpy()

    err_dist_paths = None
    if plot_error_dist is not None:
        names = ["ONJODE", "cond. exp."]
        if "additional_ref_models" in plot_error_dist:
            names += plot_error_dist["additional_ref_models"]
        if "model_names" not in plot_error_dist:
            plot_error_dist["model_names"] = names # need model names to specify for the plot
        err_dist_filename = "{}error_distribution_plot_coord{}.pdf".format(
            save_path, "{}")
        model_preds = [path_y_pred,] # (len path_t_pred, bs, dim, space steps) -> model 
        model_ts = [path_t_pred,] # at any point in time (including pre jump times)
        if use_cond_exp:
            model_preds.append(path_y_true)
            model_ts.append(path_t_true)
        if "additional_ref_models" in plot_error_dist:
            model_preds += add_model_preds
            model_ts += add_model_ts
        
        err_dist_paths = plot_error_distribution(
            true_paths=true_X, true_times=path_t_true_X,
            model_preds=model_preds, model_ts=model_ts,
            output_coords=output_coords,
            original_out_dim=out_dim, eval_times=plot_error_dist["eval_times"],
            colors=colors, model_names=plot_error_dist["model_names"],
            save_path=save_path, filename=err_dist_filename,
            coord_names=ylabels, space_grids=space_coords)  # added space grid being passed
    else:
        print("No error distribution plot")

    p=0
    for i in path_to_plot: 
        # synthetic datasets (path_y_true) use i for path index
        # model (path_y_pred) use p for path index in the passed paths_to_plot
        print("input path; {}".format(i))
        # get observation times for path i
        obs_times_bool = observed_times[i] # observed times for path i, shape (time_steps,)
        obs_times = path_t_true_X[obs_times_bool]
        
        # define eval times as observation times + evenly spaced times between them
        eval_times = np.concatenate((obs_times, T*np.array([1.]))) # include T as final time
        for j in range(len(obs_times)-1):
            obs_steps = max(1, round((obs_times[j+1] - obs_times[j]) / delta_t)) # steps to next point
            nr_intermediate = min(obs_steps, round(5*np.log(obs_steps))) # every factor of 10 is one more intermediate step
            evenly_spaced = np.linspace(obs_times[j]+delta_t, obs_times[j+1], nr_intermediate)[:-1] # exclude the last point as it is the next obs_time
            eval_times = np.concatenate((eval_times, evenly_spaced))

        eval_times = np.unique(np.sort(eval_times))
        nr_eval_times = len(eval_times)
        eval_times_idx = [np.argmin(np.abs(path_t_true_X - t)) for t in eval_times]
        eval_t_to_y_pred = [np.argmin(np.abs(path_t_pred - t)) for t in eval_times]
        eval_t_to_y_true = [np.argmin(np.abs(path_t_true - t)) for t in eval_times]


        outcoord_ind = -1
        unobserved_coord = False
        for d in range(dim): # for each dimension of X
            print("in dimension: {}".format(d))
            if d in output_coords:
                outcoord_ind += 1 # track the index in the reduced data (by defined output_coords)
                print("output coordinate: {}".format(outcoord_ind))
            fig, axs = plt.subplots(nr_eval_times, sharex=True, figsize=(10, 3*nr_eval_times))
            if nr_eval_times == 1:
                axs = [axs]
            
            nr_obs_times = 0
            for tind, t_eval in enumerate(eval_times):
                print("in eval time: {}".format(tind))
                eval_time_idx = eval_times_idx[tind]
                obs_time = eval_time_idx in np.where(observed_times[i]==True)[0] # obs_times shape (bs, time_steps) -> (time_step,)
                
                if obs_time:
                    nr_obs_times += 1
                # get the true_X in this dimension at eval_time across the space grid at observed points only
                path_s_obs = []
                path_X_obs = []

                eval_time_idx = eval_times_idx[tind]
                space_grid_bool = observed_dates[i][eval_time_idx] # shape (space_grid_size,)
                for k, od in enumerate(space_grid_bool):
                    if od == 1: # observed in this space coordinate at eval time
                        if true_M is None or (true_M is not None and
                                              true_M[i, d, eval_time_idx, k]==1): # unmasked
                            path_s_obs.append(space_coords[k]) # for now just 1 space dimension
                            path_X_obs.append(true_X[i, d, eval_time_idx, k])
                            
                print("collected true observed points at eval time")
                path_s_obs = np.array(path_s_obs)
                path_X_obs =  np.array(path_X_obs)
                # get the legend labels
                lab0 = legendlabels[0] if legendlabels is not None else 'true path' # the ground truth surface at eval time (across entire space grid)
                lab1 = legendlabels[1] if legendlabels is not None else model_name # model predictions of cond. exp. at eval time (across entire space grid)
                lab2 = legendlabels[2] if legendlabels is not None else 'true conditional expectation' # true cond. exp. at eval time (across entire space grid)
                lab3 = legendlabels[3] if legendlabels is not None else 'observed' # now true surface at eval time (observed time-space)

                # add grid
                axs[tind].grid(True, alpha=0.3)

                # CURRENTLY: Only implemented for 1d space and output dimensions
                # 0. plt the ground truth surface at eval time
                axs[tind].plot(space_coords, true_X[i, d, eval_time_idx, :] , label=lab0, color=colors[0])
                facecolors = colors[0]
                # 3. plot the observed surface (points in space) at eval time --> i.e. the observed points of the true surface (relating to input coords)
                if input_coords is not None and d not in input_coords: # not an input dimension
                    facecolors = 'none'
                    lab3 = '(un)observed'
                    unobserved_coord = True
                axs[tind].scatter(path_s_obs, path_X_obs, label=lab3,
                                  color='red', facecolors=facecolors, s=50, zorder=5)
                # 1. plot the model predicted cond. exp. at eval time (across space grid) --> i.e. the model learned cond. exp. surface (relating to output coords)
                if d in output_coords: # there is a model prediction for this coord
                    # change path_t_pred to path_s_grid
                    axs[tind].plot(space_coords, path_y_pred[eval_t_to_y_pred[tind], p, outcoord_ind, :], label=lab1, color=colors[1])  # path_y_pred shape (time_steps, bs, out_dim, space_grid)
                    if use_cond_exp: # if to plot conditional expectation
                        # 2. plot the true cond. exp. at eval time (across space grid) --> i.e. the true cond. exp. surface (relating to output coords)
                        axs[tind].plot(space_coords, path_y_true[eval_t_to_y_true[tind], i, outcoord_ind, :], label=lab2, linestyle=':', color=colors[2])

                if plot_obs_prob and dataset_metadata is not None: # plot obs prob across space
                    ax2 = axs[tind].twinx()
                    if "obs_scheme" in dataset_metadata:
                        obs_scheme = dataset_metadata["obs_scheme"]
                        # insert dealing with obs_scheme
                    else:
                        obs_perc = dataset_metadata['obs_perc'] # % chance observation at each time
                        space_obs_perc = dataset_metadata.get('space_obs_perc', None)
                        if space_obs_perc is None: # just uniform over space, with J_obs_perc of points sampled
                            space_obs_perc = 1/space_grid_size * np.ones(space_grid_size)
                    ax2.plot(space_coords, space_obs_perc, color="red", label="observation probability")
                    ax2.set_ylim(-0.1, 1.1)
                    ax2.set_ylabel("observation probability")
                    ax2.legend()

                if ylabels and obs_time: # add here an index of the observation time to track
                    print("given ylabel at obs time")
                    axs[tind].set_ylabel(ylabels[tind] + " at \n obs. time $t_{{{}}} = {:.2f}$".format(nr_obs_times, t_eval))
                elif ylabels:
                    print("given ylabel at eval time")
                    axs[tind].set_ylabel(ylabels[tind] + " at \n time $t={:.3f}$".format(t_eval))
                else:
                    if obs_time:
                        print("modifying ylabel at obs time")
                        axs[tind].set_ylabel("$(X)_{{{}}}$, eval time idx {} at \n obs. time $t_{{{}}}={:.2f}$".format(d+1, tind, nr_obs_times, t_eval))
                    else:
                        print("default ylabel at eval time")
                        axs[tind].set_ylabel("$(X)_{{{}}}$, eval time idx {} at \n time $t={:.2f}$".format(d+1, tind, t_eval))
                if same_yaxis: # whether all coords are on the same y-axis range
                    low = np.min(true_X[i, :, :, :])
                    high = np.max(true_X[i, :, :, :])
                    eps = (high - low)*0.05
                    axs[tind].set_ylim([low-eps, high+eps])

            if unobserved_coord:
                print("unobserved coord, modifying legend")
                handles, labels = axs[-1].get_legend_handles_labels()
                l, = axs[-1].plot(
                    [], [], color=colors[0], label='(un)observed',
                    linestyle='none', marker="o", fillstyle="right")
                handles[-1] = l
                labels[-1] = 'unobserved/observed'
                axs[0].legend(handles, labels)
            else:
                axs[0].legend()
            print("saving figure ...")
            plt.tight_layout()
            plt.xlabel('Space $(\\xi)$')
            save = os.path.join(save_path, filename.format(i, d)) #i and d
            plt.savefig(save, **save_extras)
            plt.close()
            unobserved_coord = False
            print("exiting plot_one_path_with_pred.")
        p += 1

    return opt_loss, current_model_loss, err_dist_paths
