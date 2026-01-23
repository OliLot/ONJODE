"""
authors: (NJODEs) Florian Krach, Calypso Herrera (ONJODE extension) Oliver Löthgren


data utilities for creating and loading synthetic test datasets
"""


# =====================================================================================================================
import numpy as np
import json, os, time
from torch.utils.data import Dataset
import torch
import copy
import pandas as pd
from absl import app
from absl import flags
import wget
from zipfile import ZipFile

from configs import config
import synthetic_datasets


# =====================================================================================================================
FLAGS = flags.FLAGS
flags.DEFINE_string("dataset_params", None,
                    "name of the dict with data hyper-params")
flags.DEFINE_string("dataset_name", None,
                    "name of the dataset to generate")
flags.DEFINE_integer("seed", 0,
                     "seed for making dataset generation reproducible")

hyperparam_default = config.hyperparam_default
_STOCK_MODELS = synthetic_datasets.DATASETS
_OPERATOR_MODELS = synthetic_datasets.OPERATOR_DATASETS # NEW: separate dictionary for ONJODE approach
data_path = config.data_path
training_data_path = config.training_data_path


# =====================================================================================================================
def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def get_dataset_overview(training_data_path=training_data_path):
    data_overview = '{}dataset_overview.csv'.format(
        training_data_path)
    makedirs(training_data_path)
    if not os.path.exists(data_overview): # changed due to datatype error the below. Used to be pd.DataFrame(data=None, columns=['name', 'id', 'description'], dtype=object)
        df_overview = pd.DataFrame({
            'name': pd.Series(dtype='string'),
            'id': pd.Series(dtype='string'),
            'description': pd.Series(dtype='string'),
        })
    else:
        df_overview = pd.read_csv(data_overview, index_col=0)
    return df_overview, data_overview


def create_dataset(
        stock_model_name="BlackScholes", 
        hyperparam_dict=hyperparam_default,
        seed=0):
    """
    create a synthetic dataset using one of the stock-models
    :param stock_model_name: str, name of the stockmodel, see _STOCK_MODELS
    :param hyperparam_dict: dict, contains all needed parameters for the model
            it can also contain additional options for dataset generation:
                - masked    None, float or array of floats. if None: no mask is
                            used; if float: lambda of the poisson distribution;
                            if array of floats: gives the bernoulli probability
                            for each coordinate to be observed
                - timelag_in_dt_steps   None or int. if None: no timelag used;
                            if int: number of (dt) steps by which the 1st
                            coordinate is shifted to generate the 2nd coord.,
                            this is used to generate mask accordingly (such that
                            second coord. is observed whenever the information
                            is already known from first coord.)
                - timelag_shift1    bool, if True: observe the second coord.
                            additionally only at one step after the observation
                            times of the first coord. with the given prob., if
                            False: observe the second coord. additionally at all
                            times within timelag_in_dt_steps after observation
                            times of first coordinate, at each time with given
                            probability (in masked); default: True
                - X_dependent_observation_prob   not given or str, if given:
                            string that can be evaluated to a function that is
                            applied to the generated paths to get the
                            observation probability for each coordinate
                - obs_scheme   dict, if given: specifies the observation scheme
                - obs_noise    dict, if given: add noise to the observations
                            the dict needs the following keys: 'distribution'
                            (defining the distribution of the noise), and keys
                            for the parameters of the distribution (depending on
                            the used distribution); supported distributions
                            {'normal'}. Be aware that the noise needs to be
                            centered for the model to be able to learn the
                            correct dynamics.

    :param seed: int, random seed for the generation of the dataset
    :return: str (path where the dataset is saved), int (time_id to identify
                the dataset)
    """
    df_overview, data_overview = get_dataset_overview()

    np.random.seed(seed=seed)
    hyperparam_dict['model_name'] = stock_model_name
    original_desc = json.dumps(hyperparam_dict, sort_keys=True)
    obs_perc = hyperparam_dict['obs_perc']
    obs_scheme = None
    if "obs_scheme" in hyperparam_dict:
        obs_scheme = hyperparam_dict["obs_scheme"]
    masked = False
    masked_lambda = None
    mask_probs = None
    timelag_in_dt_steps = None
    timelag_shift1 = True
    if ("masked" in hyperparam_dict
            and hyperparam_dict['masked'] not in [None, False]):
        masked = True
        if isinstance(hyperparam_dict['masked'], float):
            masked_lambda = hyperparam_dict['masked']
        elif isinstance(hyperparam_dict['masked'], (tuple, list)):
            mask_probs = hyperparam_dict['masked']
            assert len(mask_probs) == hyperparam_dict['dimension']
        else:
            raise ValueError("please provide a float (poisson lambda) "
                             "in hyperparam_dict['masked']")
        if "timelag_in_dt_steps" in hyperparam_dict:
            timelag_in_dt_steps = hyperparam_dict["timelag_in_dt_steps"]
        if "timelag_shift1" in hyperparam_dict:
            timelag_shift1 = hyperparam_dict["timelag_shift1"]

    stockmodel = _STOCK_MODELS[stock_model_name](**hyperparam_dict)
    # stock paths shape: [nb_paths, dim, time_points]
    stock_paths, dt = stockmodel.generate_paths()
    size = stock_paths.shape
    if obs_scheme is None:
        observed_dates = np.random.random(size=(size[0], size[2]))
        if "X_dependent_observation_prob" in hyperparam_dict:
            print("use X_dependent_observation_prob")
            prob_f = eval(hyperparam_dict["X_dependent_observation_prob"])
            obs_perc = prob_f(stock_paths)
        observed_dates = (observed_dates < obs_perc)*1
        observed_dates[:, 0] = 1
        nb_obs = np.sum(observed_dates[:, 1:], axis=1)
    else:
        if obs_scheme["name"] == "NJODE3-Example4.9":
            """
            implements the observation scheme from Example 4.9 in the NJODE3 
            paper based in 1st coordinate of the process (in case there are more 
            coordinates).
            """
            print("use observation scheme: NJODE3-Example4.9")
            observed_dates = np.zeros(shape=(size[0], size[2]))
            observed_dates[:, 0] = 1
            p = obs_scheme["p"]
            eta = obs_scheme["eta"]
            for i in range(size[0]):
                x0 = stock_paths[i, 0, 0]
                last_observation = x0
                last_obs_time = 0
                for j in range(1, size[2]):
                    v1 = np.random.binomial(1, 1/(j-last_obs_time), 1)
                    v3 = np.random.binomial(1, p, 1)
                    v2 = np.random.normal(0, eta, 1)
                    m = v1*(
                            last_observation+v2 >= stockmodel.next_cond_exp(
                        x0, j*dt, j*dt)) + (1-v1)*v3
                    observed_dates[i, j] = m
                    if m == 1:
                        last_observation = stock_paths[i, 0, j]
                        last_obs_time = j
            nb_obs = np.ones(shape=(size[0],))*size[2]
        else:
            raise ValueError("obs_scheme {} not implemented".format(
                obs_scheme["name"]))
    if masked:
        mask = np.zeros(shape=size)
        mask[:,:,0] = 1
        for i in range(size[0]):
            for j in range(1, size[2]):
                if observed_dates[i,j] == 1:
                    if masked_lambda is not None:
                        amount = min(1+np.random.poisson(masked_lambda),
                                     size[1])
                        observed = np.random.choice(
                            size[1], amount, replace=False)
                        mask[i, observed, j] = 1
                    elif mask_probs is not None:
                        for k in range(size[1]):
                            mask[i, k, j] = np.random.binomial(1, mask_probs[k])
        if timelag_in_dt_steps is not None:
            mask_shift = np.zeros_like(mask[:,0,:])
            mask_shift[:,timelag_in_dt_steps:] = mask[:,0,:-timelag_in_dt_steps]
            if timelag_shift1:
                mask_shift1 = np.zeros_like(mask[:,1,:])
                mask_shift1[:,0] = 1
                mask_shift1[:, 2:] = mask[:, 1, 1:-1]
                mask[:,1,:] = np.maximum(mask_shift1, mask_shift)
            else:
                mult = copy.deepcopy(mask[:,0,:])
                for i in range(1, timelag_in_dt_steps):
                    mult[:,i:] = np.maximum(mult[:,i:], mask[:,0,:-i])
                mask1 = mult*np.random.binomial(1, mask_probs[1], mult.shape)
                mask[:,1,:] = np.maximum(mask1, mask_shift)
        observed_dates = mask
    if "obs_noise" in hyperparam_dict:
        obs_noise_dict = hyperparam_dict["obs_noise"]
        if obs_noise_dict["distribution"] == "normal":
            obs_noise = np.random.normal(
                loc=obs_noise_dict["loc"],
                scale=obs_noise_dict["scale"],
                size=size)
            if 'noise_at_start' in obs_noise_dict and \
                    obs_noise_dict['noise_at_start']:
                pass
            else:
                obs_noise[:,:,0] = 0
        else:
            raise ValueError("obs_noise distribution {} not implemented".format(
                obs_noise_dict["distribution"]))
    else:
        obs_noise = None

    # time_id = int(time.time())
    time_id = 1
    if len(df_overview) > 0:
        time_id = np.max(df_overview["id"].values) + 1
    file_name = '{}-{}'.format(stock_model_name, time_id)
    path = '{}{}/'.format(training_data_path, file_name)
    hyperparam_dict['dt'] = dt
    desc = json.dumps(hyperparam_dict, sort_keys=True)
    if os.path.exists(path):
        print('Path already exists - abort')
        raise ValueError
    df_app = pd.DataFrame(
        data=[[stock_model_name, time_id, original_desc]],
        columns=['name', 'id', 'description']
    )
    df_overview = pd.concat([df_overview, df_app],
                            ignore_index=True)
    df_overview.to_csv(data_overview)

    os.makedirs(path)
    with open('{}data.npy'.format(path), 'wb') as f:
        np.save(f, stock_paths)
        np.save(f, observed_dates)
        np.save(f, nb_obs)
        if obs_noise is not None:
            np.save(f, obs_noise)
    with open('{}metadata.txt'.format(path), 'w') as f:
        json.dump(hyperparam_dict, f, sort_keys=True)

    # stock_path dimension: [nb_paths, dimension, time_points]
    return path, time_id


def create_combined_dataset(
        stock_model_names=("BlackScholes", "OrnsteinUhlenbeck"),
        hyperparam_dicts=(hyperparam_default, hyperparam_default),
        seed=0):
    """
    create a synthetic dataset using one of the stock-models
    :param stock_model_names: list of str, each str is a name of a stockmodel,
            see _STOCK_MODELS
    :param hyperparam_dicts: list of dict, each dict contains all needed
            parameters for the model
    :param seed: int, random seed for the generation of the dataset
    :return: str (path where the dataset is saved), int (time_id to identify
                the dataset)
    """
    df_overview, data_overview = get_dataset_overview()

    assert len(stock_model_names) == len(hyperparam_dicts)
    np.random.seed(seed=seed)

    # start to create paths from first model
    filename = 'combined_{}'.format(stock_model_names[0])
    maturity = hyperparam_dicts[0]['maturity']
    hyperparam_dicts[0]['model_name'] = stock_model_names[0]
    obs_perc = hyperparam_dicts[0]['obs_perc']
    stockmodel = _STOCK_MODELS[stock_model_names[0]](**hyperparam_dicts[0])
    stock_paths, dt = stockmodel.generate_paths()
    last = stock_paths[:, :, -1]
    size = stock_paths.shape
    observed_dates = np.random.random(size=(size[0], size[2]))
    observed_dates = (observed_dates < obs_perc) * 1
    observed_dates[:, 0] = 1

    # for every other model, add the paths created with this model starting at
    #   last point of previous model
    for i in range(1, len(stock_model_names)):
        dt_last = dt
        assert hyperparam_dicts[i]['dimension'] == \
               hyperparam_dicts[i-1]['dimension']
        assert hyperparam_dicts[i]['nb_paths'] == \
               hyperparam_dicts[i-1]['nb_paths']
        filename += '_{}'.format(stock_model_names[i])
        maturity += hyperparam_dicts[i]['maturity']
        hyperparam_dicts[i]['model_name'] = stock_model_names[i]
        stockmodel = _STOCK_MODELS[stock_model_names[i]](**hyperparam_dicts[i])
        _stock_paths, dt = stockmodel.generate_paths(start_X=last)
        size = _stock_paths.shape
        obs_perc = hyperparam_dicts[i]['obs_perc']
        _observed_dates = np.random.random(size=(size[0], size[2]))
        _observed_dates = (_observed_dates < obs_perc) * 1
        assert dt_last == dt
        last = _stock_paths[:, :, -1]
        stock_paths = np.concatenate(
            [stock_paths, _stock_paths[:, :, 1:]], axis=2)
        observed_dates = np.concatenate(
            [observed_dates, _observed_dates[:, 1:]], axis=1)
    nb_obs = np.sum(observed_dates[:, 1:], axis=1)

    time_id = 1
    if len(df_overview) >0:
        time_id = np.max(df_overview["id"].values) + 1
    file_name = '{}-{}'.format(filename, time_id)
    path = '{}{}/'.format(training_data_path, file_name)
    if os.path.exists(path):
        print('Path already exists - abort')
        raise ValueError

    metadata = {'dt': dt, 'maturity': maturity,
                'dimension': hyperparam_dicts[0]['dimension'],
                'nb_paths': hyperparam_dicts[0]['nb_paths'],
                'model_name': 'combined',
                'stock_model_names': stock_model_names,
                'hyperparam_dicts': hyperparam_dicts}
    desc = json.dumps(metadata, sort_keys=True)

    df_app = pd.DataFrame(
        data=[[filename, time_id, desc]],
        columns=['name', 'id', 'description']
    )
    df_overview = pd.concat([df_overview, df_app],
                            ignore_index=True)
    df_overview.to_csv(data_overview)

    os.makedirs(path)
    with open('{}data.npy'.format(path), 'wb') as f:
        np.save(f, stock_paths)
        np.save(f, observed_dates)
        np.save(f, nb_obs)
    with open('{}metadata.txt'.format(path), 'w') as f:
        json.dump(metadata, f, sort_keys=True)

    return path, time_id


def create_LOB_dataset(hyperparam_dict=hyperparam_default,
                       seed=0):
    """
    create Limit Order Book (LOB) datasets.
    Args:
        hyperparam_dict: dict, with all needed hyperparams
        seed: int, the seed for any random numbers
    Returns: path to dataset, id of dataset
    """
    if "which_raw_data" in hyperparam_dict and \
        hyperparam_dict["which_raw_data"] in [
        "ADA_1min", "ADA_1sec", "ADA_5min", "BTC_1min", "BTC_1sec", "BTC_5min",
        "ETH_1min", "ETH_1sec", "ETH_5min",]:
        df_raw = get_rawLOB_dataset2(hyperparam_dict, seed)
    else:
        df_raw = get_rawLOB_dataset1(hyperparam_dict, seed)

    df_overview, data_overview = get_dataset_overview()
    model_name = "LOB"

    np.random.seed(seed=seed)
    hyperparam_dict['model_name'] = model_name
    original_desc = json.dumps(hyperparam_dict, sort_keys=True)
    level = hyperparam_dict["LOB_level"]
    amount_obs = hyperparam_dict["amount_obs"]
    eval_predict_steps = hyperparam_dict["eval_predict_steps"]
    use_volume = hyperparam_dict["use_volume"]
    normalize = hyperparam_dict["normalize"]
    start_at_0 = True
    if "start_at_0" in hyperparam_dict:
        start_at_0 = hyperparam_dict["start_at_0"]
    shift = hyperparam_dict["shift"]
    max_pred_step = int(np.max(eval_predict_steps))
    hyperparam_dict["max_pred_step"] = max_pred_step
    length = amount_obs+max_pred_step

    bpc = []
    apc = []
    bvc = []
    avc = []
    for i in range(max(1, level)):
        bpc.append("bid_price_{}".format(i+1))
        apc.append("ask_price_{}".format(i+1))
        bvc.append("bid_amount_{}".format(i+1))
        avc.append("ask_amount_{}".format(i+1))

    df_raw.set_index("time", inplace=True)
    df_raw = df_raw[bpc+apc+bvc+avc]
    df_raw.drop_duplicates(keep="first", inplace=True)
    df_raw.reset_index(inplace=True)
    df_raw["time"] = pd.to_datetime(
        df_raw["time"], format="%Y-%m-%d %H:%M:%S.%f", utc=True)
    df_raw["time"] = df_raw["time"].apply(lambda x: x.timestamp())
    if normalize:
        price_mean = np.mean(df_raw[bpc+apc].values)
        price_std = np.std(df_raw[bpc+apc].values)
        vol_mean = np.mean(df_raw[bvc+avc].values)
        vol_std = np.std(df_raw[bvc+avc].values)
        df_raw[bpc+apc] = (df_raw[bpc+apc] - price_mean) / price_std
        df_raw[bvc+avc] = (df_raw[bvc+avc] - vol_mean) / vol_std
    else:
        price_mean = 0.
        price_std = 1.
        vol_mean = 0.
        vol_std = 1.
    df_raw["mid_price"] = (df_raw[apc[0]] + df_raw[bpc[0]]) / 2
    # dt = df_raw["time"].diff().min(skipna=True)
    dt = df_raw["time"].diff().median(skipna=True)
    hyperparam_dict["dt"] = dt
    hyperparam_dict["price_mean"] = price_mean
    hyperparam_dict["price_std"] = price_std
    hyperparam_dict["vol_mean"] = vol_mean
    hyperparam_dict["vol_std"] = vol_std

    # make samples
    # l = int(len(df_raw)/length)
    l = int((len(df_raw)-length)/shift)+1
    hyperparam_dict["nb_paths"] = l
    cols = ["mid_price"]
    if level > 0:
        cols += bpc+apc
        if use_volume:
            cols += bvc+avc
    dim = len(cols)
    samples = np.zeros(shape=(l, dim, amount_obs))
    times = np.zeros(shape=(l, amount_obs))
    eval_samples = np.zeros(shape=(l, dim, max_pred_step))
    eval_times = np.zeros(shape=(l, max_pred_step))

    # split up data into samples
    for i in range(l):
        # df_ = df_raw.iloc[i*length:(i+1)*length, :]
        df_ = df_raw.iloc[i*shift:i*shift+length, :]
        df = df_.iloc[:amount_obs, :]
        df_eval = df_.iloc[amount_obs:amount_obs+max_pred_step, :]
        samples[i, :, :] = np.transpose(df[cols].values)
        times[i, :] = df["time"].values - df["time"].values[0]
        eval_samples[i, :, :] = np.transpose(df_eval[cols].values)
        eval_times[i, :] = df_eval["time"].values - df["time"].values[0]

    # generate labels
    eval_labels = np.zeros(shape=(l, len(eval_predict_steps)))
    thresholds = []
    for i, k in enumerate(eval_predict_steps):
        m_minus = np.mean(samples[:, 0, -k:]*price_std + price_mean, axis=1)
        m_plus = np.mean(eval_samples[:, 0, :k]*price_std + price_mean, axis=1)
        pctc = (m_plus - m_minus) / m_minus
        threshold = np.quantile(pctc, q=2/3)
        thresholds.append(threshold)
        eval_labels[pctc > threshold, i] = 1
        eval_labels[pctc < -threshold, i] = -1
        print("steps ahead: ", k)
        print("amount label 1: {}".format(np.sum(eval_labels[:, i] == 1)))
        print("amount label 0: {}".format(np.sum(eval_labels[:, i] == 0)))
        print("amount label -1: {}".format(np.sum(eval_labels[:, i] == -1)))
    hyperparam_dict["thresholds"] = thresholds

    # shift samples and eval samples s.t. they start at 0
    if start_at_0:
        eval_samples -= np.repeat(samples[:, :, 0:1], axis=2,
                                  repeats=eval_samples.shape[2])
        samples -= np.repeat(samples[:, :, 0:1], axis=2,
                             repeats=samples.shape[2])

    # save the dataset
    time_id = 1
    if len(df_overview) > 0:
        time_id = np.max(df_overview["id"].values) + 1
    file_name = '{}-{}'.format(model_name, time_id)
    path = '{}{}/'.format(training_data_path, file_name)
    desc = json.dumps(hyperparam_dict, sort_keys=True)
    if os.path.exists(path):
        print('Path already exists - abort')
        raise ValueError
    df_app = pd.DataFrame(
        data=[[model_name, time_id, original_desc]],
        columns=['name', 'id', 'description']
    )
    df_overview = pd.concat([df_overview, df_app],
                            ignore_index=True)
    df_overview.to_csv(data_overview)

    os.makedirs(path)
    with open('{}data.npy'.format(path), 'wb') as f:
        np.save(f, samples)
        np.save(f, times)
        np.save(f, eval_samples)
        np.save(f, eval_times)
        np.save(f, eval_labels)
    with open('{}metadata.txt'.format(path), 'w') as f:
        json.dump(hyperparam_dict, f, sort_keys=True)

    # stock_path dimension: [nb_paths, dimension, time_points]
    return path, time_id


def get_rawLOB_dataset2(
        hyperparam_dict=hyperparam_default,
        seed=0):
    raw_data_path = config.LOB_data_path2
    makedirs(raw_data_path)
    np.random.seed(seed=seed)
    level = hyperparam_dict["LOB_level"]

    # load raw data and preprocess
    raw_data_dir = "{}{}.csv".format(
        raw_data_path, hyperparam_dict["which_raw_data"])
    if not os.path.exists(raw_data_dir):
        print("raw LOB ({}) data not found -> dowloading ...".format(
            hyperparam_dict["which_raw_data"]))
        zip_file = wget.download(
            "https://polybox.ethz.ch/index.php/s/JJ2eRMB3JmMTVzr/download",
            training_data_path)
        print("extracting zip ...")
        with ZipFile(zip_file, 'r') as zipObj:
            zipObj.extractall(path=training_data_path)
        print("removing zip ...")
        os.remove(zip_file)
        print("download complete!")
    df_raw = pd.read_csv(raw_data_dir, index_col=0)

    for i in range(max(1, level)):
        df_raw["bid_amount_{}".format(i+1)] = \
            df_raw["bids_limit_notional_{}".format(i)]
        df_raw["ask_amount_{}".format(i+1)] = \
            df_raw["asks_limit_notional_{}".format(i)]
        df_raw["bid_price_{}".format(i+1)] = \
            df_raw["midpoint"] + \
            df_raw["midpoint"]*df_raw["bids_distance_{}".format(i)]
        df_raw["ask_price_{}".format(i+1)] = \
            df_raw["midpoint"] + \
            df_raw["midpoint"]*df_raw["asks_distance_{}".format(i)]
    df_raw["time"] = df_raw["system_time"]

    return df_raw


def get_rawLOB_dataset1(
        hyperparam_dict=hyperparam_default,
        seed=0):
    raw_data_path = config.LOB_data_path
    np.random.seed(seed=seed)

    # load raw data and preprocess
    raw_data_dir = "{}sample.csv".format(raw_data_path)
    if not os.path.exists(raw_data_dir):
        print("raw LOB data not found -> dowloading ...")
        zip_file = wget.download(  #TODO: DELETE FOR PUBLIC!!!
            "https://drive.google.com/uc?export=download&id=123IDJxsnkWZ8t4aROqruXZpGs4WwyB9K",
            training_data_path)
        print("extracting zip ...")
        with ZipFile(zip_file, 'r') as zipObj:
            zipObj.extractall(path=training_data_path)
        print("removing zip ...")
        os.remove(zip_file)
        print("download complete!")
    df_raw = pd.read_csv(raw_data_dir)

    return df_raw


# NEW: OPERATOR NJ ODE CODE - CURRENTLY for 1d in space and arbitrary dimension in X
# FIX: include np.meshgrid for spatial grid of higher dimensions
def create_operator_dataset(
        model_name="BrownianCosine", 
        hyperparam_dict=hyperparam_default,
        seed=0):
    """
    create a synthetic dataset using one of the stock-models
    :param model_name: str, name of the synthetic data generating model, see synthetic_datasets.py
    :param hyperparam_dict: dict, contains all needed parameters for the model --> here, also: obs_perc and obs_perc_space if default observation scheme is used # NEW
            it can also contain additional options for dataset generation:
                - masked    None, float or array of floats. if None: no mask is
                            used; if float: lambda of the poisson distribution;
                            if array of floats: gives the bernoulli probability
                            for each coordinate to be observed
                - J_distribution    None, float, or dict. If None: J_i indirectly from uniform         
                                    grid sample, and the J_i spatial points are sampled
                                    on the grid. If float: J_i is Binomial(n,p) with p = 
                                    J_distribution and n=nr space points. If dict: 
                                    specifies a custom distribution for J_i.                    # NOT IMPLEMENTED: J_custom
                - space_distribution    None, dict. If None: spatial points are sampled         # NOT IMPLEMENTED: space_distribution
                                        uniformly on the grid. If dict: specifies a custom
                                        space distribution mu_xi.                               
                - obs_scheme    None or dict. If given: specifies the observation
                                scheme. If none: observation times are Bernoulli
                                with probability obs_perc, inversely calculating
                                the number of observations.
                
            e.g. for BrownianCosine: nb_paths, nb_time_steps, nb_space_steps, maturity, space_limits, space_dimension, dimension, obs_perc, J_obs_perc.
            The above options are to specify masking, J_sampling, mu_xi space distribution, and an observation scheme in time.
    
    :param seed: int, random seed for the generation of the dataset
    :return: str (path where the dataset is saved), int (time_id to identify
                the dataset)
    """
    df_overview, data_overview = get_dataset_overview()

    np.random.seed(seed=seed)
    hyperparam_dict['model_name'] = model_name
    original_desc = json.dumps(hyperparam_dict, sort_keys=True) # dataset configuration
    obs_perc = hyperparam_dict['obs_perc'] # overall time observation percentage
    obs_scheme = None
    if "obs_scheme" in hyperparam_dict:
        obs_scheme = hyperparam_dict["obs_scheme"] 
    masked = False
    # distribution type for J_i
    J_obs_perc = hyperparam_dict['J_obs_perc'] 
    J_binomial = False
    J_custom = False 
    # distribution type for xi
    space_distribution = None # custom mu_xi
    space_custom = False 
    masked_lambda = None 
    mask_probs = None 
    if ("J_distribution" in hyperparam_dict 
        and hyperparam_dict["J_distribution"] not in [None, False]): # NEW
        if isinstance(hyperparam_dict['J_distribution'], float):
            J_binomial = True
            J_obs_perc = hyperparam_dict['J_distribution']
        elif isinstance(hyperparam_dict['J_distribution'], dict):
            # Not yet implemented: Add other distributions for J_i --> handled later in the sampling function "space_sampling"
            J_custom = True
            pass
        else:
            raise ValueError("please provide None (uniform grid p),  a float (poisson lambda) or a dict (custom distribution) "
                             "in hyperparam_dict['J_distribution']")
    if ("space_distribution" in hyperparam_dict
            and hyperparam_dict['space_distribution'] not in [None, False]): # NEW
        # EXTEND: Add other distributions for spatial points --> Add e.g. dependent J_i 
        space_custom = True
        pass
    if ("masked" in hyperparam_dict
            and hyperparam_dict['masked'] not in [None, False]): # gets appropriate mask distribution given input
        masked = True
        if isinstance(hyperparam_dict['masked'], float):
            masked_lambda = hyperparam_dict['masked']
        elif isinstance(hyperparam_dict['masked'], (tuple, list)):
            mask_probs = hyperparam_dict['masked']
            assert len(mask_probs) == hyperparam_dict['dimension']
        else:
            raise ValueError("please provide a float (poisson lambda) "
                             "in hyperparam_dict['masked']")

    model = _OPERATOR_MODELS[model_name](**hyperparam_dict) 
    paths, dt, ds = model.generate_paths() # generate paths [nb_paths, dim, time_points, spatial_grid] 
    size = paths.shape
    # observations in time
    if obs_scheme is None: 
        observed_dates = np.random.random(size=(size[0], size[2])) # [nb_paths, time_points]
        observed_dates = (observed_dates < obs_perc)*1 
        observed_dates[:, 0] = 1 # all paths observed at time 0 --> not required by the ONJODE framework
        nb_obs = np.sum(observed_dates[:, 1:], axis=1) # n_obs is the additional number of observations (n-1 in ONJODE paper) --> [:, 1:] to [:, :] for total nr (n in the ONJODE paper) --> requires also changing n in the loss function
    else: # NOT IMPLEMENTED: Add precise schemes
        pass

    def space_sampling(J):
        if space_custom is False: # default iid uniform on grid
            return np.random.choice(size[3], J, replace=True)
        else:
            space_grid_size = size[3]
            # NOT IMPLEMENTED: Add other distributions over space
            # get a grid over spatial points of the observation percentages at each space point, called space_obs_perc
            # define a space_obs_perc in hyperparam_dict
            # can plot this later to show the observation percentages in the plots
            # Look at DISTRIBUTIONS dict in synthetic_datasets.py for examples
            pass
    
    # Sampling space points
    if J_binomial: # default: binomial for J
        observed_space_points = np.zeros(shape=(size[0], size[2], size[3])) # [path, time_points, spatial_grid]
        nb_obs_space = np.zeros(shape=(size[0], size[2])) # [path, time_points]
        for i in range(size[0]): # each path
            nr_space_samples = np.random.binomial(size[3], J_obs_perc, size=nb_obs[i]+1) 
            observed_date_indices = np.where(observed_dates[i,:]==1)[0]
            for k in range(nb_obs[i]+1): # for each J_i                       
                point_samples_idx = space_sampling(nr_space_samples[k])
                observed_space_points[i, observed_date_indices[k], point_samples_idx] = 1
        nb_obs_space = np.sum(observed_space_points, axis=2) # sum over spatial grid to get J_is [path, time_points]
    elif J_custom: # NOT IMPLEMENTED: Custom J_i distribution
        # make J_obs_perc in dataset_metadata a vector over the space grid 
        pass
    
    # Masking in X
    if masked is False: 
        observed_dates = observed_space_points # simply indicates which space points are observed in space-time
    else: 
        mask = np.zeros(shape=size) # [nb_paths, dim, time_points, spatial_grid]
        for i in range(size[0]): # each path
            for j in range(size[2]): # each time
                if observed_dates[i,j] == 1: # where observed date --> mask at zero not necessarily 1
                    for k in range(size[3]): # each space
                        if observed_space_points[i,j,k] == 1: # where space point observed
                            if masked_lambda is not None: # Poisson
                                amount = min(1+np.random.poisson(masked_lambda), size[1]) 
                                observed = np.random.choice(size[1], amount, replace=False) # randomly select which spatial points are seen
                                mask[i, observed, j, k] = 1
                            elif mask_probs is not None:
                                for m in range(size[1]):
                                    mask[i, m, j, k] = np.random.binomial(1, mask_probs[m]) # each dimension seen with given bernoulli probability
        observed_dates = mask # final mask to apply

    time_id = 1
    if len(df_overview) > 0:
        time_id = np.max(df_overview["id"].values) + 1 #df_overview contains data, name, id, description of params
    file_name = '{}-{}'.format(model_name, time_id)
    path = '{}{}/'.format(training_data_path, file_name) # full path for saving dataset
    hyperparam_dict['dt'] = dt # add dt to the hyperparam dict 
    hyperparam_dict['ds'] = ds # add ds to the hyperparam dict
    space_limits = model.space_limits
    if os.path.exists(path): # path already exists
        print('Path already exists - abort')
        raise ValueError
    df_app = pd.DataFrame(
        data=[[model_name, time_id, original_desc]],
        columns=['name', 'id', 'description']
    )
    df_overview = pd.concat([df_overview, df_app],
                            ignore_index=True) 
    df_overview.to_csv(data_overview) # save updated overview 

    os.makedirs(path)
    with open('{}data.npy'.format(path), 'wb') as f:
        np.save(f, paths) # raw paths [paths, dim, time_points, spatial_grid]
        np.save(f, observed_dates) # mask of dim [paths, dim, time_points, spatial_grid] or if not masked, just observed_space_points dim [paths, time_points, spatial_grid]
        np.save(f, nb_obs) # n-1 (additional observations in time) [paths, ]
        np.save(f, nb_obs_space) # J_is [paths, timesteps] # NEW
    with open('{}metadata.txt'.format(path), 'w') as f:
        json.dump(hyperparam_dict, f, sort_keys=True)

    return path, time_id


# Code for retrieving datasets
def _get_datasetname(time_id):
    df_overview, data_overview = get_dataset_overview()
    vals = df_overview.loc[df_overview["id"] == time_id, "name"].values
    if len(vals) >= 1:
        return vals[0]
    return None


def _get_time_id(stock_model_name="BlackScholes", time_id=None,
                 path=training_data_path):
    """
    if time_id=None, get the time id of the newest dataset with the given name
    :param stock_model_name: str
    :param time_id: None or int
    :return: int, time_id
    """
    if time_id is None:
        df_overview, _ = get_dataset_overview(path)
        df_overview = df_overview.loc[
            df_overview["name"] == stock_model_name]
        if len(df_overview) > 0:
            time_id = np.max(df_overview["id"].values)
        else:
            time_id = None
    return time_id


def _get_dataset_name_id_from_dict(data_dict):
    if isinstance(data_dict, str):
        data_dict = eval("config."+data_dict)
    desc = json.dumps(data_dict, sort_keys=True)
    df_overview, _ = get_dataset_overview()
    which = df_overview.loc[df_overview["description"] == desc].index
    if len(which) == 0:
        raise ValueError(
            "the given dataset does not exist yet, please generate it "
            "first using data_utils.py. \ndata_dict: {}".format(
                data_dict))
    elif len(which) > 1:
        print("WARNING: multiple datasets match the description, returning the "
              "last one. To uniquely identify the wanted dataset, please "
              "provide the dataset_id instead of the data_dict.")
    return list(df_overview.loc[which[-1], ["name", "id"]].values)


def load_metadata(stock_model_name="BlackScholes", time_id=None):
    """
    load the metadata of a dataset specified by its name and id
    :return: dict (with hyperparams of the dataset)
    """
    time_id = _get_time_id(stock_model_name=stock_model_name, time_id=time_id)
    path = '{}{}-{}/'.format(training_data_path, stock_model_name, int(time_id))
    with open('{}metadata.txt'.format(path), 'r') as f:
        hyperparam_dict = json.load(f)
    return hyperparam_dict


def load_dataset(stock_model_name="BlackScholes", time_id=None): 
    """
    load a saved dataset by its name and id
    :param stock_model_name: str, name
    :param time_id: int, id
    :return: np.arrays of stock_paths, observed_dates, number_observations
                dict of hyperparams of the dataset
    """
    time_id = _get_time_id(stock_model_name=stock_model_name, time_id=time_id) # if time_id is None gets latest id
    path = '{}{}-{}/'.format(training_data_path, stock_model_name, int(time_id))

    if stock_model_name == "LOB":
        with open('{}data.npy'.format(path), 'rb') as f:
            samples = np.load(f)
            times = np.load(f)
            eval_samples = np.load(f)
            eval_times = np.load(f)
            eval_labels = np.load(f)
        with open('{}metadata.txt'.format(path), 'r') as f:
            hyperparam_dict = json.load(f)
        return samples, times, eval_samples, eval_times, eval_labels, \
               hyperparam_dict
    elif stock_model_name in _OPERATOR_MODELS.keys(): 
        with open('{}metadata.txt'.format(path), 'r') as f:
            hyperparam_dict = json.load(f)
        with open('{}data.npy'.format(path), 'rb') as f:
            paths = np.load(f)
            observed_dates = np.load(f) 
            nb_obs = np.load(f)
            nb_obs_space = np.load(f)
        return paths, observed_dates, nb_obs, nb_obs_space, hyperparam_dict
    else:
        with open('{}metadata.txt'.format(path), 'r') as f:
            hyperparam_dict = json.load(f)
        with open('{}data.npy'.format(path), 'rb') as f:
            stock_paths = np.load(f)
            observed_dates = np.load(f)
            nb_obs = np.load(f)
            if "obs_noise" in hyperparam_dict:
                obs_noise = np.load(f)
            else:
                obs_noise = None
        return stock_paths, observed_dates, nb_obs, hyperparam_dict, obs_noise



class IrregularDataset(Dataset):
    """
    class for iterating over a dataset
    """
    def __init__(self, model_name, time_id=None, idx=None):
        stock_paths, observed_dates, nb_obs, hyperparam_dict, obs_noise = \
            load_dataset(stock_model_name=model_name, time_id=time_id)
        if idx is None:
            idx = np.arange(hyperparam_dict['nb_paths'])
        self.metadata = hyperparam_dict
        self.stock_paths = stock_paths[idx]
        self.observed_dates = observed_dates[idx]
        self.nb_obs = nb_obs[idx]
        self.obs_noise = obs_noise

    def __len__(self):
        return len(self.nb_obs)

    def __getitem__(self, idx):
        if type(idx) == int:
            idx = [idx]
        if self.obs_noise is None:
            obs_noise = None
        else:
            obs_noise = self.obs_noise[idx]
        # stock_path dimension: [BATCH_SIZE, DIMENSION, time_points]
        return {"idx": idx, "stock_path": self.stock_paths[idx], 
                "observed_dates": self.observed_dates[idx], 
                "nb_obs": self.nb_obs[idx], "dt": self.metadata['dt'],
                "obs_noise": obs_noise}


class LOBDataset(Dataset):
    def __init__(self, time_id, idx=None):
        samples, times, eval_samples, eval_times, eval_labels, \
        hp_dict = load_dataset(
            stock_model_name="LOB", time_id=time_id)
        if idx is None:
            idx = np.arange(len(samples))
        self.metadata = hp_dict
        self.samples = samples[idx]
        self.times = times[idx]
        self.eval_samples = eval_samples[idx]
        self.eval_times = eval_times[idx]
        self.eval_labels = eval_labels[idx]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if type(idx) == int:
            idx = [idx]
        # stock_path dimension: [BATCH_SIZE, DIMENSION, time_points]
        return {"idx": idx, "samples": self.samples[idx],
                "times": self.times[idx],
                "eval_samples": self.eval_samples[idx],
                "eval_times": self.eval_times[idx],
                "eval_labels": self.eval_labels[idx],
                "amount_obs": self.metadata["amount_obs"],
                "eval_predict_steps": self.metadata["eval_predict_steps"],
                "max_pred_step": self.metadata["max_pred_step"],
                "dt": self.metadata['dt']}

class OperatorDataset(Dataset):
    def __init__(self, model_name, time_id, idx=None): # idx for batch paths
        paths, observed_dates, nb_obs, nb_obs_space, hyperparameter_dict = \
            load_dataset(stock_model_name=model_name, time_id=time_id) # observed_dates is complete mask
        if idx is None:
            idx = np.arange(hyperparameter_dict['nb_paths'])
        self.metadata = hyperparameter_dict
        self.paths = paths[idx]
        self.observed_dates = observed_dates[idx]
        self.nb_obs = nb_obs[idx]
        self.nb_obs_space = nb_obs_space[idx] 
        self.obs_noise = None
        self.space_limits = hyperparameter_dict['space_limits']

    def __len__(self):
        return len(self.nb_obs) 

    def __getitem__(self, idx):
        if type(idx) == int:
            idx = [idx] # ensures consistent batch dimension (1, data) instead of (data,) 
        if self.obs_noise is None:
            obs_noise = None
        else:
            obs_noise = self.obs_noise[idx]
        # dict with all batch data. recalling info for path dimension: [BATCH_SIZE, DIMENSION, time_points, space_points] 
        return {"idx": idx, "path": self.paths[idx],
                "observed_dates": self.observed_dates[idx],
                "nb_obs": self.nb_obs[idx], "nb_obs_space": self.nb_obs_space[idx], 
                "dt": self.metadata['dt'], "ds": self.metadata['ds'],
                "obs_noise": obs_noise, "space_limits": self.space_limits} 


def _get_func(name):
    """
    transform a function given as str to a python function
    :param name: str, correspond to a function,
            supported: 'exp', 'power-x' (x the wanted power)
    :return: numpy fuction
    """
    if name in ['exp', 'exponential']:
        return np.exp
    if 'power-' in name:
        x = float(name.split('-')[1])
        def pow(input):
            return np.power(input, x)
        return pow
    else:
        try:
            return eval(name)
        except Exception:
            return None


def _get_X_with_func_appl(X, functions, axis):
    """
    apply a list of functions to the paths in X and append X by the outputs
    along the given axis
    :param X: np.array, with the data,
    :param functions: list of functions to be applied
    :param axis: int, the data_dimension (not batch and not time dim) along
            which the new paths are appended
    :return: np.array
    """
    Y = X
    for f in functions:
        Y = np.concatenate([Y, f(X)], axis=axis)
    return Y


def CustomCollateFnGen(func_names=None):
    """
    a function to get the costume collate function that can be used in
    torch.DataLoader with the wanted functions applied to the data as new
    dimensions
    -> the functions are applied on the fly to the dataset, and this additional
    data doesn't have to be saved

    :param func_names: list of str, with all function names, see _get_func
    :return: collate function, int (multiplication factor of dimension before
                and after applying the functions)
    """
    # get functions that should be applied to X, additionally to identity 
    functions = []
    if func_names is not None:
        for func_name in func_names:
            f = _get_func(func_name)
            if f is not None:
                functions.append(f)
    mult = len(functions) + 1

    def custom_collate_fn(batch):
        dt = batch[0]['dt']
        stock_paths = np.concatenate([b['stock_path'] for b in batch], axis=0)
        observed_dates = np.concatenate([b['observed_dates'] for b in batch],
                                        axis=0)
        obs_noise = None
        if batch[0]["obs_noise"] is not None:
            obs_noise = np.concatenate([b['obs_noise'] for b in batch], axis=0)
        masked = False
        mask = None
        if len(observed_dates.shape) == 3:
            masked = True
            mask = observed_dates
            observed_dates = observed_dates.max(axis=1)
        nb_obs = torch.tensor(
            np.concatenate([b['nb_obs'] for b in batch], axis=0))

        # here axis=1, since we have elements of dim
        #    [batch_size, data_dimension] => add as new data_dimensions
        sp = stock_paths[:, :, 0]
        if obs_noise is not None:
            sp = stock_paths[:, :, 0] + obs_noise[:, :, 0]
        start_X = torch.tensor(
            _get_X_with_func_appl(sp, functions, axis=1),
            dtype=torch.float32)
        X = []
        if masked:
            M = []
            start_M = torch.tensor(mask[:,:,0], dtype=torch.float32).repeat(
                (1,mult))
        else:
            M = None
            start_M = None
        times = []
        time_ptr = [0]
        obs_idx = []
        current_time = 0.
        counter = 0
        for t in range(1, observed_dates.shape[-1]):
            current_time += dt
            if observed_dates[:, t].sum() > 0:
                times.append(current_time)
                for i in range(observed_dates.shape[0]):
                    if observed_dates[i, t] == 1:
                        counter += 1
                        # here axis=0, since only 1 dim (the data_dimension),
                        #    i.e. the batch-dim is cummulated outside together
                        #    with the time dimension
                        sp = stock_paths[i, :, t]
                        if obs_noise is not None:
                            sp = stock_paths[i, :, t] + obs_noise[i, :, t]
                        X.append(_get_X_with_func_appl(sp, functions, axis=0))
                        if masked:
                            M.append(np.tile(mask[i, :, t], reps=mult))
                        obs_idx.append(i)
                time_ptr.append(counter)
        # if obs_noise is not None:
        #     print("noisy observations used")

        assert len(obs_idx) == observed_dates[:, 1:].sum()
        if masked:
            M = torch.tensor(np.array(M), dtype=torch.float32)
        res = {'times': np.array(times), 'time_ptr': np.array(time_ptr),
               'obs_idx': torch.tensor(obs_idx, dtype=torch.long),
               'start_X': start_X, 'n_obs_ot': nb_obs,
               'X': torch.tensor(np.array(X), dtype=torch.float32),
               'true_paths': stock_paths, 'observed_dates': observed_dates,
               'true_mask': mask, 'obs_noise': obs_noise,
               'M': M, 'start_M': start_M}
        return res

    return custom_collate_fn, mult


def LOBCollateFnGen(data_type="train", use_eval_on_train=True,
                    train_classifier=False):
    """
    a function to get the costume collate function that can be used in
    torch.DataLoader with the wanted functions applied to the data as new
    dimensions
    -> the functions are applied on the fly to the dataset, and this additional
    data doesn't have to be saved

    :param data_type: one of {"train", "test"}
    :param use_eval_on_train: bool, whether to use the eval parts of the samples
    :param train_classifier: bool, whether a classifier is trained

    :return: collate function
    """

    def custom_collate_fn(batch):
        amount_obs = batch[0]['amount_obs']
        eval_predict_steps = batch[0]['eval_predict_steps']
        max_pred_step = batch[0]['max_pred_step']
        samples = np.concatenate([b['samples'] for b in batch], axis=0)
        times = np.concatenate([b['times'] for b in batch], axis=0)
        eval_samples = np.concatenate([b['eval_samples'] for b in batch],axis=0)
        eval_times = np.concatenate([b['eval_times'] for b in batch], axis=0)
        eval_labels = np.concatenate([b['eval_labels'] for b in batch], axis=0)

        if data_type == "train":
            if use_eval_on_train and not train_classifier:
                samples = np.concatenate([samples, eval_samples], axis=2)
                times = np.concatenate([times, eval_times], axis=1)
                nb_obs = amount_obs+max_pred_step
                predict_times = None
                predict_vals = None
                predict_labels = None
            else:
                nb_obs = amount_obs
                pred_indices = [x-1 for x in eval_predict_steps]
                predict_times = eval_times[:, pred_indices]
                predict_vals = eval_samples[:, :, pred_indices]
                predict_labels = torch.tensor(eval_labels+1, dtype=torch.long)
            max_time = np.max(times[:, -1])
        else:
            nb_obs = amount_obs
            pred_indices = [x-1 for x in eval_predict_steps]
            predict_times = eval_times[:, pred_indices]
            predict_vals = eval_samples[:, :, pred_indices]
            predict_labels = torch.tensor(eval_labels+1, dtype=torch.long)
            max_time = np.max(predict_times[:, -1])

        start_X = torch.tensor(samples[:, :, 0], dtype=torch.float32)
        X = []
        time_ptr = [0]
        obs_idx = []
        counter = 0
        all_times = []
        curr_times = copy.deepcopy(times[:, 1])
        curr_time_ptr = np.ones_like(curr_times)

        ti = time.time()
        while np.any(curr_time_ptr < nb_obs):
            next_t = np.min(curr_times)
            all_times.append(next_t)
            which = curr_times == next_t
            for i, w in enumerate(which):
                if w == 1:
                    counter += 1
                    X.append(samples[i, :, int(curr_time_ptr[i])])
                    obs_idx.append(i)
                    curr_time_ptr[i] += 1
                    if curr_time_ptr[i] < nb_obs:
                        curr_times[i] = times[i, int(curr_time_ptr[i])]
                    else:
                        curr_times[i] = np.infty
            time_ptr.append(counter)
        # print("collate time: {}".format(time.time()-ti))

        assert len(obs_idx) == (nb_obs-1)*samples.shape[0]
        # predict_labels has values in {0,1,2} (which represent {-1,0,1})
        res = {'times': np.array(all_times), 'time_ptr': np.array(time_ptr),
               'obs_idx': torch.tensor(obs_idx, dtype=torch.long),
               'start_X': start_X,
               'n_obs_ot': torch.ones(size=(start_X.shape[0],))*nb_obs,
               'X': torch.tensor(np.array(X), dtype=torch.float32),
               'true_samples': samples, 'true_times': times,
               'true_eval_samples': eval_samples, 'true_eval_times': eval_times,
               'true_eval_labels': eval_labels,
               'predict_times': predict_times, 'predict_vals': predict_vals,
               'predict_labels': predict_labels,
               'max_time': max_time, 'coord_to_compare': [0],
               'true_mask': None, 'M': None, 'start_M': None}
        return res

    return custom_collate_fn


def LOBCollateFnGen2(): # Simpler than above, for the purpose of just comparing true labels and predicted labels
    """
    a function to get the costume collate function that can be used in
    torch.DataLoader with the wanted functions applied to the data as new
    dimensions
    -> the functions are applied on the fly to the dataset, and this additional
    data doesn't have to be saved

    :return: collate function
    """

    def custom_collate_fn(batch):
        amount_obs = batch[0]['amount_obs']
        eval_predict_steps = batch[0]['eval_predict_steps']
        max_pred_step = batch[0]['max_pred_step']
        samples = np.concatenate([b['samples'] for b in batch], axis=0)
        samples = np.transpose(samples[:, 1:, :], axes=(0, 2, 1))
        samples = np.expand_dims(samples, axis=1)
        times = np.concatenate([b['times'] for b in batch], axis=0)
        eval_samples = np.concatenate([b['eval_samples'] for b in batch],axis=0)
        eval_times = np.concatenate([b['eval_times'] for b in batch], axis=0)
        eval_labels = np.concatenate([b['eval_labels'] for b in batch], axis=0)

        predict_labels = torch.tensor(eval_labels+1, dtype=torch.long)

        res = {'samples': torch.tensor(samples),
               'true_labels': eval_labels,
               'labels': predict_labels,
               }
        return res

    return custom_collate_fn


# NEW: OperatorCollateFnGen function 
def OperatorCollateFnGen(func_names=None): # NEW
    """
    a function to get the costume collate function that can be used in
    torch.DataLoader with the wanted functions applied to the data as new
    dimensions
    -> the functions are applied on the fly to the dataset, and this additional
    data doesn't have to be saved

    :param func_names: list of str, with all function names, see _get_func
    :return: collate function, int (multiplication factor of dimension before
                and after applying the functions)
    """
    # get functions that should be applied to X, additionally to identity -- this is when you want dimensions that are functitons of the  
    functions = []
    if func_names is not None:
        for func_name in func_names:
            f = _get_func(func_name)
            if f is not None:
                functions.append(f)
    # mult = len(functions) + 1 # nr of repeat dimensions
    mult=1 # COMMENT: for operator models, do not consider input-output function applications

    def custom_collate_fn(batch): # collate function for the DataLoader is applied to each batch of data
        # Batches are lists of dicts, each dict has keys as defined in the OperatorDataset class
        dt = batch[0]['dt'] 
        ds = batch[0]['ds'] 
        # collect stock paths and observed dates from each sample in batch
        paths = np.concatenate([b['path'] for b in batch], axis=0) 
        observed_dates = np.concatenate([b['observed_dates'] for b in batch], axis=0) # [batch_size, dim, time_points, spatial_grid] masked or [batch_size, time_points, spatial_grid] not masked
        masked = False
        mask = None
        if len(observed_dates.shape) == 4: # masked
            masked = True
            mask = observed_dates 
            timeshape = observed_dates.shape[2]
            spaceshape = observed_dates.shape[3]
            observed_dates = observed_dates.max(axis=1) # just path, time, space
            observed_times = observed_dates.max(axis=2) # just path, time
        else: # unmasked
            timeshape = observed_dates.shape[1] 
            spaceshape = observed_dates.shape[2]
            observed_times = observed_dates.max(axis=2) # just path, time
        
        nb_obs = np.concatenate([b['nb_obs'] for b in batch], axis=0) # [batch_size,]
        nb_obs_space = np.concatenate([b['nb_obs_space'] for b in batch], axis=0) # [batch_size, time_points]
        space_limits = batch[0]['space_limits'] 
        # NOTE: Currently only implemented for a 1D spatial grid
        left_limit = space_limits[0]

        initial_observed = observed_dates[:, 0, :]
        start = paths[:, :, 0, :] * initial_observed[:, np.newaxis, :] # [batch_size, dim, timesteps+1, gridsteps] 
        # observed start (for synthetic dataset)
        start_X = torch.tensor(
            _get_X_with_func_appl(start, functions, axis=1),
            dtype=torch.float32) # adapted for input-output functions, but these are not used
        # if default: functions = None, the function application and appending never happens, just returns the original start
        if masked:
            M = []
            start_M = torch.tensor(mask[:,:,0,:], dtype=torch.long)#.repeat((1,mult))
        else:
            M = None
            start_M = None

        '''
        Create arrays representing all the information (some are 1d, some are lists of vectors of dimension dX)
            - X: all observed X values ordered in time->batch->space->dimension, where for each time it is [Xij_at_each_batch_over_t, dimension]
                    e.g. (X^(batch=1)_{t1, j1}, X^(batch=1)_{t1, j2}, ..., X^(b=1)_{t1, J(t1)^(b=1)}, X^(b=3)_{t1, j1}, ... , X^(b=3)_{t1, J(t1)^(b=3)}, X^(b=2)_{t2, j1}, ...), where each X is a vector of dimension dX
            - M: the corresponding masks for X if masked
                    e.g. (M^(b=1)_{t1, j1}, M^(batch=1)_{t1, j2}, ..., M^(b=1)_{t1, J(t1)^(b=1)}, M^(b=3)_{t1, j1}, ... , M^(b=3)_{t1, J(t1)^(b=3)}, M^(b=2)_{t2, j1}, ...), where each M is a vector of dimension dX
            - times: the times at which observations are made (in order)
                    e.g. (t1, t2, t3, ...) --> observation times
            - time_ptr: pointer to the indices in X where a new observation time starts
                    e.g. (0, J(t1)^(b=1)+J(t1)^(b=3), previous + J(t2)^(b=2) + ... , ...)
            - time_idx: time indices of times in the timegrid
            - space_points: the space points corresponding to the X^(b)_{ti, j}
                    e.g. (xi^(b=1)_j1^t1, ..., xi^(b=1)_(J(t1)^(b=1))^t1, xi^(b=3)_j1^t1, ..., xi^(b=3)_J(t1)^(b=3)^t1, xi^(b=2)_j1^t2, ...), where each xi is a scalar (1d space)
            - space_idx: space indices of space points in the spacegrid
            - n_space_ot: the number of space points for each batch group through time
                    e.g. (J(t1)^(b=1), J(t1)^(b=3), J(t2)^(b=2), ...)
            - batch_ptr: pointer to the indices in n_space_ot relevant to time ti
                    e.g. (0, 2, ...)
            - obs_idx: the batch index for each observation in X
                    e.g. (1, 3, 2, ...)
            - space_points: the spatial points at which observations are made (in order)
                    e.g. (xi_{t1,j1}, xi_{t1,j2}, ..., xi_{t1,J(t1)^(b=1)}, xi_{t1,j1}, ... , xi_{t1,J(t1)^(b=3)}, xi_{t2,j1}, ...) --> again, each one represents the spatial point for all dimensions of X at that time/space point
            - eval_points: all the spatial points relevant for each batch path (to be used for calculating loss)
                    e.g. ({xi_j^1}_j=1^J_1(b=1), ..., {xi_j^n(b=1)}_j=1^J_n(b=1), ..., {xi_j^1}_j=1^J_1(b=B), ..., {xi_j^n(b=B)}_j=1^J_n(b=B)) --> list of lists, each sublist has all spatial points for that batch path
            - eval_ptr: pointer to the indices in eval_points corresponding to each batch path
                    e.g. (0, J^(b=1), prev + J^(b=2), ..., prev + J^(b=B)) with J^(b) = sum over i=1 to n(b) of J_i^(b)
            - start_X: initial X values for the known conditional expectation model dataset.next_cond_exp
                    Note, these are the complete masked and must be filtered using observed_dates
            - start_M: initial M values for the known conditional expectation model dataset.next_cond_exp
        Initial X and M are stored separately as start_X and start_M
        '''

        # collect the batch-wise unique evaluation points through time
        space_coords = left_limit + ds * np.arange(spaceshape) # spatial grid points
        unique_eval_points = [] # list of lists, each sublist has all spatial points for that batch path
        batch_size = observed_dates.shape[0]

        for b in range(batch_size):
            where_obs = observed_dates[b, :, :].sum(axis=0) > 0 # spatial points observed at any time for batch path b
            eval_idx_b = np.where(where_obs)[0]
            unique_eval_points.append(space_coords[eval_idx_b])
        
        J_b = np.array([len(unique_b) for unique_b in unique_eval_points]) # number of unique spatial points observed for each batch path
        J_tot = J_b.sum() # total number of spatial points observed across all batch paths
        eval_points = np.concatenate(unique_eval_points) # preallocate for speed
        eval_ptr = np.cumsum(np.concatenate(([0], J_b))) # pointer to where each batch path's spatial points start in eval_points

        X = []
        times = [] # observation times ti
        time_ptr = [0] # pointer in X to data correpsonding to obs times
        time_idx = [] # obs time index in grid
        
        space_points = [] # space points for corresponding X data
        space_idx = [] # space indices in grid for corresponding X data
        n_space_ot = [] # J_i for each batch path at each time ti
        batch_ptr = [0] # pointer to which batch path (and J_i) are seen at a certain obs time ti
        obs_idx = [] # stores batch_paths observed at time ti

        counter = 0
        batch_path_counter = 0
        
        time_coords = dt * np.arange(timeshape) # time grid points
        for t in range(0, timeshape): # loop over time steps, including t0
            current_time = time_coords[t]
            if observed_times[:, t].sum() > 0: # if any batch has an observation at the time step
                time_idx.append(t) 
                times.append(current_time)
                for b in range(batch_size): 
                    if observed_times[b, t] == 1: # if batch has an observation at the time step
                        obs_idx.append(b) 
                        batch_path_counter += 1
                        n_space_ot.append(nb_obs_space[b, t]) # store J_i
                        for j in range(spaceshape): # add points over space
                            current_space = space_coords[j]
                            if observed_dates[b, t, j] == 1: # if spatial point j is observed at time t for batch b
                                counter += 1
                                space_idx.append(j) # store indices of space points corresponding to X data
                                space_points.append(current_space) 
                                
                                X_vals = paths[b, :, t, j]
                                X.append(_get_X_with_func_appl(X_vals, functions, axis=0)) # no i/o functionality, this will just return X_vals
                                if masked: # if masked
                                    M.append(np.tile(mask[b, :, t, j], reps=mult)) # no i/o functionality, this will just return mask values for each X data point
                batch_ptr.append(batch_path_counter)
                time_ptr.append(counter)

        assert len(obs_idx) == observed_dates[:, :, :].max(axis=2).sum(axis=0).sum() # now including t0
        
        if masked:
            M = torch.tensor(np.array(M), dtype=torch.float32)
        res = {'times': np.array(times), 'time_ptr': np.array(time_ptr), 
               'time_idx': np.array(time_idx, dtype='int'), 'space_points': torch.tensor(np.array(space_points), dtype=torch.float32),
               'n_space_ot': torch.tensor(np.array(n_space_ot), dtype=torch.long), 'batch_ptr': np.array(batch_ptr), 'space_idx': np.array(space_idx, dtype='int'),
               'eval_points': torch.tensor(np.array(eval_points), dtype=torch.float32), 'eval_ptr': np.array(eval_ptr),
               'space_coords': torch.tensor(space_coords, dtype=torch.float32), # needed for evaluations 
               'obs_idx': obs_idx, 
               'n_obs_ot': torch.tensor(nb_obs, dtype=torch.long),
               'X': torch.tensor(np.array(X), dtype=torch.float32),
               'true_paths': paths, 'observed_dates': observed_dates,
               'true_mask': mask, 'start_X': start_X, 
               'start_M': start_M,
               'M': M, } 
        return res

    return custom_collate_fn, mult


def main(arg):
    """
    function to generate datasets
    """
    del arg
    if FLAGS.dataset_name:
        dataset_name = FLAGS.dataset_name
        print('dataset_name: {}'.format(dataset_name))
    else:
        raise ValueError("Please provide --dataset_name")
    if FLAGS.dataset_params:
        dataset_params = eval("config."+FLAGS.dataset_params)
        print('dataset_params: {}'.format(dataset_params))
    else:
        raise ValueError("Please provide --dataset_params")
    if "combined_" in dataset_name:
        smn = dataset_name.split("_")[1:]
        create_combined_dataset(
            stock_model_names=smn, hyperparam_dicts=dataset_params,
            seed=FLAGS.seed)
    elif "LOB" in dataset_name:
        create_LOB_dataset(hyperparam_dict=dataset_params, seed=FLAGS.seed)
    elif dataset_name in _OPERATOR_MODELS.keys(): 
        create_operator_dataset(
            model_name=dataset_name, hyperparam_dict=dataset_params,
            seed=FLAGS.seed)
    else:
        create_dataset(
            stock_model_name=dataset_name, hyperparam_dict=dataset_params,
            seed=FLAGS.seed)







if __name__ == '__main__':
    app.run(main)



    pass

