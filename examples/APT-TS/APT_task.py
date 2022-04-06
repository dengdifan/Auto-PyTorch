import numpy as np

from autoPyTorch.api.time_series_forecasting import TimeSeriesForecastingTask
import pandas as pds
from datetime import datetime
import warnings
import os
import copy
from pathlib import Path

import argparse

import csv
import shutil

from autoPyTorch.datasets.resampling_strategy import (
    CrossValTypes,
    HoldoutValTypes,
)

import data_loader
from constant import VALUE_COL_NAME, TIME_COL_NAME, SEASONALITY_MAP, FREQUENCY_MAP, DATASETS


def compute_loss(forecast_horizon, seasonality, final_forecasts, test_series_list, train_series_list):
    epsilon = 0.1

    MASE = []
    sMAPE = []
    msMAPE = []
    MAE = []
    RMSE = []

    sqrt_forecast_horizon = np.sqrt(forecast_horizon)

    idx = 0

    for f, y, y_data in zip(final_forecasts, test_series_list, train_series_list):

        M = len(y_data)

        diff_abs = np.abs(f - y)

        if M == seasonality:
            mase_denominator = 0
        else:
            mase_denominator_coefficient = forecast_horizon / (M - seasonality)
            mase_denominator = mase_denominator_coefficient * \
                               np.sum(np.abs(y_data[seasonality:] - y_data[:-seasonality]))

            abs_loss = np.sum(diff_abs)
            mase = abs_loss / mase_denominator

        if mase_denominator == 0:
            mase_denominator_coefficient = forecast_horizon / (M - 1)
            mase_denominator = mase_denominator_coefficient * \
                               np.sum(np.abs(y_data[1:] - y_data[:-1]))
            mase = abs_loss / mase_denominator

        if np.isnan(mase) or np.isinf(mase):
            # see the R file
            pass
        else:
            MASE.append(mase)

        smape = 2 * diff_abs / (np.abs(y) + np.abs(f))
        smape[diff_abs == 0] = 0
        smape = np.sum(smape) / forecast_horizon
        sMAPE.append(smape)

        msmape = np.sum(2 * diff_abs / (np.maximum(np.abs(y) + np.abs(f) + epsilon, epsilon + 0.5))) / forecast_horizon
        msMAPE.append(msmape)

        mae = abs_loss / forecast_horizon
        MAE.append(mae)

        rmse = np.linalg.norm(f - y) / sqrt_forecast_horizon
        RMSE.append(rmse)


        idx += 1
    res = {}

    res['Mean MASE'] = np.mean(MASE)

    res['Median MASE'] = np.median(MASE)

    res['Mean sMAPE'] = np.mean(sMAPE)
    res['Median sMAPE'] = np.median(sMAPE)

    res['Mean mSMAPE'] = np.mean(msMAPE)
    res['Median mSMAPE'] = np.median(msMAPE)

    res['Mean MAE'] = np.mean(MAE)
    res['Median MAE'] = np.median(MAE)

    res['Mean RMSE'] = np.mean(RMSE)
    res['Median RMSE'] = np.median(RMSE)


    return res


def main(working_dir="/home/$USER/tmp/tmp",
         dataset_name='nn5_daily',
         budget_type='dataset_size',
         res_dir="/home/$USER//tmp/tsf_res",
         validation='holdout',
         seed=1):
    file_name, external_forecast_horizon, integer_conversion = DATASETS[dataset_name]

    dataset_path = Path(working_dir) / "tsf_data" / dataset_name / file_name

    df, frequency, forecast_horizon, contain_missing_values, contain_equal_length = \
        data_loader.convert_tsf_to_dataframe(str(dataset_path))

    # If the forecast horizon is not given within the .tsf file, then it should be provided as a function input
    if forecast_horizon is None:
        if external_forecast_horizon is None:
            raise Exception("Please provide the required prediction steps")
        else:
            forecast_horizon = external_forecast_horizon

    train_series_list = []
    test_series_list = []

    X_train = []
    y_train = []

    X_test = []
    y_test = []

    if frequency is not None:
        freq = FREQUENCY_MAP[frequency]
        seasonality = SEASONALITY_MAP[frequency]
    else:
        freq = "1Y"
        seasonality = 1

    shortest_sequence = np.inf
    train_start_time_list = []

    for index, row in df.iterrows():
        if TIME_COL_NAME in df.columns:
            train_start_time = row[TIME_COL_NAME]
        else:
            train_start_time = datetime.strptime('1900-01-01 00-00-00',
                                                 '%Y-%m-%d %H-%M-%S')  # Adding a dummy timestamp, if the timestamps are not available in the dataset or consider_time is False
        train_start_time_list.append(pds.Timestamp(train_start_time, freq=freq))

        series_data = row[VALUE_COL_NAME].to_numpy()
        # Creating training and test series. Test series will be only used during evaluation
        train_series_data = series_data[:len(series_data) - forecast_horizon]
        test_series_data = series_data[(len(series_data) - forecast_horizon): len(series_data)]

        y_test.append(series_data[-forecast_horizon:])

        train_series_list.append(train_series_data)
        test_series_list.append(test_series_data)

        shortest_sequence = min(len(train_series_data), shortest_sequence)

    if validation == 'cv':
        n_splits = 3
        while shortest_sequence - forecast_horizon - forecast_horizon * n_splits <= 0:
            n_splits -= 1

        if n_splits >= 2:
            resampling_strategy = CrossValTypes.time_series_cross_validation
            resampling_strategy_args = {'num_splits': n_splits}

        else:
            warnings.warn('The dataset is not suitable for cross validation, we will try holdout instead')
            validation = 'holdout'
    elif validation == 'holdout_ts':
        resampling_strategy = CrossValTypes.time_series_ts_cross_validation
        resampling_strategy_args = None
    if validation == 'holdout':
        resampling_strategy = HoldoutValTypes.time_series_hold_out_validation
        resampling_strategy_args = None


    X_train = copy.deepcopy(train_series_list)
    y_train = copy.deepcopy(train_series_list)

    X_test = copy.deepcopy(X_train)

    path = Path(working_dir) / 'APT_run'
    path_log = str(path / dataset_name / budget_type / f'{seed}' / "log")
    path_pred = str(path / dataset_name / budget_type / f'{seed}' / "output")

    # Remove intermediate files
    try:
        shutil.rmtree(path_log)
        shutil.rmtree(path_pred)
    except OSError as e:
        print("Error: %s - %s." % (e.filename, e.strerror))

    smac_source_dir = Path(path_log) / "smac3-output"


    api = TimeSeriesForecastingTask(
        #delete_tmp_folder_after_terminate=False,
        #delete_output_folder_after_terminate=False,
        seed=seed,
        ensemble_size=20,
        resampling_strategy=resampling_strategy,
        resampling_strategy_args=resampling_strategy_args,
        temporary_directory=path_log,
        output_directory=path_pred,
    )

    api.set_pipeline_config(device="cuda",
                            torch_num_threads=8,
                            early_stopping=20)
    if budget_type == "random_search":
        budget_kwargs = {'budget_type': 'random_search',
                         'max_budget': None,
                         'min_budget': None}

    elif budget_type != 'full_budget':
        from autoPyTorch.constants_forecasting import FORECASTING_BUDGET_TYPE
        if budget_type not in FORECASTING_BUDGET_TYPE and budget_type != 'epochs':
            raise NotImplementedError('Unknown Budget Type!')
        budget_kwargs = {'budget_type': budget_type,
                         'max_budget': 50 if budget_type == 'epochs' else 1.0,
                         'min_budget': 5 if budget_type == 'epochs' else 0.1}
    else:
        budget_kwargs = {'budget_type': 'epochs',
                         'max_budget': 50,
                         'min_budget': 50}

    api.search(
        X_train=None,
        y_train=copy.deepcopy(y_train),
        optimize_metric='mean_MASE_forecasting',
        n_prediction_steps=forecast_horizon,
        **budget_kwargs,
        freq=freq,
        start_times_train=train_start_time_list,
        memory_limit=32 * 1024,
        normalize_y=False,
        total_walltime_limit=3600 * 10,
        min_num_test_instances=1000,
    )

    from autoPyTorch.data.time_series_forecasting_validator import TimeSeriesForecastingInputValidator

    res_dir = Path(res_dir)

    res_dir_task = res_dir / dataset_name / budget_type / f'{seed}'

    smac_res_path = res_dir_task / 'smac3-output'

    if not os.path.exists(str(res_dir_task)):
        os.makedirs(str(res_dir_task))

    try:
        shutil.rmtree(smac_res_path)
    except OSError as e:
        print("Error: %s - %s." % (e.filename, e.strerror))

    try:
        shutil.copytree(str(smac_source_dir), smac_res_path)
    except OSError as e:  # python >2.5
        print("Error: %s - %s." % (e.filename, e.strerror))


    refit_dataset = api.dataset.create_refit_set()

    train_pred_seq = []
    test_sets = api.dataset.generatet_test_seqs()

    try:
        api.refit(refit_dataset, 0)

        pred = api.predict(test_sets)

    except Exception as e:
        print(e)
        exit()


    if integer_conversion:
        final_forecasts = np.round(pred)
    else:
        final_forecasts = pred

    if frequency is not None:
        freq = FREQUENCY_MAP[frequency]
        seasonality = SEASONALITY_MAP[frequency]
    else:
        freq = "1Y"
        seasonality = 1

    if isinstance(seasonality, list):
        seasonality = min(seasonality)  # Use to calculate MASE
    seasonality = int(seasonality)

    res = compute_loss(forecast_horizon, seasonality, pred_val, y_test, train_series_data)


    # write the forecasting results to a file
    forecast_file_path = res_dir_task / f"{dataset_name}_{budget_type}_results.txt"

    with open(forecast_file_path, "w") as output:
        writer = csv.writer(output, lineterminator='\n')
        writer.writerows(final_forecasts)

    # Write training dataset and the actual results into separate files, which are then used for error calculations
    # We do not use the built-in evaluation method in GluonTS as some of the error measures we use are not implemented in that
    temp_dataset_path = res_dir_task / f"{dataset_name}_dataset.txt"
    temp_results_path = res_dir_task / f"{dataset_name}_ground_truth.txt"

    # with open(str(temp_dataset_path), "w") as output_dataset:
    #    writer = csv.writer(output_dataset, lineterminator='\n')
    #    writer.writerows(train_series_list)

    with open(str(temp_results_path), "w") as output_results:
        writer = csv.writer(output_results, lineterminator='\n')
        writer.writerows(test_series_list)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='APT_Task')
    parser.add_argument('--dataset_name', type=str, default="m3_yearly", help='dataset name')
    parser.add_argument("--budget_type", default="epochs", type=str, help='budget type')
    parser.add_argument("--working_dir", default="/home/$USER/tmp", type=str,
                        help="directory where datasets and tmp files are stored")
    parser.add_argument('--validation', type=str, default="holdout", help='type of validation')
    parser.add_argument('--seed', type=int, default="10", help='random seed')

    args = parser.parse_args()

    dataset_name = args.dataset_name
    budget_type = args.budget_type
    working_dir = args.working_dir
    validation = args.validation
    seed = args.seed

    main(working_dir=working_dir, dataset_name=dataset_name, budget_type=budget_type, validation=validation, seed=seed)
