import copy
import json
import logging.handlers
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import ConfigSpace
from ConfigSpace.configuration_space import Configuration

import dask.distributed

from smac.facade.smac_ac_facade import SMAC4AC
from smac.intensification.hyperband import Hyperband
from smac.runhistory.runhistory import RunHistory
from smac.runhistory.runhistory2epm import RunHistory2EPM4LogCost
from smac.scenario.scenario import Scenario
from smac.tae.dask_runner import DaskParallelRunner
from smac.tae.serial_runner import SerialRunner
from smac.utils.io.traj_logging import TrajEntry

from autoPyTorch.automl_common.common.utils.backend import Backend
from autoPyTorch.datasets.base_dataset import BaseDataset
from autoPyTorch.datasets.resampling_strategy import (
    CrossValTypes,
    DEFAULT_RESAMPLING_PARAMETERS,
    HoldoutValTypes,
)
from autoPyTorch.ensemble.ensemble_builder import EnsembleBuilderManager
from autoPyTorch.evaluation.tae import ExecuteTaFuncWithQueue, get_cost_of_crash
from autoPyTorch.evaluation.time_series_forecasting_train_evaluator import TimeSeriesForecastingTrainEvaluator
from autoPyTorch.optimizer.utils import read_return_initial_configurations

from autoPyTorch.pipeline.components.training.metrics.base import autoPyTorchMetric
from autoPyTorch.utils.hyperparameter_search_space_update import HyperparameterSearchSpaceUpdates
from autoPyTorch.utils.logging_ import get_named_client_logger
from autoPyTorch.utils.stopwatch import StopWatch


def get_smac_object(
    scenario_dict: Dict[str, Any],
    seed: int,
    ta: Callable,
    ta_kwargs: Dict[str, Any],
    n_jobs: int,
    initial_budget: int,
    max_budget: int,
    dask_client: Optional[dask.distributed.Client],
    initial_configurations: Optional[List[Configuration]] = None,
) -> SMAC4AC:
    """
    This function returns an SMAC object that is gonna be used as
    optimizer of pipelines

    Args:
        scenario_dict (Dict[str, Any]): constrain on how to run
            the jobs
        seed (int): to make the job deterministic
        ta (Callable): the function to be intensifier by smac
        ta_kwargs (Dict[str, Any]): Arguments to the above ta
        n_jobs (int): Amount of cores to use for this task
        dask_client (dask.distributed.Client): User provided scheduler
        initial_configurations (List[Configuration]): List of initial
            configurations which smac will run before starting the search process

    Returns:
        (SMAC4AC): sequential model algorithm configuration object

    """
    intensifier = Hyperband

    rh2EPM = RunHistory2EPM4LogCost
    return SMAC4AC(
        scenario=Scenario(scenario_dict),
        rng=seed,
        runhistory2epm=rh2EPM,
        tae_runner=ta,
        tae_runner_kwargs=ta_kwargs,
        initial_configurations=initial_configurations,
        run_id=seed,
        intensifier=intensifier,
        intensifier_kwargs={'initial_budget': initial_budget, 'max_budget': max_budget,
                            'eta': 3, 'min_chall': 1, 'instance_order': 'shuffle_once'},
        dask_client=dask_client,
        n_jobs=n_jobs,
    )


class AutoMLSMBO(object):
    def __init__(self,
                 config_space: ConfigSpace.ConfigurationSpace,
                 dataset_name: str,
                 backend: Backend,
                 total_walltime_limit: float,
                 func_eval_time_limit_secs: float,
                 memory_limit: Optional[int],
                 metric: autoPyTorchMetric,
                 watcher: StopWatch,
                 n_jobs: int,
                 dask_client: Optional[dask.distributed.Client],
                 pipeline_config: Dict[str, Any],
                 start_num_run: int = 1,
                 seed: int = 1,
                 resampling_strategy: Union[HoldoutValTypes, CrossValTypes] = HoldoutValTypes.holdout_validation,
                 resampling_strategy_args: Optional[Dict[str, Any]] = None,
                 include: Optional[Dict[str, Any]] = None,
                 exclude: Optional[Dict[str, Any]] = None,
                 disable_file_output: List = [],
                 smac_scenario_args: Optional[Dict[str, Any]] = None,
                 get_smac_object_callback: Optional[Callable] = None,
                 all_supported_metrics: bool = True,
                 ensemble_callback: Optional[EnsembleBuilderManager] = None,
                 logger_port: Optional[int] = None,
                 search_space_updates: Optional[HyperparameterSearchSpaceUpdates] = None,
                 portfolio_selection: Optional[str] = None,
                 pynisher_context: str = 'spawn',
                 min_budget: int = 5,
                 max_budget: int = 50,
                 time_series_forecasting: bool = False
                 ):
        """
        Interface to SMAC. This method calls the SMAC optimize method, and allows
        to pass a callback (ensemble_callback) to make launch task at the end of each
        optimize() algorithm. The later is needed due to the nature of blocking long running
        tasks in Dask.

        Args:
            config_space (ConfigSpace.ConfigurationSpac):
                The configuration space of the whole process
            dataset_name (str):
                The name of the dataset, used to identify the current job
            backend (Backend):
                An interface with disk
            total_walltime_limit (float):
                The maximum allowed time for this job
            func_eval_time_limit_secs (float):
                How much each individual task is allowed to last
            memory_limit (Optional[int]):
                Maximum allowed CPU memory this task can use
            metric (autoPyTorchMetric):
                An scorer object to evaluate the performance of each jon
            watcher (StopWatch):
                A stopwatch object to debug time consumption
            n_jobs (int):
                How many workers are allowed in each task
            dask_client (Optional[dask.distributed.Client]):
                An user provided scheduler. Else smac will create its own.
            start_num_run (int):
                The ID index to start runs
            seed (int):
                To make the run deterministic
            resampling_strategy (str):
                What strategy to use for performance validation
            resampling_strategy_args (Optional[Dict[str, Any]]):
                Arguments to the resampling strategy -- like number of folds
            include (Optional[Dict[str, Any]] = None):
                Optimal Configuration space modifiers
            exclude (Optional[Dict[str, Any]] = None):
                Optimal Configuration space modifiers
            disable_file_output List:
                Support to disable file output to disk -- to reduce space
            smac_scenario_args (Optional[Dict[str, Any]]):
                Additional arguments to the smac scenario
            get_smac_object_callback (Optional[Callable]):
                Allows to create a user specified SMAC object
            pynisher_context (str):
                A string indicating the multiprocessing context to use
            ensemble_callback (Optional[EnsembleBuilderManager]):
                A callback used in this scenario to start ensemble building subtasks
            portfolio_selection (Optional[str]):
                This argument controls the initial configurations that
                AutoPyTorch uses to warm start SMAC for hyperparameter
                optimization. By default, no warm-starting happens.
                The user can provide a path to a json file containing
                configurations, similar to (autoPyTorch/configs/greedy_portfolio.json).
                Additionally, the keyword 'greedy' is supported,
                which would use the default portfolio from
                `AutoPyTorch Tabular <https://arxiv.org/abs/2006.13799>_`
            min_budget (int):
                Auto-PyTorch uses `Hyperband <https://arxiv.org/abs/1603.06560>_` to
                trade-off resources between running many pipelines at min_budget and
                running the top performing pipelines on max_budget.
                min_budget states the minimum resource allocation a pipeline should have
                so that we can compare and quickly discard bad performing models.
                For example, if the budget_type is epochs, and min_budget=5, then we will
                run every pipeline to a minimum of 5 epochs before performance comparison.
            max_budget (int):
                Auto-PyTorch uses `Hyperband <https://arxiv.org/abs/1603.06560>_` to
                trade-off resources between running many pipelines at min_budget and
                running the top performing pipelines on max_budget.
                max_budget states the maximum resource allocation a pipeline is going to
                be ran. For example, if the budget_type is epochs, and max_budget=50,
                then the pipeline training will be terminated after 50 epochs.
            time_series_prediction (bool):
                If we want to apply this optimizer to optimize time series prediction tasks (which has a different
                tae)
        """
        super(AutoMLSMBO, self).__init__()
        # data related
        self.dataset_name = dataset_name
        self.datamanager: Optional[BaseDataset] = None
        self.metric = metric
        self.task: Optional[str] = None
        self.backend = backend
        self.all_supported_metrics = all_supported_metrics

        self.pipeline_config = pipeline_config
        # the configuration space
        self.config_space = config_space

        # the number of parallel workers/jobs
        self.n_jobs = n_jobs
        self.dask_client = dask_client

        # Evaluation
        self.resampling_strategy = resampling_strategy
        if resampling_strategy_args is None:
            resampling_strategy_args = DEFAULT_RESAMPLING_PARAMETERS[resampling_strategy]
        self.resampling_strategy_args = resampling_strategy_args

        # and a bunch of useful limits
        self.worst_possible_result = get_cost_of_crash(self.metric)
        self.total_walltime_limit = int(total_walltime_limit)
        self.func_eval_time_limit_secs = int(func_eval_time_limit_secs)
        self.memory_limit = memory_limit
        self.watcher = watcher
        self.seed = seed
        self.start_num_run = start_num_run
        self.include = include
        self.exclude = exclude
        self.disable_file_output = disable_file_output
        self.smac_scenario_args = smac_scenario_args
        self.get_smac_object_callback = get_smac_object_callback
        self.pynisher_context = pynisher_context
        self.min_budget = min_budget
        self.max_budget = max_budget

        self.ensemble_callback = ensemble_callback

        self.search_space_updates = search_space_updates

        self.time_series_prediction = time_series_prediction

        if logger_port is None:
            self.logger_port = logging.handlers.DEFAULT_TCP_LOGGING_PORT
        else:
            self.logger_port = logger_port
        logger_name = '%s(%d):%s' % (self.__class__.__name__, self.seed, ":" + self.dataset_name)
        self.logger = get_named_client_logger(name=logger_name,
                                              port=self.logger_port)
        self.logger.info("initialised {}".format(self.__class__.__name__))

        self.initial_configurations: Optional[List[Configuration]] = None
        if portfolio_selection is not None:
            self.initial_configurations = read_return_initial_configurations(config_space=config_space,
                                                                             portfolio_selection=portfolio_selection)

    def reset_data_manager(self) -> None:
        if self.datamanager is not None:
            del self.datamanager
        self.datamanager = self.backend.load_datamanager()

        if self.datamanager is not None and self.datamanager.task_type is not None:
            self.task = self.datamanager.task_type

    def run_smbo(self, func: Optional[Callable] = None
                 ) -> Tuple[RunHistory, List[TrajEntry], str]:

        self.watcher.start_task('SMBO')
        self.logger.info("Started run of SMBO")
        # == first things first: load the datamanager
        self.reset_data_manager()

        # == Initialize non-SMBO stuff
        # first create a scenario
        seed = self.seed
        self.config_space.seed(seed)
        # allocate a run history
        num_run = self.start_num_run

        # Initialize some SMAC dependencies

        if isinstance(self.resampling_strategy, CrossValTypes):
            num_splits = self.resampling_strategy_args['num_splits']
            instances = [[json.dumps({'task_id': self.dataset_name,
                                      'fold': fold_number})]
                         for fold_number in range(num_splits)]
        else:
            instances = [[json.dumps({'task_id': self.dataset_name})]]

        # TODO rebuild target algorithm to be it's own target algorithm
        # evaluator, which takes into account that a run can be killed prior
        # to the model being fully fitted; thus putting intermediate results
        # into a queue and querying them once the time is over
        ta_kwargs = dict(
            backend=copy.deepcopy(self.backend),
            seed=seed,
            initial_num_run=num_run,
            include=self.include if self.include is not None else dict(),
            exclude=self.exclude if self.exclude is not None else dict(),
            metric=self.metric,
            memory_limit=self.memory_limit,
            disable_file_output=self.disable_file_output,
            ta=func,
            logger_port=self.logger_port,
            all_supported_metrics=self.all_supported_metrics,
            pipeline_config=self.pipeline_config,
            search_space_updates=self.search_space_updates,
            pynisher_context=self.pynisher_context,
        )

        if self.time_series_prediction:
            ta_kwargs["evaluator_class"] = TimeSeriesForecastingTrainEvaluator
        ta = ExecuteTaFuncWithQueue
        self.logger.info("Finish creating Target Algorithm (TA) function")

        startup_time = self.watcher.wall_elapsed(self.dataset_name)
        total_walltime_limit = self.total_walltime_limit - startup_time - 5
        scenario_dict = {
            'abort_on_first_run_crash': False,
            'cs': self.config_space,
            'cutoff_time': self.func_eval_time_limit_secs,
            'deterministic': 'true',
            'instances': instances,
            'memory_limit': self.memory_limit,
            'output-dir': self.backend.get_smac_output_directory(),
            'run_obj': 'quality',
            'wallclock_limit': total_walltime_limit,
            'cost_for_crash': self.worst_possible_result,
        }
        if self.smac_scenario_args is not None:
            for arg in [
                'abort_on_first_run_crash',
                'cs',
                'deterministic',
                'instances',
                'output-dir',
                'run_obj',
                'shared-model',
                'cost_for_crash',
            ]:
                if arg in self.smac_scenario_args:
                    self.logger.warning('Cannot override scenario argument %s, '
                                        'will ignore this.', arg)
                    del self.smac_scenario_args[arg]
            for arg in [
                'cutoff_time',
                'memory_limit',
                'wallclock_limit',
            ]:
                if arg in self.smac_scenario_args:
                    self.logger.warning(
                        'Overriding scenario argument %s: %s with value %s',
                        arg,
                        scenario_dict[arg],
                        self.smac_scenario_args[arg]
                    )
            scenario_dict.update(self.smac_scenario_args)

        budget_type = self.pipeline_config['budget_type']
        if budget_type == 'epochs':
            initial_budget = self.pipeline_config['min_epochs']
            max_budget = self.pipeline_config['epochs']
        elif budget_type == 'resolution':
            initial_budget = self.pipeline_config.get('min_resolution', 0.1)
            max_budget = self.pipeline_config.get('full_resolution', 1.0)
        else:
            raise ValueError("Illegal value for budget type, must be one of "
                             "('epochs', 'runtime'), but is : %s" %
                             budget_type)

        if self.get_smac_object_callback is not None:
            smac = self.get_smac_object_callback(scenario_dict=scenario_dict,
                                                 seed=seed,
                                                 ta=ta,
                                                 ta_kwargs=ta_kwargs,
                                                 n_jobs=self.n_jobs,
                                                 initial_budget=self.min_budget,
                                                 max_budget=self.max_budget,
                                                 dask_client=self.dask_client,
                                                 initial_configurations=self.initial_configurations)
        else:
            smac = get_smac_object(scenario_dict=scenario_dict,
                                   seed=seed,
                                   ta=ta,
                                   ta_kwargs=ta_kwargs,
                                   n_jobs=self.n_jobs,
                                   initial_budget=self.min_budget,
                                   max_budget=self.max_budget,
                                   dask_client=self.dask_client,
                                   initial_configurations=self.initial_configurations)

        if self.ensemble_callback is not None:
            smac.register_callback(self.ensemble_callback)

        self.logger.info("initialised SMBO, running SMBO.optimize()")

        smac.optimize()

        self.logger.info("finished SMBO.optimize()")

        self.runhistory = smac.solver.runhistory
        self.trajectory = smac.solver.intensifier.traj_logger.trajectory
        if isinstance(smac.solver.tae_runner, DaskParallelRunner):
            self._budget_type = smac.solver.tae_runner.single_worker.budget_type
        elif isinstance(smac.solver.tae_runner, SerialRunner):
            self._budget_type = smac.solver.tae_runner.budget_type
        else:
            raise NotImplementedError(type(smac.solver.tae_runner))

        return self.runhistory, self.trajectory, self._budget_type
