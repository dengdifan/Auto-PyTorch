TABULAR_CLASSIFICATION = 1
IMAGE_CLASSIFICATION = 2
TABULAR_REGRESSION = 3
IMAGE_REGRESSION = 4
TIMESERIES_FORECASTING = 5

REGRESSION_TASKS = [TABULAR_REGRESSION, IMAGE_REGRESSION]
CLASSIFICATION_TASKS = [TABULAR_CLASSIFICATION, IMAGE_CLASSIFICATION]
FORECASTING_TASKS = [TIMESERIES_FORECASTING]  # TODO extend FORECASTING TASKS to Classification and regression tasks

TABULAR_TASKS = [TABULAR_CLASSIFICATION, TABULAR_REGRESSION]
IMAGE_TASKS = [IMAGE_CLASSIFICATION, IMAGE_REGRESSION]
TIMESERIES_TASKS = [FORECASTING_TASKS]

TASK_TYPES = REGRESSION_TASKS + CLASSIFICATION_TASKS + FORECASTING_TASKS

TASK_TYPES_TO_STRING = \
    {TABULAR_CLASSIFICATION: 'tabular_classification',
     IMAGE_CLASSIFICATION: 'image_classification',
     TABULAR_REGRESSION: 'tabular_regression',
     IMAGE_REGRESSION: 'image_regression',
     TIMESERIES_FORECASTING: 'time_series_forecasting'}

STRING_TO_TASK_TYPES = \
    {'tabular_classification': TABULAR_CLASSIFICATION,
     'image_classification': IMAGE_CLASSIFICATION,
     'tabular_regression': TABULAR_REGRESSION,
     'image_regression': IMAGE_REGRESSION,
     'time_series_forecasting': TIMESERIES_FORECASTING}

# Output types have been defined as in scikit-learn type_of_target
# (https://scikit-learn.org/stable/modules/generated/sklearn.utils.multiclass.type_of_target.html)
BINARY = 10
CONTINUOUSMULTIOUTPUT = 11
MULTICLASS = 12
CONTINUOUS = 13
MULTICLASSMULTIOUTPUT = 14

OUTPUT_TYPES = [BINARY, CONTINUOUSMULTIOUTPUT, MULTICLASS, CONTINUOUS]

OUTPUT_TYPES_TO_STRING = \
    {BINARY: 'binary',
     CONTINUOUSMULTIOUTPUT: 'continuous-multioutput',
     MULTICLASS: 'multiclass',
     CONTINUOUS: 'continuous',
     MULTICLASSMULTIOUTPUT: 'multiclass-multioutput'}

STRING_TO_OUTPUT_TYPES = \
    {'binary': BINARY,
     'continuous-multioutput': CONTINUOUSMULTIOUTPUT,
     'multiclass': MULTICLASS,
     'continuous': CONTINUOUS,
     'multiclass-multioutput': MULTICLASSMULTIOUTPUT}

CLASSIFICATION_OUTPUTS = [BINARY, MULTICLASS, MULTICLASSMULTIOUTPUT]
REGRESSION_OUTPUTS = [CONTINUOUS, CONTINUOUSMULTIOUTPUT]
