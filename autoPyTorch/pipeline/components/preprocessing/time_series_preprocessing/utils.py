from typing import Any, Dict, List

from sklearn.base import BaseEstimator


def get_time_series_preprocessers(X: Dict[str, Any]) -> Dict[str, List[BaseEstimator]]:
    """
    Expects fit_dictionary(X) to have numerical/categorical preprocessors
    (fitted numerical/categorical preprocessing nodes) that will build a pipeline in the TimeSeriesTransformer.
    This function parses X and extracts such components.
    Creates a dictionary with two keys,
    numerical- containing list of numerical preprocessors
    categorical- containing list of categorical preprocessors
    Args:
        X: fit dictionary
    Returns:
        (Dict[str, List[BaseEstimator]]): dictionary with list of numerical and categorical preprocessors
    """
    preprocessor = dict(numerical=list(), categorical=list())  # type: Dict[str, List[BaseEstimator]]
    for key, value in X.items():
        if isinstance(value, dict):
            # as each preprocessor is child of BaseEstimator
            if 'numerical' in value and isinstance(value['numerical'], BaseEstimator):
                preprocessor['numerical'].append(value['numerical'])
            if 'categorical' in value and isinstance(value['categorical'], BaseEstimator):
                preprocessor['categorical'].append(value['categorical'])

    return preprocessor
