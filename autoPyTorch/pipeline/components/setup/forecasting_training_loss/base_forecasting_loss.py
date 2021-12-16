from typing import Dict, Any

from autoPyTorch.pipeline.components.base_component import autoPyTorchComponent

from autoPyTorch.utils.common import FitRequirement


class ForecastingLossComponents(autoPyTorchComponent):
    _required_properties = ["name", "handles_tabular", "handles_image", "handles_time_series",
                            'handles_regression', 'handles_classification']
    loss = None
    required_net_out_put_type = None

    def __init__(self,
                 **kwargs: Any):
        super().__init__()
        self.add_fit_requirements([
            FitRequirement('task_type', (str,), user_defined=True, dataset_property=True),
        ])

    def fit(self, X: Dict[str, Any], y: Any = None) -> "autoPyTorchComponent":
        self.check_requirements(X, y)
        return self

    def transform(self, X: Dict[str, Any]) -> Dict[str, Any]:
        X.update({"loss": self.loss,
                  'required_net_out_put_type': self.required_net_out_put_type})
        return X