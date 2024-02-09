from typing import Callable

from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.metrics import cohen_kappa_score, balanced_accuracy_score, matthews_corrcoef
from sklearn.metrics import mean_absolute_error, mean_squared_error
from functools import partial

from ._predsamplewrapper import PredSampleWrapper

__all__ = ["ROCAUC", "AP", "F1Score", "QKappa", "BACC", "MCC", "MSE", "MAE"]

# Base metric class
class Metric:
    def __init__(self, metric: Callable, int_input: bool=False, binary=False) -> None:
        """A wrapper for metrics that take predictions and ground truth labels as two arguments.

        Args:
            metric (Callable): The metric of choice. The typical ones are ROC-AUC, Average precision etc. See more in the [sklearn documentation](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics).
            int_input (bool, optional): Defines whether the metric takes predictions as integers. Defaults to False.
            binary (bool, optional): Defines whether the metric expects an input of a binary classifier, 
        """
        self.binary = binary
        self.metric = metric
        self.int_input = int_input

    def __call__(self, sample: PredSampleWrapper) -> float:
        if self.int_input: # Handling the case when the metric expects an integer input, i.e. cohen's cappa
            return self.metric(sample.gt, sample.predictions_am)
        return self.metric(sample.gt, sample.predictions)
    
# Classification metrics
class ROCAUC(Metric):
    """The ROC-AUC metric. Defined for Binary classifiers.
    """
    def __init__(self) -> None:
        Metric.__init__(self, roc_auc_score, int_input=False, binary=True)

    def __str__(self) -> str:
        return "roc_auc"

class AP(Metric):
    """The Average Precision Score metric. Defined for Binary classifiers.
    """
    def __init__(self) -> None:
        Metric.__init__(self, average_precision_score, int_input=False, binary=True)

    def __str__(self) -> str:
        return "average_precision"
    
class F1Score(Metric):
    """The F1 score metric. Defined for Binary classifiers.
    """
    def __init__(self) -> None:
        Metric.__init__(self, f1_score, int_input=False, binary=True)

    def __str__(self) -> str:
        return "average_precision"

class QKappa(Metric):
    """Cohen's kappa score (quadratic).
    """
    def __init__(self) -> None:
        Metric.__init__(self, partial(cohen_kappa_score, weights="quadratic"), int_input=True, binary=False)

    def __str__(self) -> str:
        return "kappa"

class BACC(Metric):
    """The balanced accuracy score.
    """
    def __init__(self) -> None:
        Metric.__init__(self, balanced_accuracy_score, int_input=True, binary=False)

    def __str__(self) -> str:
        return "balanced_accuracy"

class MCC(Metric):
    """The Matthew's correlation coefficient
    """
    def __init__(self) -> None:
        Metric.__init__(self, matthews_corrcoef, int_input=True, binary=False)

    def __str__(self) -> str:
        return "mcc"

# Regression metrics

class MSE(Metric):
    """The Mean Squared Error
    """
    def __init__(self) -> None:
        Metric.__init__(self, mean_squared_error, int_input=False, binary=False)

    def __str__(self) -> str:
        return "mse"

class MAE(Metric):
    """The mean absolute error.
    """
    def __init__(self) -> None:
        Metric.__init__(self, mean_absolute_error, int_input=False, binary=False)

    def __str__(self) -> str:
        return "mae"




    
