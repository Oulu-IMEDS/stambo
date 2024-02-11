from typing import Callable

from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.metrics import cohen_kappa_score, balanced_accuracy_score, matthews_corrcoef
from sklearn.metrics import mean_absolute_error, mean_squared_error
from functools import partial

from ._predsamplewrapper import PredSampleWrapper

__all__ = ["Metric", "ROCAUC", "AP", "F1Score", "QKappa", "BACC", "MCC", "MSE", "MAE"]

# Base metric class
class Metric:
    """A wrapper for metrics that take predictions and ground truth labels as two arguments.
    """
    def __init__(self, metric: Callable, int_input: bool=False) -> None:
        """Constructor of the metric wrapper class

        Args:
            metric (Callable): The metric of choice. The typical ones are ROC-AUC, Average precision etc. 
            See more in the https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics.
            int_input (bool, optional): Defines whether the metric takes predictions as integers. Defaults to False.
        """
        self.metric = metric
        self.int_input = int_input

    def __call__(self, sample: PredSampleWrapper) -> float:
        """The call method. This runs the metric on the supplied data. If the metric is meanto to be run on integer input.
        it wil use the argmaxed predictions that are stored by the `PredSampleWrapper` object.

        Args:
            sample (PredSampleWrapper): Data on which the metric is computed. 

        Returns:
            float: Metric value. 
        """
        if self.int_input: # Handling the case when the metric expects an integer input, i.e. cohen's cappa
            return self.metric(sample.gt, sample.predictions_am)
        return self.metric(sample.gt, sample.predictions)
    
# Classification metrics
class ROCAUC(Metric):
    """The ROC-AUC metric. Defined for Binary classifiers.
    """
    def __init__(self) -> None:
        Metric.__init__(self, roc_auc_score, int_input=False)

    def __str__(self) -> str:
        return "ROCAUC"

class AP(Metric):
    """The Average Precision metric. Defined for Binary classifiers.
    """
    def __init__(self) -> None:
        Metric.__init__(self, average_precision_score, int_input=False)

    def __str__(self) -> str:
        return "AP"
    
class F1Score(Metric):
    """The F1 score metric. Defined for Binary classifiers.
    """
    def __init__(self) -> None:
        Metric.__init__(self, f1_score, int_input=False)

    def __str__(self) -> str:
        return "F1Score"

class QKappa(Metric):
    """Cohen's kappa score (quadratic).
    """
    def __init__(self) -> None:
        Metric.__init__(self, partial(cohen_kappa_score, weights="quadratic"), int_input=True)

    def __str__(self) -> str:
        return "QKappa"

class BACC(Metric):
    """The balanced accuracy score.
    """
    def __init__(self) -> None:
        Metric.__init__(self, balanced_accuracy_score, int_input=True)

    def __str__(self) -> str:
        return "BACC"

class MCC(Metric):
    """The Matthew's correlation coefficient
    """
    def __init__(self) -> None:
        Metric.__init__(self, matthews_corrcoef, int_input=True)

    def __str__(self) -> str:
        return "MCC"

# Regression metrics

class MSE(Metric):
    """The Mean squared error
    """
    def __init__(self) -> None:
        Metric.__init__(self, mean_squared_error, int_input=False)

    def __str__(self) -> str:
        return "MSE"

class MAE(Metric):
    """The mean absolute error.
    """
    def __init__(self) -> None:
        Metric.__init__(self, mean_absolute_error, int_input=False)

    def __str__(self) -> str:
        return "MAE"




    
