import numpy as np
from typing import Optional, Dict, Callable, Tuple, Union
from tqdm import tqdm
import numpy.typing as npt

from ._utils import pbar
from ._predsamplewrapper import PredSampleWrapper
from .metrics import Metric

from . import metrics as metricslib

def two_sample_test(sample_1: Union[npt.NDArray[np.int_], npt.NDArray[np.float], PredSampleWrapper], 
                    sample_2: Union[npt.NDArray[np.int_], npt.NDArray[np.float], PredSampleWrapper], 
                    statistics: Dict[str, Callable]=None, 
                    alpha: float=0.05, 
                    two_tailed: bool = True,
                    n_bootstrap: int=10000, seed: int=None, 
                    silent: bool=False) -> Dict[str, Tuple[float]]:
    """Compares whether the empirical difference of statistics computed own two samples is statistically significant or not.
    Note that the statistics are computed independently, and should thus be treated independently.

    Args:
        sample_1 (np.ndarray): _description_
        sample_2 (np.ndarray): _description_
        statistics (Dict[str, Callable]): Statistics to compare the samples by.
        alpha (float, optional): A signficance level for confidence intervals (from 0 to 1).
        n_bootstrap (int, optional): The number of bootstrap iterations. Defaults to 10000.
        seed (int, optional): _description_. Random seed. Defaults to None.
        silent (bool, optional): Whether to execute the function silently, i.e. not showing the progress bar. Defaults to False.

    Returns:
        Dict[Tuple[float]]: A dictionary containing a tuple with the empirical value of the metric, and the p-value. 
                            The expected format in the output in every dict entry is: p-value, empirical value (sample 1), 
                            CI low (sample 1), CI high (sample 1), empirical value (sample 2), 
                            CI low (sample 2), CI high (sample 2).

    """
    
    if seed is not None:
        np.random.seed(seed)

    alpha = 100 * alpha

    # Dict to to store the null bootstrap distribution
    result = {s_tag: np.zeros((n_bootstrap, 2)) for s_tag in statistics}

    for bootstrap_iter in pbar(range(n_bootstrap), total=n_bootstrap, desc="Bootstrapping", silent=silent):
        ind = np.random.choice(len(sample_1), len(sample_1), replace=True)

        for s_tag in statistics:
            v1 = statistics[s_tag](sample_1[ind])
            v2 = statistics[s_tag](sample_2[ind])
            result[s_tag][bootstrap_iter, 0] = v1
            result[s_tag][bootstrap_iter, 1] = v2
    
    result_final = {}
    for s_tag in result:
        emp_s1 = statistics[s_tag](sample_1)
        emp_s2 = statistics[s_tag](sample_2) 
        emp_diff = emp_s2 - emp_s1
        # The null distribution for the standard error
        null = result[s_tag][:, 1] - result[s_tag][:, 0] - emp_diff
        # Bootstrap checks whether the empirical difference is within the margins of the standard error
        if two_tailed:
            emp_diff = abs(emp_diff)
        p_val = ((null >= abs(emp_diff)).sum() + 1.) / (n_bootstrap + 1)
        # We also want to compute the confidence intervals
        ci_s1 = (np.percentile(result[s_tag][:, 0], alpha / 2.), np.percentile(result[s_tag][:, 0], 100 - alpha / 2.))
        ci_s2 = (np.percentile(result[s_tag][:, 1], alpha / 2.), np.percentile(result[s_tag][:, 1], 100 - alpha / 2.))
        # And we report the p-value, empirical values, as well as the confidence intervals. 
        # The the format in the documentation.
        result_final[s_tag] = (p_val, emp_s1, ci_s1[0], ci_s1[1], emp_s2, ci_s2[0], ci_s2[1])
    return result_final


def compare_models(y_test: np.ndarray, preds_1: np.ndarray, preds_2: np.ndarray, 
                   metrics: Tuple[Union[str, Metric]],
                   alpha: Optional[float]=0.05,
                   two_tailed: bool=True,
                   n_bootstrap: int=10000, seed: int=None, 
                   silent: bool=False) -> Dict[str, Tuple[float]]:
    """Compares predictions from two models. By default, the function assumes that we are solving a
    classification problem. The expected dimension of `preds_1` and `preds_1` is `$N_{samples} \times N_{classes}$`. 
    When `$N_{classes}$` is 1, the function expects the problem to be either binary classification or regression. 
    

    Args:
        y_test (np.ndarray): Ground truth
        preds_1 (np.ndarray): Prediction from model 1
        preds_2 (np.ndarray): Prediction from model 2
        metrics (Tuple[Union[str, Metric]]): A set of metrics to call. Here, the user either specifies the metrcis available from the stambo library, or adds an instance of the custom-defined metrics.
        alpha (float, optional): A signficance level for confidence intervals (from 0 to 1).
        two_tailed (bool, optional): Whether to conduct a two-tailed test. Usually the tests we care about are single-tailed, i.e. the `H_0: model 2 = model 2` vs `H_1: model 2 > model 1`
        n_bootstrap (int, optional): The number of bootstrap iterations. Defaults to 10000.
        seed (int, optional): Random seed. Defaults to None.
        silent (bool, optional): Whether to execute the function silently, i.e. not showing the progress bar. Defaults to False.

    Returns:
        Dict[Tuple[float]]: A dictionary containing a tuple with the empirical value of the metric, and the p-value. 
                            The expected format in the output in every dict entry is: 
                            * p-value
                            * empirical value (model 1)
                            * CI low (model 1)
                            * CI high (model 1)
                            * empirical value (model 2)
                            * CI low (model 2)
                            * CI high (model 2)
    """

    # Data samples need to be prepared
    sample_1 = PredSampleWrapper(preds_1, y_test, multiclass=len(preds_1.shape) != 1)
    sample_2 = PredSampleWrapper(preds_2, y_test, multiclass=len(preds_1.shape) != 1)

    metrics_dict = {}
    for metric in metrics:
        if isinstance(metric, Metric):
            metrics_dict[str(metric)] = metric # note that the object must be instantiated
        elif isinstance(metric, str):
            assert hasattr(metricslib, metric), f"Metric {metric} is not defined"
            metrics_dict[metric] = getattr(metricslib, metric)()

    return two_sample_test(sample_1, sample_2, statistics=metrics_dict, alpha=alpha,
                           two_tailed=two_tailed, n_bootstrap=n_bootstrap, 
                           seed=seed, silent=silent)
    

