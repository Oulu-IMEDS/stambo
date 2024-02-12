# StaMBO: Statistical model comparison with bootstrap 
[![PyPI version](https://badge.fury.io/py/stambo.svg?branch=master)](https://badge.fury.io/py/stambo)
[![docs](https://github.com/oulu-imeds/stambo/workflows/documentation/badge.svg)](https://oulu-imeds.github.io/stambo/)
[![License](http://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat)](LICENSE.md)
------------------------
This package is aimed to be a one-stop-shop for statistical testing in machine learning when it comes to evaluating models on a test set and comparing whether our *improved* model is really beating the baseline. That is, we cover the following very typical use-case in machine learning:
![usecase](docs/source/_static/usecase.png)

Currently, we support the cases of classification, regresson, and semantic segmentation. We do not yet support the significance of ranking, as well as grouped data. It is coming in the future releases.

## In practice
Install from PyPI:
```
pip install stambo
```

The use of the library is then straightforward:
```
import stambo
...
seed = 42
testing_result = stambo.compare_models(y_test, preds_1, preds_2, metrics=("ROCAUC", "AP", "QKappa", "BACC", "MCC"), seed=seed)
print(stambo.to_latex(testing_result))
```

The above will print a LaTeX table, which one can easily copy-paste. As an example, below is the rendered table, which was returned in `notebooks/Classification_example.ipynb`:
![Table](docs/source/_static/example_table.png)

For more advanced documentation see the documentation. By default, binary, multi-class, and multi-label classification, as well as regression are supported.

One can also use the library to perform a simple two-sample test. For example, to compare the means of two distributions:
```
import stambo
...
seed = 42
res = stambo.two_sample_test(sample_1, sample_2, statistics={"Mean": lambda x: x.mean()})
```


## Contributing

To setup a dev environment, you should use the provided environment file, and compile the documentation locally:
```
conda env create -f env.yaml
conda activate stambo-dev
pip install -e .
cd docs
make html
```

## Author
Aleksei Tiulpin
