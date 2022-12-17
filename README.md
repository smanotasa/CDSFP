# Computing for Data Science Final Project (BSE 22/23)
## Customer Churn Prediction
Authors: Santiago Manotas-Arroyave, Catalina Odizzio, Agostina Alexandra Pissinis, Shaney Sze
Contents
- What is `lumpia`?
- [Installation](#installation)
- [Scaling Guidelines]
- [Contributing](#contributing)


## What is `lumpia`?
Lumpia is a library to predict customer churn using an end-to-end pipelines:  run a prediction model, create features, optimize hypterparameters, score, and unit test.

## Installation
Install the latest version for a stable release.

```bash
pip install -U git+https://github.com/smanotasa/CDSFP.git
```

## Scaling Guidelines
Before creating your own library which extends `lumpia`, please consider would the extension be also useful also for general usage. If it could be useful also for general usage, please create a new issue describing the enhancement request and even better if the issue is backed up by a pull request.

- *Adding Preprocessors*
    - Navigate to `src/lumpia/pre_processing/pre_processing.py`
    - Under the `preprocessor` class, define the desired functions to be added.  By default, `lumpia` utilizes `drop_nan` and `fill_mean` functions

- *Adding Features*
    - Navigate to `src/lumpia/features/features.py`
    - Define the desired functions to be added.  By default, `lumpia` utilizes `gen_dummies`, `normalize`, and `take_log` functions to modify the customer churn data

- *Adding Models*
    - Navigate to `src/lumpia/model/model.py`
    - Define the desired functions to be added.  By default, `lumpia` utilizes the `train_model` and uses random forest to predict values

- *Adding Metrics*
    - Navigate to `src/lumpia/evaluation/evaluation.py`
    - Define the desired metrics to be added.  By default, `lumpia` utilizes the `get_roc_auc_score` and the ROC area under curve method to score the model

## Contributing
Want to contribute to `lumpia`? Feel free to add any new agents or environments by [opening an issue](https://github.com/smanotasa/CDSFP/issues/new)! Additionally, make sure any new code is properly accompanied by working through the following checklist:
  - [ ] Tests have been added for any new functionality in ``CDSFP/tests``
  - [ ] All other tests still pass (when running ``pytest`` in the root directory)
  - [ ] Any functions added have [``numpy`` style docstrings](http://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html)
