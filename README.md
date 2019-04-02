# Symbolic Regression in Dynamic Scenarios With Gradually Changing Targets - supplementary data

This repository contains supplementary data for the article **Symbolic Regression in Dynamic Scenarios With Gradually Changing Targets**.

## Datasets
The files [artificial.zip](artificial.zip), [MM.zip](MM.zip), [MM2.zip](MM2.zip), [PS.zip](PS.zip) and [PS2.zip](PS2.zip) contain the datasets used in the article (with corresponding names).
Each of the zip files contains two files - `train.csv` which contains the training data (used to run the algorithms in the experiments) and `test.csv` which contains the testing data (used to evaluate the algorithms).
For the `artificial` dataset, there are 20 pairs of those files, prefixed with a number from `00` do `19`, which contain the 20 samplings of this dataset.

The CSV files have the following form:

* each line represents one data point
* first `N` columns (for PS and MM `N = 2`, for PS2 and MM2 `N = 4`, for artificial `N = 7`) represent the coordinates of the data points, i.e. the features
* the remaining columns represent the target value of the data points for the different stages, with first of those columns representing the first stage, the next column the second stage, etc.

## Code
The file [runner.py](runner.py) is a Python script that uses our Evo framework [[1](#ref-evo)] and sets it up for the dynamic experiments.
When run, the working directory needs to be the same as that of the `setup.py` file in Evo, or Evo needs to be installed in your python environment.

---

[1] <a id="ref-evo"></a>[https://github.com/zegkljan/evo](https://github.com/zegkljan/evo), commit [108f140](https://github.com/zegkljan/evo/tree/108f1407aa388357dd77f45f3c71a5eb64957a48)