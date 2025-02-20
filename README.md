# Cyclic Coordinate Descent for Regularized Logistic Regression

This project focuses on implementing the Cyclic Coordinate Descent (CCD) algorithm for L1-regularized logistic regression. https://www.jstatsoft.org/article/view/v033i01

## Project Tasks

### Task 1: Data Collection and Preparation
* Select 4 real-world classification datasets.
* Preprocess the data (missing values, collinearity, feature scaling).
* Generate synthetic datasets.
### Task 2: Algorithm Implementation
* Implement CCD-based logistic regression (LogRegCCD).
* Implement evaluation functions and visualization.
### Task 3: Experiments and Analysis
* Compare LogRegCCD with sklearn’s LogisticRegression.
* Analyze performance using various evaluation metrics.

## Code Requirements
* Implementation details:
  * fit(...)
  * validate(..., measure)
  * predict proba(X, ...)
  * plot score(measure, ...) – comparison on different lambda
  * plot coeff(...) – comparison on different lambda
* optimal lambda should be find using validation set
* code should be maintained on a public repository, e.g., GitHub
* README.md
* documentation in docstrings
* general readability of code and use of good practices for granularity of functions
* saving all files (plots, tables) used for the report in separate files
* reproducibility of results on new data:
  * one notebook/script file that can create all assets used in the report for a given data
  * for a script, path to data should be passed as an argument, for a notebook should be clearly defined as a variable
  * data are supposed to be in CSV format, with no headers, with a target value in the last column; splitting should be done by the students’ code
  * instruction for running the code is obligatory (remember about requirements!)


