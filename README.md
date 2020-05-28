# RAYLOGP
A machine learning QSPR model predicting the octanol-water partition coefficient (logP) using PaDEL physicochemical descriptors.


This repository contains the experimental files used to compare different methods of representing a molecule in-silico to effectively model a chemical space to logP as described by [Lui et al. (2020) JCAMD 34: 523-534](https://link.springer.com/article/10.1007/s10822-020-00279-0).


The overall best model, 'RAYLOGP', was a [stochastic gradient descent-optimised linear regression](https://scikit-learn.org/0.20/modules/generated/sklearn.linear_model.SGDRegressor.html) with 1,438 physicochemical descriptors calculated in [PaDEL](http://www.yapcwsoft.com/dd/padeldescriptor/). External validation was performed by predicting for 11 protein kinase inhibitor fragment-like molecules prepared for the [2019 SAMPL6 LogP Prediction Challenge](https://github.com/samplchallenges/SAMPL6/tree/master/physical_properties/logP), returning an average RMSE of 0.49 log units (submission ID: 'hdpuj').
