# RAYLOGP
A physicochemical property-based method for the estimation of the octanol-water partition coefficient, logP, submitted for the 2019 SAMPL6 LogP Prediction Challenge (https://github.com/samplchallenges/SAMPL6/tree/master/physical_properties/logP).

This repository contains the experimental files used to compare different methods of representing a molecule in-silico to effectively represent a chemical space for linear correlation with logP via quantitative structure-property relationships (QSPRs). Molecular representation comparisons were performed by assessing the continuity of structure-lipophilicity landscapes and internally benchmarking the performance of linear QSPR models predicting logP.

The overall best model, codenamed 'RAYLOGP', was revealed to be a stochastic gradient descent-optimised multilinear regression with 1,438 physicochemical descriptors. RAYLOGP was externally tested in a real-world drug-like scenario by generating blind predictions for the 11 protein kinase inhibitor fragment-like molecules prepared for the SAMPL6 Challenge. Our submission ID was 'hdpuj' and returned an RMSE of 0.49 log units.
