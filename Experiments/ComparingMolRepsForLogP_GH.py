'''
Raymond Lui
rlui9522@uni.sydney.edu.au
Pharmacoinformatics Laboratory
Discipline of Pharmacology, School of Medical Sciences
Faculty of Medicine and Health, The University of Sydney
Sydney, New South Wales, 2006, Australia
'''



import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
from time import time



print('********** Comparing molecular representations for logP QSPR prediction **********')
start0 = time()
seed=69



#   ===== EXPERIMENTAL PIPELINE =====
#   
#      (0) Package dependencies.......................Line 46
#  
#   MOLECULAR DATA PREPARATION:
#      (1) Data loading...............................Line 69
#      (2) Feature selection..........................Line 283
#      (3) Composite representation generation........Line 424
#      (4) Data scaling...............................Line 489
#   
#   QSPR MODEL DEVELOPMENT:
#      (5) Parameter optimisation.....................Line 630
#      (6) Algorithm training and model predictions...Line 1049
#      (7) Model benchmarking.........................Line 1447
#      (8) Result visualisation.......................Line 1762



#=============================================================================
# (0) PACKAGE DEPENDENCIES
#=============================================================================
print('\n\n\nPackages used:')

import sys
print('> Python v%s' % sys.version) #v3.6.8
import numpy as np
print('> NumPy v%s' % np.__version__) #v1.16.4
import pandas as pd
print('> pandas v%s' % pd.__version__) #v0.22.0
import scipy as sp
print('> SciPy v%s' % sp.__version__) #v1.3.1
import sklearn
print('> scikit-learn v%s' % sklearn.__version__) #v0.20.3
import matplotlib
import matplotlib.pyplot as plt
print('> Matplotlib v%s' % matplotlib.__version__) #v3.1.1
import seaborn as sns
print('> seaborn v%s' % sns.__version__) #v0.9.0



#=============================================================================
# (1) DATA LOADING
#=============================================================================
print('\n\n\n########## DATA LOADING ##########')
start1 = time()



#   ===== DATA VARIABLE NOMENCLATURE =====
#   
#   TYPE OF VARIABLE:
#      x_ = molecular representation; QSPR predictor variables (i.e. y = mX1 + mX2 + ... +mXn + b)
#      y_ = logP; QSPR response variable (i.e. Y = mx1 + mx2 + ... +mXn + b)
#   
#   TYPE OF DATASET:
#      T_  = Training set
#      Tc_ = Training set used to generate Composite representation
#      V_  = Validation set
#   
#   TYPE OF MOLECULAR REPRESENTATION:
#      PC  = PhysicoChemical descriptors
#      SK  = Structural Key
#      FP  = circular FingerPrint
#      COM = COMposite representation
#   
#   TYPE OF VARIABLE: (for SK, FP, COM)
#      bn = BiNary
#      ct = CounT
#   
#   DEGREE OF FEATURE DENSITY:
#      None = 100% full representation
#      F1   = 75% feature selected (PC, SK, COM) or hashed (FP, COM)
#      F2   = 50% feature selected (PC, SK, COM) or hashed (FP, COM)
#      F3   = 25% feature selected (PC, SK, COM) or hashed (FP, COM)
#      F4   = 12.5% feature selected (PC, SK, COM) or hashed (FP, COM)



# TRAINING SETS
## Physicochemical descriptors
T_PC = pd.read_csv('Mansouri14045Train_DescriptorsFinal.csv')
x_T_PC = T_PC.iloc[:,0:1438].values
y_T_PC = T_PC.iloc[:,1438].values

## Structural keys
T_SKbn = pd.read_csv('Mansouri14050Train_StructuralKeyFinal.csv')
x_T_SKbn = T_SKbn.iloc[:,0:1354].values
y_T_SKbn = T_SKbn.iloc[:,1354].values

T_SKct = pd.read_csv('Mansouri14050Train_StructuralKeyFinal_counts.csv')
x_T_SKct = T_SKct.iloc[:,0:1354].values
y_T_SKct = T_SKct.iloc[:,1354].values

## Circular fingerprints
T_FPbn = pd.read_csv('Mansouri13710Train_MorganFP1024Final.csv')
x_T_FPbn = T_FPbn.iloc[:,0:1024].values
y_T_FPbn = T_FPbn.iloc[:,1024].values
T_FPct = pd.read_csv('Mansouri13710Train_MorganFP1024Final_counts.csv')
x_T_FPct = T_FPct.iloc[:,0:1024].values
y_T_FPct = T_FPct.iloc[:,1024].values

T_FPbnF1 = pd.read_csv('Mansouri13710Train_MorganFP768Final.csv')
x_T_FPbnF1 = T_FPbnF1.iloc[:,0:768].values
y_T_FPbnF1 = T_FPbnF1.iloc[:,768].values
T_FPctF1 = pd.read_csv('Mansouri13710Train_MorganFP768Final_counts.csv')
x_T_FPctF1 = T_FPctF1.iloc[:,0:768].values
y_T_FPctF1 = T_FPctF1.iloc[:,768].values

T_FPbnF2 = pd.read_csv('Mansouri13710Train_MorganFP512Final.csv')
x_T_FPbnF2 = T_FPbnF2.iloc[:,0:512].values
y_T_FPbnF2 = T_FPbnF2.iloc[:,512].values
T_FPctF2 = pd.read_csv('Mansouri13710Train_MorganFP512Final_counts.csv')
x_T_FPctF2 = T_FPctF2.iloc[:,0:512].values
y_T_FPctF2 = T_FPctF2.iloc[:,512].values

T_FPbnF3 = pd.read_csv('Mansouri13710Train_MorganFP256Final.csv')
x_T_FPbnF3 = T_FPbnF3.iloc[:,0:256].values
y_T_FPbnF3 = T_FPbnF3.iloc[:,256].values
T_FPctF3 = pd.read_csv('Mansouri13710Train_MorganFP256Final_counts.csv')
x_T_FPctF3 = T_FPctF3.iloc[:,0:256].values
y_T_FPctF3 = T_FPctF3.iloc[:,256].values

T_FPbnF4 = pd.read_csv('Mansouri13710Train_MorganFP128Final.csv')
x_T_FPbnF4 = T_FPbnF4.iloc[:,0:128].values
y_T_FPbnF4 = T_FPbnF4.iloc[:,128].values
T_FPctF4 = pd.read_csv('Mansouri13710Train_MorganFP128Final_counts.csv')
x_T_FPctF4 = T_FPctF4.iloc[:,0:128].values
y_T_FPctF4 = T_FPctF4.iloc[:,128].values

print('\nTRAINING SETS\nPHYSICOCHEMICAL DESCRIPTORS:\n> %s molecules with %s descriptors' % (y_T_PC.shape[0], x_T_PC.shape[1]),
      '\nSTRUCTURAL KEYS:\n> %s molecules with %s binary bits' % (y_T_SKbn.shape[0], x_T_SKbn.shape[1]),
      '\n> %s molecules with %s count bits' % (y_T_SKct.shape[0], x_T_SKct.shape[1]),
      '\nCIRCULAR FINGERPRINTS:\n> %s molecules with %s binary bits' % (y_T_FPbn.shape[0], x_T_FPbn.shape[1]),
      '\n> %s molecules with %s count bits' % (y_T_FPct.shape[0], x_T_FPct.shape[1]),
      '\n> %s molecules with  %s binary bits' % (y_T_FPbnF1.shape[0], x_T_FPbnF1.shape[1]),
      '\n> %s molecules with  %s count bits' % (y_T_FPctF1.shape[0], x_T_FPctF1.shape[1]),
      '\n> %s molecules with  %s binary bits' % (y_T_FPbnF2.shape[0], x_T_FPbnF2.shape[1]),
      '\n> %s molecules with  %s count bits' % (y_T_FPctF2.shape[0], x_T_FPctF2.shape[1]),
      '\n> %s molecules with  %s binary bits' % (y_T_FPbnF3.shape[0], x_T_FPbnF3.shape[1]),
      '\n> %s molecules with  %s count bits' % (y_T_FPctF3.shape[0], x_T_FPctF3.shape[1]),
      '\n> %s molecules with  %s binary bits' % (y_T_FPbnF4.shape[0], x_T_FPbnF4.shape[1]),
      '\n> %s molecules with  %s count bits' % (y_T_FPctF4.shape[0], x_T_FPctF4.shape[1]))



# COMPOSITE TRAINING SETS
## Physicochemical descriptors
Tc_PC = pd.read_csv('Mansouri13705Train_DescriptorEnsembleFinal.csv')
x_Tc_PC = Tc_PC.iloc[:,0:1438].values
y_Tc = Tc_PC.iloc[:,1438].values

## Structural keys
x_Tc_SKbn = pd.read_csv('Mansouri13705Train_StructureKeyEnsembleFinal.csv').iloc[:,0:1354].values
x_Tc_SKct = pd.read_csv('Mansouri13705Train_StructureKeyCTEnsembleFinal.csv').iloc[:,0:1354].values

## Circular fingerprints
x_Tc_FPbn = pd.read_csv('Mansouri13705Train_MorganFP1024bnEnsembleFinal.csv').iloc[:,0:1024].values
x_Tc_FPct = pd.read_csv('Mansouri13705Train_MorganFP1024ctEnsembleFinal.csv').iloc[:,0:1024].values
x_Tc_FPbnF1 = pd.read_csv('Mansouri13705Train_MorganFP768bnEnsembleFinal.csv').iloc[:,0:768].values
x_Tc_FPctF1 = pd.read_csv('Mansouri13705Train_MorganFP768ctEnsembleFinal.csv').iloc[:,0:768].values
x_Tc_FPbnF2 = pd.read_csv('Mansouri13705Train_MorganFP512bnEnsembleFinal.csv').iloc[:,0:512].values
x_Tc_FPctF2 = pd.read_csv('Mansouri13705Train_MorganFP512ctEnsembleFinal.csv').iloc[:,0:512].values
x_Tc_FPbnF3 = pd.read_csv('Mansouri13705Train_MorganFP256bnEnsembleFinal.csv').iloc[:,0:256].values
x_Tc_FPctF3 = pd.read_csv('Mansouri13705Train_MorganFP256ctEnsembleFinal.csv').iloc[:,0:256].values
x_Tc_FPbnF4 = pd.read_csv('Mansouri13705Train_MorganFP128bnEnsembleFinal.csv').iloc[:,0:128].values
x_Tc_FPctF4 = pd.read_csv('Mansouri13705Train_MorganFP128ctEnsembleFinal.csv').iloc[:,0:128].values

print('\nCOMPOSITE TRAINING SETS\nPHYSICOCHEMICAL DESCRIPTORS:\n> %s molecules with %s descriptors' % (y_Tc.shape[0], x_Tc_PC.shape[1]),
      '\nSTRUCTURAL KEYS:\n> %s molecules with %s binary bits' % (y_Tc.shape[0], x_Tc_SKbn.shape[1]),
      '\n> %s molecules with %s count bits' % (y_Tc.shape[0], x_Tc_SKct.shape[1]),
      '\nCIRCULAR FINGERPRINTS:\n> %s molecules with %s binary bits' % (y_Tc.shape[0], x_Tc_FPbn.shape[1]),
      '\n> %s molecules with %s count bits' % (y_Tc.shape[0], x_Tc_FPct.shape[1]),
      '\n> %s molecules with  %s binary bits' % (y_Tc.shape[0], x_Tc_FPbnF1.shape[1]),
      '\n> %s molecules with  %s count bits' % (y_Tc.shape[0], x_Tc_FPctF1.shape[1]),
      '\n> %s molecules with  %s binary bits' % (y_Tc.shape[0], x_Tc_FPbnF2.shape[1]),
      '\n> %s molecules with  %s count bits' % (y_Tc.shape[0], x_Tc_FPctF2.shape[1]),
      '\n> %s molecules with  %s binary bits' % (y_Tc.shape[0], x_Tc_FPbnF3.shape[1]),
      '\n> %s molecules with  %s count bits' % (y_Tc.shape[0], x_Tc_FPctF3.shape[1]),
      '\n> %s molecules with  %s binary bits' % (y_Tc.shape[0], x_Tc_FPbnF4.shape[1]),
      '\n> %s molecules with  %s count bits' % (y_Tc.shape[0], x_Tc_FPctF4.shape[1]))



# VALIDATION SETS
## Physicochemical descriptors
V_PC = pd.read_csv('Martel707Validation_DescriptorsFinal.csv')
x_V_PC = V_PC.iloc[:,0:1438].values
y_V_PC = V_PC.iloc[:,1438].values

## Structural keys
V_SKbn = pd.read_csv('Martel707Validation_StructuralKeyFinal.csv')
x_V_SKbn = V_SKbn.iloc[:,0:1354].values
y_V_SKbn = V_SKbn.iloc[:,1354].values

V_SKct = pd.read_csv('Martel707Validation_StructuralKeyFinal_counts.csv')
x_V_SKct = V_SKct.iloc[:,0:1354].values
y_V_SKct = V_SKct.iloc[:,1354].values

## Circular fingerprints
V_FPbn = pd.read_csv('Martel707Validation_MorganFP1024Final.csv')
x_V_FPbn = V_FPbn.iloc[:,0:1024].values
y_V_FPbn = V_FPbn.iloc[:,1024].values
V_FPct = pd.read_csv('Martel707Validation_MorganFP1024Final_counts.csv')
x_V_FPct = V_FPct.iloc[:,0:1024].values
y_V_FPct = V_FPct.iloc[:,1024].values

V_FPbnF1 = pd.read_csv('Martel707Validation_MorganFP768Final.csv')
x_V_FPbnF1 = V_FPbnF1.iloc[:,0:768].values
y_V_FPbnF1 = V_FPbnF1.iloc[:,768].values
V_FPctF1 = pd.read_csv('Martel707Validation_MorganFP768Final_counts.csv')
x_V_FPctF1 = V_FPctF1.iloc[:,0:768].values
y_V_FPctF1 = V_FPctF1.iloc[:,768].values

V_FPbnF2 = pd.read_csv('Martel707Validation_MorganFP512Final.csv')
x_V_FPbnF2 = V_FPbnF2.iloc[:,0:512].values
y_V_FPbnF2 = V_FPbnF2.iloc[:,512].values
V_FPctF2 = pd.read_csv('Martel707Validation_MorganFP512Final_counts.csv')
x_V_FPctF2 = V_FPctF2.iloc[:,0:512].values
y_V_FPctF2 = V_FPctF2.iloc[:,512].values

V_FPbnF3 = pd.read_csv('Martel707Validation_MorganFP256Final.csv')
x_V_FPbnF3 = V_FPbnF3.iloc[:,0:256].values
y_V_FPbnF3 = V_FPbnF3.iloc[:,256].values
V_FPctF3 = pd.read_csv('Martel707Validation_MorganFP256Final_counts.csv')
x_V_FPctF3 = V_FPctF3.iloc[:,0:256].values
y_V_FPctF3 = V_FPctF3.iloc[:,256].values

V_FPbnF4 = pd.read_csv('Martel707Validation_MorganFP128Final.csv')
x_V_FPbnF4 = V_FPbnF4.iloc[:,0:128].values
y_V_FPbnF4 = V_FPbnF4.iloc[:,128].values
V_FPctF4 = pd.read_csv('Martel707Validation_MorganFP128Final_counts.csv')
x_V_FPctF4 = V_FPctF4.iloc[:,0:128].values
y_V_FPctF4 = V_FPctF4.iloc[:,128].values

print('\nVALIDATION SETS\nPHYSICOCHEMICAL DESCRIPTORS:\n> %s molecules with %s descriptors' % (y_V_PC.shape[0], x_V_PC.shape[1]),
      '\nSTRUCTURAL KEYS:\n> %s molecules with %s binary bits' % (y_V_SKbn.shape[0], x_V_SKbn.shape[1]),
      '\n> %s molecules with %s count bits' % (y_V_SKct.shape[0], x_V_SKct.shape[1]),
      '\nCIRCULAR FINGERPRINTS:\n> %s molecules with %s binary bits' % (y_V_FPbn.shape[0], x_V_FPbn.shape[1]),
      '\n> %s molecules with %s count bits' % (y_V_FPct.shape[0], x_V_FPct.shape[1]),
      '\n> %s molecules with  %s binary bits' % (y_V_FPbnF1.shape[0], x_V_FPbnF1.shape[1]),
      '\n> %s molecules with  %s count bits' % (y_V_FPctF1.shape[0], x_V_FPctF1.shape[1]),
      '\n> %s molecules with  %s binary bits' % (y_V_FPbnF2.shape[0], x_V_FPbnF2.shape[1]),
      '\n> %s molecules with  %s count bits' % (y_V_FPctF2.shape[0], x_V_FPctF2.shape[1]),
      '\n> %s molecules with  %s binary bits' % (y_V_FPbnF3.shape[0], x_V_FPbnF3.shape[1]),
      '\n> %s molecules with  %s count bits' % (y_V_FPctF3.shape[0], x_V_FPctF3.shape[1]),
      '\n> %s molecules with  %s binary bits' % (y_V_FPbnF4.shape[0], x_V_FPbnF4.shape[1]),
      '\n> %s molecules with  %s count bits' % (y_V_FPctF4.shape[0], x_V_FPctF4.shape[1]))



print('\n(%.2fs)' % (time()-start1))



#=============================================================================
# (2) FEATURE SELECTION
#=============================================================================
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel



print('\n\n\n########## FEATURE SELECTION ##########')
start2 = time()



# FEATURE SELECTION 1: 75%
# Physicochemical descriptors
FeatureSelector1PC = SelectFromModel((RandomForestRegressor(n_jobs=-1, random_state=seed).fit(x_T_PC, y_T_PC)), max_features=1079, threshold=-np.inf, prefit=True)
x_T_PCF1 = FeatureSelector1PC.transform(x_T_PC)
x_Tc_PCF1 = FeatureSelector1PC.transform(x_Tc_PC)
x_V_PCF1 = FeatureSelector1PC.transform(x_V_PC)

# Structural keys (binary)
FeatureSelector1SKbn = SelectFromModel((RandomForestRegressor(n_jobs=-1, random_state=seed).fit(x_T_SKbn, y_T_SKbn)), max_features=1016, threshold=-np.inf, prefit=True)
x_T_SKbnF1 = FeatureSelector1SKbn.transform(x_T_SKbn)
x_Tc_SKbnF1 = FeatureSelector1SKbn.transform(x_Tc_SKbn)
x_V_SKbnF1 = FeatureSelector1SKbn.transform(x_V_SKbn)

# Structural keys (count)
FeatureSelector1SKct = SelectFromModel((RandomForestRegressor(n_jobs=-1, random_state=seed).fit(x_T_SKct, y_T_SKct)), max_features=1016, threshold=-np.inf, prefit=True)
x_T_SKctF1 = FeatureSelector1SKct.transform(x_T_SKct)
x_Tc_SKctF1 = FeatureSelector1SKct.transform(x_Tc_SKct)
x_V_SKctF1 = FeatureSelector1SKct.transform(x_V_SKct)

print('\n75%% FEATURE SELECTION\nTRAINING SETS:\n> %s molecules with %s physicochemical descriptors' % (x_T_PCF1.shape[0], x_T_PCF1.shape[1]),
      '\n> %s molecules with %s binary structural key bits' % (x_T_SKbnF1.shape[0], x_T_SKbnF1.shape[1]),
      '\n> %s molecules with %s count structural key bits' % (x_T_SKctF1.shape[0], x_T_SKctF1.shape[1]),
      '\nCOMPOSITE TRAINING SETS:\n> %s molecules with %s physicochemical descriptors' % (x_Tc_PCF1.shape[0], x_Tc_PCF1.shape[1]),
      '\n> %s molecules with %s binary structural key bits' % (x_Tc_SKbnF1.shape[0], x_Tc_SKbnF1.shape[1]),
      '\n> %s molecules with %s count structural key bits' % (x_Tc_SKctF1.shape[0], x_Tc_SKctF1.shape[1]),
      '\nVALIDATION SETS:\n> %s molecules with %s physicochemical descriptors' % (x_V_PCF1.shape[0], x_V_PCF1.shape[1]),
      '\n> %s molecules with %s binary structural key bits' % (x_V_SKbnF1.shape[0], x_V_SKbnF1.shape[1]),
      '\n> %s molecules with %s count structural key bits'% (x_V_SKctF1.shape[0], x_V_SKctF1.shape[1]))



# FEATURE SELECTION 2: 50%
# Physicochemical descriptors
FeatureSelector2PC = SelectFromModel((RandomForestRegressor(n_jobs=-1, random_state=seed).fit(x_T_PC, y_T_PC)), max_features=719, threshold=-np.inf, prefit=True)
x_T_PCF2 = FeatureSelector2PC.transform(x_T_PC)
x_Tc_PCF2 = FeatureSelector2PC.transform(x_Tc_PC)
x_V_PCF2 = FeatureSelector2PC.transform(x_V_PC)

# Structural keys (binary)
FeatureSelector2SKbn = SelectFromModel((RandomForestRegressor(n_jobs=-1, random_state=seed).fit(x_T_SKbn, y_T_SKbn)), max_features=677, threshold=-np.inf, prefit=True)
x_T_SKbnF2 = FeatureSelector2SKbn.transform(x_T_SKbn)
x_Tc_SKbnF2 = FeatureSelector2SKbn.transform(x_Tc_SKbn)
x_V_SKbnF2 = FeatureSelector2SKbn.transform(x_V_SKbn)

# Structural keys (count)
FeatureSelector2SKct = SelectFromModel((RandomForestRegressor(n_jobs=-1, random_state=seed).fit(x_T_SKct, y_T_SKct)), max_features=677, threshold=-np.inf, prefit=True)
x_T_SKctF2 = FeatureSelector2SKct.transform(x_T_SKct)
x_Tc_SKctF2 = FeatureSelector2SKct.transform(x_Tc_SKct)
x_V_SKctF2 = FeatureSelector2SKct.transform(x_V_SKct)

print('\n50%% FEATURE SELECTION\nTRAINING SETS:\n> %s molecules with %s physicochemical descriptors' % (x_T_PCF2.shape[0], x_T_PCF2.shape[1]),
      '\n> %s molecules with %s binary structural key bits' % (x_T_SKbnF2.shape[0], x_T_SKbnF2.shape[1]),
      '\n> %s molecules with %s count structural key bits' % (x_T_SKctF2.shape[0], x_T_SKctF2.shape[1]),
      '\nCOMPOSITE TRAINING SETS:\n> %s molecules with %s physicochemical descriptors' % (x_Tc_PCF2.shape[0], x_Tc_PCF2.shape[1]),
      '\n> %s molecules with %s binary structural key bits' % (x_Tc_SKbnF2.shape[0], x_Tc_SKbnF2.shape[1]),
      '\n> %s molecules with %s count structural key bits' % (x_Tc_SKctF2.shape[0], x_Tc_SKctF2.shape[1]),
      '\nVALIDATION SETS:\n> %s molecules with %s physicochemical descriptors' % (x_V_PCF2.shape[0], x_V_PCF2.shape[1]),
      '\n> %s molecules with %s binary structural key bits' % (x_V_SKbnF2.shape[0], x_V_SKbnF2.shape[1]),
      '\n> %s molecules with %s count structural key bits'% (x_V_SKctF2.shape[0], x_V_SKctF2.shape[1]))



# FEATURE SELECTION 3: 25%
# Physicochemical descriptors
FeatureSelector3PC = SelectFromModel((RandomForestRegressor(n_jobs=-1, random_state=seed).fit(x_T_PC, y_T_PC)), max_features=360, threshold=-np.inf, prefit=True)
x_T_PCF3 = FeatureSelector3PC.transform(x_T_PC)
x_Tc_PCF3 = FeatureSelector3PC.transform(x_Tc_PC)
x_V_PCF3 = FeatureSelector3PC.transform(x_V_PC)

# Structural keys (binary)
FeatureSelector3SKbn = SelectFromModel((RandomForestRegressor(n_jobs=-1, random_state=seed).fit(x_T_SKbn, y_T_SKbn)), max_features=339, threshold=-np.inf, prefit=True)
x_T_SKbnF3 = FeatureSelector3SKbn.transform(x_T_SKbn)
x_Tc_SKbnF3 = FeatureSelector3SKbn.transform(x_Tc_SKbn)
x_V_SKbnF3 = FeatureSelector3SKbn.transform(x_V_SKbn)

# Structural keys (count)
FeatureSelector3SKct = SelectFromModel((RandomForestRegressor(n_jobs=-1, random_state=seed).fit(x_T_SKct, y_T_SKct)), max_features=339, threshold=-np.inf, prefit=True)
x_T_SKctF3 = FeatureSelector3SKct.transform(x_T_SKct)
x_Tc_SKctF3 = FeatureSelector3SKct.transform(x_Tc_SKct)
x_V_SKctF3 = FeatureSelector3SKct.transform(x_V_SKct)

print('\n25%% FEATURE SELECTION\nTRAINING SETS:\n> %s molecules with %s physicochemical descriptors' % (x_T_PCF3.shape[0], x_T_PCF3.shape[1]),
      '\n> %s molecules with %s binary structural key bits' % (x_T_SKbnF3.shape[0], x_T_SKbnF3.shape[1]),
      '\n> %s molecules with %s  count structural key bits' % (x_T_SKctF3.shape[0], x_T_SKctF3.shape[1]),
      '\nCOMPOSITE TRAINING SETS:\n> %s molecules with %s physicochemical descriptors' % (x_Tc_PCF3.shape[0], x_Tc_PCF3.shape[1]),
      '\n> %s molecules with %s binary structural key bits' % (x_Tc_SKbnF3.shape[0], x_Tc_SKbnF3.shape[1]),
      '\n> %s molecules with %s count structural key bits' % (x_Tc_SKctF3.shape[0], x_Tc_SKctF3.shape[1]),
      '\nVALIDATION SETS:\n> %s molecules with %s physicochemical descriptors' % (x_V_PCF3.shape[0], x_V_PCF3.shape[1]),
      '\n> %s molecules with %s binary structural key bits' % (x_V_SKbnF3.shape[0], x_V_SKbnF3.shape[1]),
      '\n> %s molecules with %s count structural key bits'% (x_V_SKctF3.shape[0], x_V_SKctF3.shape[1]))



# FEATURE SELECTION 4: 12.5%
# Physicochemical descriptors
FeatureSelector4PC = SelectFromModel((RandomForestRegressor(n_jobs=-1, random_state=seed).fit(x_T_PC, y_T_PC)), max_features=180, threshold=-np.inf, prefit=True)
x_T_PCF4 = FeatureSelector4PC.transform(x_T_PC)
x_Tc_PCF4 = FeatureSelector4PC.transform(x_Tc_PC)
x_V_PCF4 = FeatureSelector4PC.transform(x_V_PC)

# Structural keys (binary)
FeatureSelector4SKbn = SelectFromModel((RandomForestRegressor(n_jobs=-1, random_state=seed).fit(x_T_SKbn, y_T_SKbn)), max_features=170, threshold=-np.inf, prefit=True)
x_T_SKbnF4 = FeatureSelector4SKbn.transform(x_T_SKbn)
x_Tc_SKbnF4 = FeatureSelector4SKbn.transform(x_Tc_SKbn)
x_V_SKbnF4 = FeatureSelector4SKbn.transform(x_V_SKbn)

# Structural keys (count)
FeatureSelector4SKct = SelectFromModel((RandomForestRegressor(n_jobs=-1, random_state=seed).fit(x_T_SKct, y_T_SKct)), max_features=170, threshold=-np.inf, prefit=True)
x_T_SKctF4 = FeatureSelector4SKct.transform(x_T_SKct)
x_Tc_SKctF4 = FeatureSelector4SKct.transform(x_Tc_SKct)
x_V_SKctF4 = FeatureSelector4SKct.transform(x_V_SKct)

print('\n12.5%% FEATURE SELECTION\nTRAINING SETS:\n> %s molecules with %s physicochemical descriptors' % (x_T_PCF4.shape[0], x_T_PCF4.shape[1]),
      '\n> %s molecules with %s binary structural key bits' % (x_T_SKbnF4.shape[0], x_T_SKbnF4.shape[1]),
      '\n> %s molecules with %s count structural key bits' % (x_T_SKctF4.shape[0], x_T_SKctF4.shape[1]),
      '\nCOMPOSITE TRAINING SETS:\n> %s molecules with %s physicochemical descriptors' % (x_Tc_PCF4.shape[0], x_Tc_PCF4.shape[1]),
      '\n> %s molecules with %s binary structural key bits' % (x_Tc_SKbnF4.shape[0], x_Tc_SKbnF4.shape[1]),
      '\n> %s molecules with %s count structural key bits' % (x_Tc_SKctF4.shape[0], x_Tc_SKctF4.shape[1]),
      '\nVALIDATION SETS:\n> %s molecules with %s physicochemical descriptors' % (x_V_PCF4.shape[0], x_V_PCF4.shape[1]),
      '\n> %s molecules with %s binary structural key bits' % (x_V_SKbnF4.shape[0], x_V_SKbnF4.shape[1]),
      '\n> %s molecules with %s count structural key bits'% (x_V_SKctF4.shape[0], x_V_SKctF4.shape[1]))



print('\n(%.2fs)' % (time()-start2))



#=============================================================================
# (3) COMPOSITE REPRESENTATION GENERATION
#=============================================================================
print('\n\n\n########## COMPOSITE REPRESENTATIONS ##########')
start3 = time()



# Full composite representations
x_T_COMbn = np.hstack((x_Tc_PC, x_Tc_SKbn, x_Tc_FPbn))
x_V_COMbn = np.hstack((x_V_PC, x_V_SKbn, x_V_FPbn))
x_T_COMct = np.hstack((x_Tc_PC, x_Tc_SKct, x_Tc_FPct))
x_V_COMct = np.hstack((x_V_PC, x_V_SKct, x_V_FPct))

# 75% composite representations
x_T_COMbnF1 = np.hstack((x_Tc_PCF1, x_Tc_SKbnF1, x_Tc_FPbnF1))
x_V_COMbnF1 = np.hstack((x_V_PCF1, x_V_SKbnF1, x_V_FPbnF1))
x_T_COMctF1 = np.hstack((x_Tc_PCF1, x_Tc_SKctF1, x_Tc_FPctF1))
x_V_COMctF1 = np.hstack((x_V_PCF1, x_V_SKctF1, x_V_FPctF1))

# 50% composite representations
x_T_COMbnF2 = np.hstack((x_Tc_PCF2, x_Tc_SKbnF2, x_Tc_FPbnF2))
x_V_COMbnF2 = np.hstack((x_V_PCF2, x_V_SKbnF2, x_V_FPbnF2))
x_T_COMctF2 = np.hstack((x_Tc_PCF2, x_Tc_SKctF2, x_Tc_FPctF2))
x_V_COMctF2 = np.hstack((x_V_PCF2, x_V_SKctF2, x_V_FPctF2))

# 25% composite representations
x_T_COMbnF3 = np.hstack((x_Tc_PCF3, x_Tc_SKbnF3, x_Tc_FPbnF3))
x_V_COMbnF3 = np.hstack((x_V_PCF3, x_V_SKbnF3, x_V_FPbnF3))
x_T_COMctF3 = np.hstack((x_Tc_PCF3, x_Tc_SKctF3, x_Tc_FPctF3))
x_V_COMctF3 = np.hstack((x_V_PCF3, x_V_SKctF3, x_V_FPctF3))

# 12.5% composite representations
x_T_COMbnF4 = np.hstack((x_Tc_PCF4, x_Tc_SKbnF4, x_Tc_FPbnF4))
x_V_COMbnF4 = np.hstack((x_V_PCF4, x_V_SKbnF4, x_V_FPbnF4))
x_T_COMctF4 = np.hstack((x_Tc_PCF4, x_Tc_SKctF4, x_Tc_FPctF4))
x_V_COMctF4 = np.hstack((x_V_PCF4, x_V_SKctF4, x_V_FPctF4))

print('\nFULL COMPOSITE REPRESENTATIONS\nTRAINING SETS:\n> %s molecules with %s binary features' % (x_T_COMbn.shape[0], x_T_COMbn.shape[1]),
      '\n> %s molecules with %s count features' % (x_T_COMct.shape[0], x_T_COMct.shape[1]),
      '\nVALIDATION SETS:\n> %s molecules with %s binary features' % (x_V_COMbn.shape[0], x_V_COMbn.shape[1]),
      '\n> %s molecules with %s count features' % (x_V_COMct.shape[0], x_V_COMct.shape[1]),
      '\n\n75%% COMPOSITE REPRESENTATIONS\nTRAINING SETS:\n> %s molecules with %s  binary features' % (x_T_COMbnF1.shape[0], x_T_COMbnF1.shape[1]),
      '\n> %s molecules with %s  count selected features' % (x_T_COMctF1.shape[0], x_T_COMctF1.shape[1]),
      '\nVALIDATION SETS:\n> %s molecules with %s binary features' % (x_V_COMbnF1.shape[0], x_V_COMbnF1.shape[1]),
      '\n> %s molecules with %s count features' % (x_V_COMctF1.shape[0], x_V_COMctF1.shape[1]),
      '\n\n50%% COMPOSITE REPRESENTATIONS\nTRAINING SETS:\n> %s molecules with %s binary features' % (x_T_COMbnF2.shape[0], x_T_COMbnF2.shape[1]),
      '\n> %s molecules with %s count features' % (x_T_COMctF2.shape[0], x_T_COMctF2.shape[1]),
      '\nVALIDATION SETS:\n> %s molecules with %s binary features' % (x_V_COMbnF2.shape[0], x_V_COMbnF2.shape[1]),
      '\n> %s molecules with %s count features' % (x_V_COMctF2.shape[0], x_V_COMctF2.shape[1]),
      '\n\n25%% COMPOSITE REPRESENTATIONS\nTRAINING SETS:\n> %s molecules with %s binary features' % (x_T_COMbnF3.shape[0], x_T_COMbnF3.shape[1]),
      '\n> %s molecules with %s count features' % (x_T_COMctF3.shape[0], x_T_COMctF3.shape[1]),
      '\nVALIDATION SETS:\n> %s molecules with %s binary features' % (x_V_COMbnF3.shape[0], x_V_COMbnF3.shape[1]),
      '\n> %s molecules with %s count features' % (x_V_COMctF3.shape[0], x_V_COMctF3.shape[1]),
      '\n\n12.5%% COMPOSITE REPRESENTATIONS\nTRAINING SETS:\n> %s molecules with %s binary features' % (x_T_COMbnF4.shape[0], x_T_COMbnF4.shape[1]),
      '\n> %s molecules with %s count features' % (x_T_COMctF4.shape[0], x_T_COMctF4.shape[1]),
      '\nVALIDATION SETS:\n> %s molecules with %s binary features' % (x_V_COMbnF4.shape[0], x_V_COMbnF4.shape[1]),
      '\n> %s molecules with %s count features' % (x_V_COMctF4.shape[0], x_V_COMctF4.shape[1]))



print('\n(%.2fs)' % (time()-start3))



#=============================================================================
# (4) DATA SCALING
#=============================================================================
from sklearn.preprocessing import MinMaxScaler



print('\n\n\n########## DATA SCALING ##########')
start4 = time()



# PHYSICOCHEMICAL DESCRIPTORS
# Full representation
ScalerPC = MinMaxScaler().fit(x_T_PC)
x_T_PC = ScalerPC.transform(x_T_PC)
x_V_PC = ScalerPC.transform(x_V_PC)
# 75% feature selected
ScalerPCF1 = MinMaxScaler().fit(x_T_PCF1)
x_T_PCF1 = ScalerPCF1.transform(x_T_PCF1)
x_V_PCF1 = ScalerPCF1.transform(x_V_PCF1)
# 50% feature selected
ScalerPCF2 = MinMaxScaler().fit(x_T_PCF2)
x_T_PCF2 = ScalerPCF2.transform(x_T_PCF2)
x_V_PCF2 = ScalerPCF2.transform(x_V_PCF2)
# 25% feature selected
ScalerPCF3 = MinMaxScaler().fit(x_T_PCF3)
x_T_PCF3 = ScalerPCF3.transform(x_T_PCF3)
x_V_PCF3 = ScalerPCF3.transform(x_V_PCF3)
# 12.5% feature selected
ScalerPCF4 = MinMaxScaler().fit(x_T_PCF4)
x_T_PCF4 = ScalerPCF4.transform(x_T_PCF4)
x_V_PCF4 = ScalerPCF4.transform(x_V_PCF4)
print('> Physicochemical descriptors scaled')



# STRUCTURAL KEYS (COUNT)
# Full representation
ScalerSKct = MinMaxScaler().fit(x_T_SKct)
x_T_SKct = ScalerSKct.transform(x_T_SKct)
x_V_SKct = ScalerSKct.transform(x_V_SKct)
# 75% feature selected
ScalerSKctF1 = MinMaxScaler().fit(x_T_SKctF1)
x_T_SKctF1 = ScalerSKctF1.transform(x_T_SKctF1)
x_V_SKctF1 = ScalerSKctF1.transform(x_V_SKctF1)
# 50% feature selected
ScalerSKctF2 = MinMaxScaler().fit(x_T_SKctF2)
x_T_SKctF2 = ScalerSKctF2.transform(x_T_SKctF2)
x_V_SKctF2 = ScalerSKctF2.transform(x_V_SKctF2)
# 25% feature selected
ScalerSKctF3 = MinMaxScaler().fit(x_T_SKctF3)
x_T_SKctF3 = ScalerSKctF3.transform(x_T_SKctF3)
x_V_SKctF3 = ScalerSKctF3.transform(x_V_SKctF3)
# 12.5% feature selected
ScalerSKctF4 = MinMaxScaler().fit(x_T_SKctF4)
x_T_SKctF4 = ScalerSKctF4.transform(x_T_SKctF4)
x_V_SKctF4 = ScalerSKctF4.transform(x_V_SKctF4)
print('> Count structural keys scaled')



# CIRCULAR FINGERPRINTS (COUNT)
# Full representation
ScalerFPct = MinMaxScaler().fit(x_T_FPct)
x_T_FPct = ScalerFPct.transform(x_T_FPct)
x_V_FPct = ScalerFPct.transform(x_V_FPct)
# 75% hashed
ScalerFPctF1 = MinMaxScaler().fit(x_T_FPctF1)
x_T_FPctF1 = ScalerFPctF1.transform(x_T_FPctF1)
x_V_FPctF1 = ScalerFPctF1.transform(x_V_FPctF1)
# 50% hashed
ScalerFPctF2 = MinMaxScaler().fit(x_T_FPctF2)
x_T_FPctF2 = ScalerFPctF2.transform(x_T_FPctF2)
x_V_FPctF2 = ScalerFPctF2.transform(x_V_FPctF2)
# 25% hashed
ScalerFPctF3 = MinMaxScaler().fit(x_T_FPctF3)
x_T_FPctF3 = ScalerFPctF3.transform(x_T_FPctF3)
x_V_FPctF3 = ScalerFPctF3.transform(x_V_FPctF3)
# 12.5% hashed
ScalerFPctF4 = MinMaxScaler().fit(x_T_FPctF4)
x_T_FPctF4 = ScalerFPctF4.transform(x_T_FPctF4)
x_V_FPctF4 = ScalerFPctF4.transform(x_V_FPctF4)
print('> Count circular fingerprints scaled')



# COMPOSITE REPRESENTATION (BINARY)
# Full representation
ScalerCOMbn = MinMaxScaler().fit(x_T_COMbn)
x_T_COMbn = ScalerCOMbn.transform(x_T_COMbn)
x_V_COMbn = ScalerCOMbn.transform(x_V_COMbn)
# 75% feature selected
ScalerCOMbnF1 = MinMaxScaler().fit(x_T_COMbnF1)
x_T_COMbnF1 = ScalerCOMbnF1.transform(x_T_COMbnF1)
x_V_COMbnF1 = ScalerCOMbnF1.transform(x_V_COMbnF1)
# 50% feature selected
ScalerCOMbnF2 = MinMaxScaler().fit(x_T_COMbnF2)
x_T_COMbnF2 = ScalerCOMbnF2.transform(x_T_COMbnF2)
x_V_COMbnF2 = ScalerCOMbnF2.transform(x_V_COMbnF2)
# 25% feature selected
ScalerCOMbnF3 = MinMaxScaler().fit(x_T_COMbnF3)
x_T_COMbnF3 = ScalerCOMbnF3.transform(x_T_COMbnF3)
x_V_COMbnF3 = ScalerCOMbnF3.transform(x_V_COMbnF3)
# 12.5% feature selected
ScalerCOMbnF4 = MinMaxScaler().fit(x_T_COMbnF4)
x_T_COMbnF4 = ScalerCOMbnF4.transform(x_T_COMbnF4)
x_V_COMbnF4 = ScalerCOMbnF4.transform(x_V_COMbnF4)
print('> Binary composite representations scaled')



# COMPOSITE REPRESENTATION (COUNT)
# Full representation
ScalerCOMct = MinMaxScaler().fit(x_T_COMct)
x_T_COMct = ScalerCOMct.transform(x_T_COMct)
x_V_COMct = ScalerCOMct.transform(x_V_COMct)
# 75% feature selected
ScalerCOMctF1 = MinMaxScaler().fit(x_T_COMctF1)
x_T_COMctF1 = ScalerCOMctF1.transform(x_T_COMctF1)
x_V_COMctF1 = ScalerCOMctF1.transform(x_V_COMctF1)
# 50% feature selected
ScalerCOMctF2 = MinMaxScaler().fit(x_T_COMctF2)
x_T_COMctF2 = ScalerCOMctF2.transform(x_T_COMctF2)
x_V_COMctF2 = ScalerCOMctF2.transform(x_V_COMctF2)
# 25% feature selected
ScalerCOMctF3 = MinMaxScaler().fit(x_T_COMctF3)
x_T_COMctF3 = ScalerCOMctF3.transform(x_T_COMctF3)
x_V_COMctF3 = ScalerCOMctF3.transform(x_V_COMctF3)
# 12.5% feature selected
ScalerCOMctF4 = MinMaxScaler().fit(x_T_COMctF4)
x_T_COMctF4 = ScalerCOMctF4.transform(x_T_COMctF4)
x_V_COMctF4 = ScalerCOMctF4.transform(x_V_COMctF4)
print('> Count composite representations scaled')



print('\n(%.2fs)' % (time()-start4))


'''
#=============================================================================
# (5) PARAMETER OPTIMISATION
#=============================================================================
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso, Ridge, ElasticNet, SGDRegressor



print('\n\n\n########## PARAMETER OPTIMISATION ##########')
start5 = time()
# n_jobs set to 1 instead of -1 to avoid "Userwarning: A worker stopped while some jobs were given to the executor. This can be caused by a too short worker timeout or by a memory leak."



# Lasso (l1-regularised) multilinear regression
lasso = Lasso(max_iter=3000, random_state=seed)
lasso_paramgrid = {'alpha': [0.001, 0.1, 1, 10], # Difficulty converging at alpha = 0.0001
		   'selection': ['cyclic', 'random']}

start5a = time()

LassoPC_params   = GridSearchCV(estimator=lasso, param_grid=lasso_paramgrid, cv=3, refit=True, n_jobs=1).fit(x_T_PC, y_T_PC).best_params_
LassoPCF1_params = GridSearchCV(estimator=lasso, param_grid=lasso_paramgrid, cv=3, refit=True, n_jobs=1).fit(x_T_PCF1, y_T_PC).best_params_
LassoPCF2_params = GridSearchCV(estimator=lasso, param_grid=lasso_paramgrid, cv=3, refit=True, n_jobs=1).fit(x_T_PCF2, y_T_PC).best_params_
LassoPCF3_params = GridSearchCV(estimator=lasso, param_grid=lasso_paramgrid, cv=3, refit=True, n_jobs=1).fit(x_T_PCF3, y_T_PC).best_params_
LassoPCF4_params = GridSearchCV(estimator=lasso, param_grid=lasso_paramgrid, cv=3, refit=True, n_jobs=1).fit(x_T_PCF4, y_T_PC).best_params_

LassoSKbn_params   = GridSearchCV(estimator=lasso, param_grid=lasso_paramgrid, cv=3, refit=True, n_jobs=1).fit(x_T_SKbn, y_T_SKbn).best_params_
LassoSKbnF1_params = GridSearchCV(estimator=lasso, param_grid=lasso_paramgrid, cv=3, refit=True, n_jobs=1).fit(x_T_SKbnF1, y_T_SKbn).best_params_
LassoSKbnF2_params = GridSearchCV(estimator=lasso, param_grid=lasso_paramgrid, cv=3, refit=True, n_jobs=1).fit(x_T_SKbnF2, y_T_SKbn).best_params_
LassoSKbnF3_params = GridSearchCV(estimator=lasso, param_grid=lasso_paramgrid, cv=3, refit=True, n_jobs=1).fit(x_T_SKbnF3, y_T_SKbn).best_params_
LassoSKbnF4_params = GridSearchCV(estimator=lasso, param_grid=lasso_paramgrid, cv=3, refit=True, n_jobs=1).fit(x_T_SKbnF4, y_T_SKbn).best_params_

LassoSKct_params   = GridSearchCV(estimator=lasso, param_grid=lasso_paramgrid, cv=3, refit=True, n_jobs=1).fit(x_T_SKct, y_T_SKct).best_params_
LassoSKctF1_params = GridSearchCV(estimator=lasso, param_grid=lasso_paramgrid, cv=3, refit=True, n_jobs=1).fit(x_T_SKctF1, y_T_SKct).best_params_
LassoSKctF2_params = GridSearchCV(estimator=lasso, param_grid=lasso_paramgrid, cv=3, refit=True, n_jobs=1).fit(x_T_SKctF2, y_T_SKct).best_params_
LassoSKctF3_params = GridSearchCV(estimator=lasso, param_grid=lasso_paramgrid, cv=3, refit=True, n_jobs=1).fit(x_T_SKctF3, y_T_SKct).best_params_
LassoSKctF4_params = GridSearchCV(estimator=lasso, param_grid=lasso_paramgrid, cv=3, refit=True, n_jobs=1).fit(x_T_SKctF4, y_T_SKct).best_params_

LassoFPbn_params   = GridSearchCV(estimator=lasso, param_grid=lasso_paramgrid, cv=3, refit=True, n_jobs=1).fit(x_T_FPbn, y_T_FPbn).best_params_
LassoFPbnF1_params = GridSearchCV(estimator=lasso, param_grid=lasso_paramgrid, cv=3, refit=True, n_jobs=1).fit(x_T_FPbnF1, y_T_FPbn).best_params_
LassoFPbnF2_params = GridSearchCV(estimator=lasso, param_grid=lasso_paramgrid, cv=3, refit=True, n_jobs=1).fit(x_T_FPbnF2, y_T_FPbn).best_params_
LassoFPbnF3_params = GridSearchCV(estimator=lasso, param_grid=lasso_paramgrid, cv=3, refit=True, n_jobs=1).fit(x_T_FPbnF3, y_T_FPbn).best_params_
LassoFPbnF4_params = GridSearchCV(estimator=lasso, param_grid=lasso_paramgrid, cv=3, refit=True, n_jobs=1).fit(x_T_FPbnF4, y_T_FPbn).best_params_

LassoFPct_params   = GridSearchCV(estimator=lasso, param_grid=lasso_paramgrid, cv=3, refit=True, n_jobs=1).fit(x_T_FPct, y_T_FPct).best_params_
LassoFPctF1_params = GridSearchCV(estimator=lasso, param_grid=lasso_paramgrid, cv=3, refit=True, n_jobs=1).fit(x_T_FPctF1, y_T_FPct).best_params_
LassoFPctF2_params = GridSearchCV(estimator=lasso, param_grid=lasso_paramgrid, cv=3, refit=True, n_jobs=1).fit(x_T_FPctF2, y_T_FPct).best_params_
LassoFPctF3_params = GridSearchCV(estimator=lasso, param_grid=lasso_paramgrid, cv=3, refit=True, n_jobs=1).fit(x_T_FPctF3, y_T_FPct).best_params_
LassoFPctF4_params = GridSearchCV(estimator=lasso, param_grid=lasso_paramgrid, cv=3, refit=True, n_jobs=1).fit(x_T_FPctF4, y_T_FPct).best_params_

LassoCOMbn_params   = GridSearchCV(estimator=lasso, param_grid=lasso_paramgrid, cv=3, refit=True, n_jobs=1).fit(x_T_COMbn, y_Tc).best_params_
LassoCOMbnF1_params = GridSearchCV(estimator=lasso, param_grid=lasso_paramgrid, cv=3, refit=True, n_jobs=1).fit(x_T_COMbnF1, y_Tc).best_params_
LassoCOMbnF2_params = GridSearchCV(estimator=lasso, param_grid=lasso_paramgrid, cv=3, refit=True, n_jobs=1).fit(x_T_COMbnF2, y_Tc).best_params_
LassoCOMbnF3_params = GridSearchCV(estimator=lasso, param_grid=lasso_paramgrid, cv=3, refit=True, n_jobs=1).fit(x_T_COMbnF3, y_Tc).best_params_
LassoCOMbnF4_params = GridSearchCV(estimator=lasso, param_grid=lasso_paramgrid, cv=3, refit=True, n_jobs=1).fit(x_T_COMbnF4, y_Tc).best_params_

LassoCOMct_params   = GridSearchCV(estimator=lasso, param_grid=lasso_paramgrid, cv=3, refit=True, n_jobs=1).fit(x_T_COMct, y_Tc).best_params_
LassoCOMctF1_params = GridSearchCV(estimator=lasso, param_grid=lasso_paramgrid, cv=3, refit=True, n_jobs=1).fit(x_T_COMctF1, y_Tc).best_params_
LassoCOMctF2_params = GridSearchCV(estimator=lasso, param_grid=lasso_paramgrid, cv=3, refit=True, n_jobs=1).fit(x_T_COMctF2, y_Tc).best_params_
LassoCOMctF3_params = GridSearchCV(estimator=lasso, param_grid=lasso_paramgrid, cv=3, refit=True, n_jobs=1).fit(x_T_COMctF3, y_Tc).best_params_
LassoCOMctF4_params = GridSearchCV(estimator=lasso, param_grid=lasso_paramgrid, cv=3, refit=True, n_jobs=1).fit(x_T_COMctF4, y_Tc).best_params_

print('\nLASSO GRID SEARCH:',
      '\n> LassoPC:',LassoPC_params, '\n> LassoPCF1:',LassoPCF1_params, '\n> LassoPCF2:',LassoPCF2_params, '\n> LassoPCF3:',LassoPCF3_params, '\n> LassoPCF4:',LassoPCF4_params,
      '\n> LassoSKbn:',LassoSKbn_params, '\n> LassoSKbnF1:',LassoSKbnF1_params, '\n> LassoSKbnF2:',LassoSKbnF2_params, '\n> LassoSKbnF3:',LassoSKbnF3_params, '\n> LassoSKbnF4:',LassoSKbnF4_params,
      '\n> LassoSKct:',LassoSKct_params, '\n> LassoSKctF1:',LassoSKctF1_params, '\n> LassoSKctF2:',LassoSKctF2_params, '\n> LassoSKctF3:',LassoSKctF3_params, '\n> LassoSKctF4:',LassoSKctF4_params,
      '\n> LassoFPbn:',LassoFPbn_params, '\n> LassoFPbnF1:',LassoFPbnF1_params, '\n> LassoFPbnF2:',LassoFPbnF2_params, '\n> LassoFPbnF3:',LassoFPbnF3_params, '\n> LassoFPbnF4:',LassoFPbnF4_params,
      '\n> LassoFPct:',LassoFPct_params, '\n> LassoFPctF1:',LassoFPctF1_params, '\n> LassoFPctF2:',LassoFPctF2_params, '\n> LassoFPctF3:',LassoFPctF3_params, '\n> LassoFPctF4:',LassoFPctF4_params,
      '\n> LassoCOMbn:',LassoCOMbn_params, '\n> LassoCOMbnF1:',LassoCOMbnF1_params, '\n> LassoCOMbnF2:',LassoCOMbnF2_params, '\n> LassoCOMbnF3:',LassoCOMbnF3_params, '\n> LassoCOMbnF4:',LassoCOMbnF4_params,
      '\n> LassoCOMct:',LassoCOMct_params, '\n> LassoCOMctF1:',LassoCOMctF1_params, '\n> LassoCOMctF2:',LassoCOMctF2_params, '\n> LassoCOMctF3:',LassoCOMctF3_params, '\n> LassoCOMctF4:',LassoCOMctF4_params)
print('(%.2fs)' % (time()-start5a))

#   >>> Output
#   LASSO GRID SEARCH: 
#   > LassoPC: {'alpha': 0.001, 'selection': 'random'} 
#   > LassoPCF1: {'alpha': 0.001, 'selection': 'cyclic'} 
#   > LassoPCF2: {'alpha': 0.001, 'selection': 'cyclic'} 
#   > LassoPCF3: {'alpha': 0.001, 'selection': 'random'} 
#   > LassoPCF4: {'alpha': 0.001, 'selection': 'cyclic'} 
#   > LassoSKbn: {'alpha': 0.001, 'selection': 'cyclic'} 
#   > LassoSKbnF1: {'alpha': 0.001, 'selection': 'cyclic'} 
#   > LassoSKbnF2: {'alpha': 0.001, 'selection': 'cyclic'} 
#   > LassoSKbnF3: {'alpha': 0.001, 'selection': 'cyclic'} 
#   > LassoSKbnF4: {'alpha': 0.001, 'selection': 'cyclic'} 
#   > LassoSKct: {'alpha': 0.001, 'selection': 'cyclic'} 
#   > LassoSKctF1: {'alpha': 0.001, 'selection': 'cyclic'} 
#   > LassoSKctF2: {'alpha': 0.001, 'selection': 'cyclic'} 
#   > LassoSKctF3: {'alpha': 0.001, 'selection': 'random'} 
#   > LassoSKctF4: {'alpha': 0.001, 'selection': 'random'} 
#   > LassoFPbn: {'alpha': 0.001, 'selection': 'random'} 
#   > LassoFPbnF1: {'alpha': 0.001, 'selection': 'random'} 
#   > LassoFPbnF2: {'alpha': 0.001, 'selection': 'random'} 
#   > LassoFPbnF3: {'alpha': 0.001, 'selection': 'random'} 
#   > LassoFPbnF4: {'alpha': 0.001, 'selection': 'cyclic'} 
#   > LassoFPct: {'alpha': 0.001, 'selection': 'cyclic'} 
#   > LassoFPctF1: {'alpha': 0.001, 'selection': 'random'} 
#   > LassoFPctF2: {'alpha': 0.001, 'selection': 'cyclic'} 
#   > LassoFPctF3: {'alpha': 0.001, 'selection': 'random'} 
#   > LassoFPctF4: {'alpha': 0.001, 'selection': 'random'} 
#   > LassoCOMbn: {'alpha': 0.001, 'selection': 'cyclic'} 
#   > LassoCOMbnF1: {'alpha': 0.001, 'selection': 'random'} 
#   > LassoCOMbnF2: {'alpha': 0.001, 'selection': 'cyclic'} 
#   > LassoCOMbnF3: {'alpha': 0.001, 'selection': 'cyclic'} 
#   > LassoCOMbnF4: {'alpha': 0.001, 'selection': 'cyclic'} 
#   > LassoCOMct: {'alpha': 0.001, 'selection': 'cyclic'} 
#   > LassoCOMctF1: {'alpha': 0.001, 'selection': 'random'} 
#   > LassoCOMctF2: {'alpha': 0.001, 'selection': 'random'} 
#   > LassoCOMctF3: {'alpha': 0.001, 'selection': 'cyclic'} 
#   > LassoCOMctF4: {'alpha': 0.001, 'selection': 'cyclic'}
#   (1459.86s)



# Ridge (l2-regularised) multilinear regression
ridge = Ridge(random_state=seed)
ridge_paramgrid = {'alpha': [0.001, 0.1, 1, 10, 100],
		   'solver': ['auto','svd','cholesky','lsqr','sparse_cg','sag','saga']}

start5b = time()

RidgePC_params   = GridSearchCV(estimator=ridge, param_grid=ridge_paramgrid, cv=3, refit=True, n_jobs=1).fit(x_T_PC, y_T_PC).best_params_
RidgePCF1_params = GridSearchCV(estimator=ridge, param_grid=ridge_paramgrid, cv=3, refit=True, n_jobs=1).fit(x_T_PCF1, y_T_PC).best_params_
RidgePCF2_params = GridSearchCV(estimator=ridge, param_grid=ridge_paramgrid, cv=3, refit=True, n_jobs=1).fit(x_T_PCF2, y_T_PC).best_params_
RidgePCF3_params = GridSearchCV(estimator=ridge, param_grid=ridge_paramgrid, cv=3, refit=True, n_jobs=1).fit(x_T_PCF3, y_T_PC).best_params_
RidgePCF4_params = GridSearchCV(estimator=ridge, param_grid=ridge_paramgrid, cv=3, refit=True, n_jobs=1).fit(x_T_PCF4, y_T_PC).best_params_

RidgeSKbn_params   = GridSearchCV(estimator=ridge, param_grid=ridge_paramgrid, cv=3, refit=True, n_jobs=1).fit(x_T_SKbn, y_T_SKbn).best_params_
RidgeSKbnF1_params = GridSearchCV(estimator=ridge, param_grid=ridge_paramgrid, cv=3, refit=True, n_jobs=1).fit(x_T_SKbnF1, y_T_SKbn).best_params_
RidgeSKbnF2_params = GridSearchCV(estimator=ridge, param_grid=ridge_paramgrid, cv=3, refit=True, n_jobs=1).fit(x_T_SKbnF2, y_T_SKbn).best_params_
RidgeSKbnF3_params = GridSearchCV(estimator=ridge, param_grid=ridge_paramgrid, cv=3, refit=True, n_jobs=1).fit(x_T_SKbnF3, y_T_SKbn).best_params_
RidgeSKbnF4_params = GridSearchCV(estimator=ridge, param_grid=ridge_paramgrid, cv=3, refit=True, n_jobs=1).fit(x_T_SKbnF4, y_T_SKbn).best_params_

RidgeSKct_params   = GridSearchCV(estimator=ridge, param_grid=ridge_paramgrid, cv=3, refit=True, n_jobs=1).fit(x_T_SKct, y_T_SKct).best_params_
RidgeSKctF1_params = GridSearchCV(estimator=ridge, param_grid=ridge_paramgrid, cv=3, refit=True, n_jobs=1).fit(x_T_SKctF1, y_T_SKct).best_params_
RidgeSKctF2_params = GridSearchCV(estimator=ridge, param_grid=ridge_paramgrid, cv=3, refit=True, n_jobs=1).fit(x_T_SKctF2, y_T_SKct).best_params_
RidgeSKctF3_params = GridSearchCV(estimator=ridge, param_grid=ridge_paramgrid, cv=3, refit=True, n_jobs=1).fit(x_T_SKctF3, y_T_SKct).best_params_
RidgeSKctF4_params = GridSearchCV(estimator=ridge, param_grid=ridge_paramgrid, cv=3, refit=True, n_jobs=1).fit(x_T_SKctF4, y_T_SKct).best_params_

RidgeFPbn_params   = GridSearchCV(estimator=ridge, param_grid=ridge_paramgrid, cv=3, refit=True, n_jobs=1).fit(x_T_FPbn, y_T_FPbn).best_params_
RidgeFPbnF1_params = GridSearchCV(estimator=ridge, param_grid=ridge_paramgrid, cv=3, refit=True, n_jobs=1).fit(x_T_FPbnF1, y_T_FPbn).best_params_
RidgeFPbnF2_params = GridSearchCV(estimator=ridge, param_grid=ridge_paramgrid, cv=3, refit=True, n_jobs=1).fit(x_T_FPbnF2, y_T_FPbn).best_params_
RidgeFPbnF3_params = GridSearchCV(estimator=ridge, param_grid=ridge_paramgrid, cv=3, refit=True, n_jobs=1).fit(x_T_FPbnF3, y_T_FPbn).best_params_
RidgeFPbnF4_params = GridSearchCV(estimator=ridge, param_grid=ridge_paramgrid, cv=3, refit=True, n_jobs=1).fit(x_T_FPbnF4, y_T_FPbn).best_params_

RidgeFPct_params   = GridSearchCV(estimator=ridge, param_grid=ridge_paramgrid, cv=3, refit=True, n_jobs=1).fit(x_T_FPct, y_T_FPct).best_params_
RidgeFPctF1_params = GridSearchCV(estimator=ridge, param_grid=ridge_paramgrid, cv=3, refit=True, n_jobs=1).fit(x_T_FPctF1, y_T_FPct).best_params_
RidgeFPctF2_params = GridSearchCV(estimator=ridge, param_grid=ridge_paramgrid, cv=3, refit=True, n_jobs=1).fit(x_T_FPctF2, y_T_FPct).best_params_
RidgeFPctF3_params = GridSearchCV(estimator=ridge, param_grid=ridge_paramgrid, cv=3, refit=True, n_jobs=1).fit(x_T_FPctF3, y_T_FPct).best_params_
RidgeFPctF4_params = GridSearchCV(estimator=ridge, param_grid=ridge_paramgrid, cv=3, refit=True, n_jobs=1).fit(x_T_FPctF4, y_T_FPct).best_params_

RidgeCOMbn_params   = GridSearchCV(estimator=ridge, param_grid=ridge_paramgrid, cv=3, refit=True, n_jobs=1).fit(x_T_COMbn, y_Tc).best_params_
RidgeCOMbnF1_params = GridSearchCV(estimator=ridge, param_grid=ridge_paramgrid, cv=3, refit=True, n_jobs=1).fit(x_T_COMbnF1, y_Tc).best_params_
RidgeCOMbnF2_params = GridSearchCV(estimator=ridge, param_grid=ridge_paramgrid, cv=3, refit=True, n_jobs=1).fit(x_T_COMbnF2, y_Tc).best_params_
RidgeCOMbnF3_params = GridSearchCV(estimator=ridge, param_grid=ridge_paramgrid, cv=3, refit=True, n_jobs=1).fit(x_T_COMbnF3, y_Tc).best_params_
RidgeCOMbnF4_params = GridSearchCV(estimator=ridge, param_grid=ridge_paramgrid, cv=3, refit=True, n_jobs=1).fit(x_T_COMbnF4, y_Tc).best_params_

RidgeCOMct_params   = GridSearchCV(estimator=ridge, param_grid=ridge_paramgrid, cv=3, refit=True, n_jobs=1).fit(x_T_COMct, y_Tc).best_params_
RidgeCOMctF1_params = GridSearchCV(estimator=ridge, param_grid=ridge_paramgrid, cv=3, refit=True, n_jobs=1).fit(x_T_COMctF1, y_Tc).best_params_
RidgeCOMctF2_params = GridSearchCV(estimator=ridge, param_grid=ridge_paramgrid, cv=3, refit=True, n_jobs=1).fit(x_T_COMctF2, y_Tc).best_params_
RidgeCOMctF3_params = GridSearchCV(estimator=ridge, param_grid=ridge_paramgrid, cv=3, refit=True, n_jobs=1).fit(x_T_COMctF3, y_Tc).best_params_
RidgeCOMctF4_params = GridSearchCV(estimator=ridge, param_grid=ridge_paramgrid, cv=3, refit=True, n_jobs=1).fit(x_T_COMctF4, y_Tc).best_params_

print('\nRIDGE GRID SEARCH:',
      '\n> RidgePC:',RidgePC_params, '\n> RidgePCF1:',RidgePCF1_params, '\n> RidgePCF2:',RidgePCF2_params, '\n> RidgePCF3:',RidgePCF3_params, '\n> RidgePCF4:',RidgePCF4_params,
      '\n> RidgeSKbn:',RidgeSKbn_params, '\n> RidgeSKbnF1:',RidgeSKbnF1_params, '\n> RidgeSKbnF2:',RidgeSKbnF2_params, '\n> RidgeSKbnF3:',RidgeSKbnF3_params, '\n> RidgeSKbnF4:',RidgeSKbnF4_params,
      '\n> RidgeSKct:',RidgeSKct_params, '\n> RidgeSKctF1:',RidgeSKctF1_params, '\n> RidgeSKctF2:',RidgeSKctF2_params, '\n> RidgeSKctF3:',RidgeSKctF3_params, '\n> RidgeSKctF4:',RidgeSKctF4_params,
      '\n> RidgeFPbn:',RidgeFPbn_params, '\n> RidgeFPbnF1:',RidgeFPbnF1_params, '\n> RidgeFPbnF2:',RidgeFPbnF2_params, '\n> RidgeFPbnF3:',RidgeFPbnF3_params, '\n> RidgeFPbnF4:',RidgeFPbnF4_params,
      '\n> RidgeFPct:',RidgeFPct_params, '\n> RidgeFPctF1:',RidgeFPctF1_params, '\n> RidgeFPctF2:',RidgeFPctF2_params, '\n> RidgeFPctF3:',RidgeFPctF3_params, '\n> RidgeFPctF4:',RidgeFPctF4_params,
      '\n> RidgeCOMbn:',RidgeCOMbn_params, '\n> RidgeCOMbnF1:',RidgeCOMbnF1_params, '\n> RidgeCOMbnF2:',RidgeCOMbnF2_params, '\n> RidgeCOMbnF3:',RidgeCOMbnF3_params, '\n> RidgeCOMbnF4:',RidgeCOMbnF4_params,
      '\n> RidgeCOMct:',RidgeCOMct_params, '\n> RidgeCOMctF1:',RidgeCOMctF1_params, '\n> RidgeCOMctF2:',RidgeCOMctF2_params, '\n> RidgeCOMctF3:',RidgeCOMctF3_params, '\n> RidgeCOMctF4:',RidgeCOMctF4_params)
print('(%.2fs)' % (time()-start5b))

#   >>> Output
#   RIDGE GRID SEARCH: 
#   > RidgePC: {'alpha': 0.1, 'solver': 'sag'} 
#   > RidgePCF1: {'alpha': 0.1, 'solver': 'auto'} 
#   > RidgePCF2: {'alpha': 0.1, 'solver': 'auto'} 
#   > RidgePCF3: {'alpha': 10, 'solver': 'svd'} 
#   > RidgePCF4: {'alpha': 10, 'solver': 'sag'} 
#   > RidgeSKbn: {'alpha': 1, 'solver': 'saga'} 
#   > RidgeSKbnF1: {'alpha': 1, 'solver': 'lsqr'} 
#   > RidgeSKbnF2: {'alpha': 1, 'solver': 'saga'} 
#   > RidgeSKbnF3: {'alpha': 1, 'solver': 'saga'} 
#   > RidgeSKbnF4: {'alpha': 1, 'solver': 'sparse_cg'} 
#   > RidgeSKct: {'alpha': 1, 'solver': 'auto'} 
#   > RidgeSKctF1: {'alpha': 0.1, 'solver': 'sag'} 
#   > RidgeSKctF2: {'alpha': 0.1, 'solver': 'sag'} 
#   > RidgeSKctF3: {'alpha': 0.001, 'solver': 'auto'} 
#   > RidgeSKctF4: {'alpha': 0.001, 'solver': 'sag'} 
#   > RidgeFPbn: {'alpha': 10, 'solver': 'lsqr'} 
#   > RidgeFPbnF1: {'alpha': 10, 'solver': 'saga'} 
#   > RidgeFPbnF2: {'alpha': 100, 'solver': 'saga'} 
#   > RidgeFPbnF3: {'alpha': 100, 'solver': 'saga'} 
#   > RidgeFPbnF4: {'alpha': 100, 'solver': 'saga'} 
#   > RidgeFPct: {'alpha': 1, 'solver': 'saga'} 
#   > RidgeFPctF1: {'alpha': 1, 'solver': 'lsqr'} 
#   > RidgeFPctF2: {'alpha': 1, 'solver': 'sparse_cg'} 
#   > RidgeFPctF3: {'alpha': 1, 'solver': 'lsqr'} 
#   > RidgeFPctF4: {'alpha': 10, 'solver': 'sparse_cg'} 
#   > RidgeCOMbn: {'alpha': 1, 'solver': 'auto'} 
#   > RidgeCOMbnF1: {'alpha': 1, 'solver': 'auto'} 
#   > RidgeCOMbnF2: {'alpha': 1, 'solver': 'svd'} 
#   > RidgeCOMbnF3: {'alpha': 1, 'solver': 'saga'} 
#   > RidgeCOMbnF4: {'alpha': 10, 'solver': 'lsqr'} 
#   > RidgeCOMct: {'alpha': 1, 'solver': 'svd'} 
#   > RidgeCOMctF1: {'alpha': 1, 'solver': 'svd'} 
#   > RidgeCOMctF2: {'alpha': 1, 'solver': 'svd'} 
#   > RidgeCOMctF3: {'alpha': 0.1, 'solver': 'auto'} 
#   > RidgeCOMctF4: {'alpha': 1, 'solver': 'auto'}
#   (24621.41s)



# ElasticNet (l1/l2-regularised) multilinear regression
elasticnet = ElasticNet(max_iter=5000, selection='random', random_state=seed)
elasticnet_paramgrid = {'alpha':[0.001, 0.1, 1],
			'l1_ratio':[0.1,0.3,0.5,0.7,0.9]}

start5c = time()

ElasticNetPC_params   = GridSearchCV(estimator=elasticnet, param_grid=elasticnet_paramgrid, cv=3, refit=True, n_jobs=1).fit(x_T_PC, y_T_PC).best_params_
ElasticNetPCF1_params = GridSearchCV(estimator=elasticnet, param_grid=elasticnet_paramgrid, cv=3, refit=True, n_jobs=1).fit(x_T_PCF1, y_T_PC).best_params_
ElasticNetPCF2_params = GridSearchCV(estimator=elasticnet, param_grid=elasticnet_paramgrid, cv=3, refit=True, n_jobs=1).fit(x_T_PCF2, y_T_PC).best_params_
ElasticNetPCF3_params = GridSearchCV(estimator=elasticnet, param_grid=elasticnet_paramgrid, cv=3, refit=True, n_jobs=1).fit(x_T_PCF3, y_T_PC).best_params_
ElasticNetPCF4_params = GridSearchCV(estimator=elasticnet, param_grid=elasticnet_paramgrid, cv=3, refit=True, n_jobs=1).fit(x_T_PCF4, y_T_PC).best_params_

ElasticNetSKbn_params   = GridSearchCV(estimator=elasticnet, param_grid=elasticnet_paramgrid, cv=3, refit=True, n_jobs=1).fit(x_T_SKbn, y_T_SKbn).best_params_
ElasticNetSKbnF1_params = GridSearchCV(estimator=elasticnet, param_grid=elasticnet_paramgrid, cv=3, refit=True, n_jobs=1).fit(x_T_SKbnF1, y_T_SKbn).best_params_
ElasticNetSKbnF2_params = GridSearchCV(estimator=elasticnet, param_grid=elasticnet_paramgrid, cv=3, refit=True, n_jobs=1).fit(x_T_SKbnF2, y_T_SKbn).best_params_
ElasticNetSKbnF3_params = GridSearchCV(estimator=elasticnet, param_grid=elasticnet_paramgrid, cv=3, refit=True, n_jobs=1).fit(x_T_SKbnF3, y_T_SKbn).best_params_
ElasticNetSKbnF4_params = GridSearchCV(estimator=elasticnet, param_grid=elasticnet_paramgrid, cv=3, refit=True, n_jobs=1).fit(x_T_SKbnF4, y_T_SKbn).best_params_

ElasticNetSKct_params   = GridSearchCV(estimator=elasticnet, param_grid=elasticnet_paramgrid, cv=3, refit=True, n_jobs=1).fit(x_T_SKct, y_T_SKct).best_params_
ElasticNetSKctF1_params = GridSearchCV(estimator=elasticnet, param_grid=elasticnet_paramgrid, cv=3, refit=True, n_jobs=1).fit(x_T_SKctF1, y_T_SKct).best_params_
ElasticNetSKctF2_params = GridSearchCV(estimator=elasticnet, param_grid=elasticnet_paramgrid, cv=3, refit=True, n_jobs=1).fit(x_T_SKctF2, y_T_SKct).best_params_
ElasticNetSKctF3_params = GridSearchCV(estimator=elasticnet, param_grid=elasticnet_paramgrid, cv=3, refit=True, n_jobs=1).fit(x_T_SKctF3, y_T_SKct).best_params_
ElasticNetSKctF4_params = GridSearchCV(estimator=elasticnet, param_grid=elasticnet_paramgrid, cv=3, refit=True, n_jobs=1).fit(x_T_SKctF4, y_T_SKct).best_params_

ElasticNetFPbn_params   = GridSearchCV(estimator=elasticnet, param_grid=elasticnet_paramgrid, cv=3, refit=True, n_jobs=1).fit(x_T_FPbn, y_T_FPbn).best_params_
ElasticNetFPbnF1_params = GridSearchCV(estimator=elasticnet, param_grid=elasticnet_paramgrid, cv=3, refit=True, n_jobs=1).fit(x_T_FPbnF1, y_T_FPbn).best_params_
ElasticNetFPbnF2_params = GridSearchCV(estimator=elasticnet, param_grid=elasticnet_paramgrid, cv=3, refit=True, n_jobs=1).fit(x_T_FPbnF2, y_T_FPbn).best_params_
ElasticNetFPbnF3_params = GridSearchCV(estimator=elasticnet, param_grid=elasticnet_paramgrid, cv=3, refit=True, n_jobs=1).fit(x_T_FPbnF3, y_T_FPbn).best_params_
ElasticNetFPbnF4_params = GridSearchCV(estimator=elasticnet, param_grid=elasticnet_paramgrid, cv=3, refit=True, n_jobs=1).fit(x_T_FPbnF4, y_T_FPbn).best_params_

ElasticNetFPct_params   = GridSearchCV(estimator=elasticnet, param_grid=elasticnet_paramgrid, cv=3, refit=True, n_jobs=1).fit(x_T_FPct, y_T_FPct).best_params_
ElasticNetFPctF1_params = GridSearchCV(estimator=elasticnet, param_grid=elasticnet_paramgrid, cv=3, refit=True, n_jobs=1).fit(x_T_FPctF1, y_T_FPct).best_params_
ElasticNetFPctF2_params = GridSearchCV(estimator=elasticnet, param_grid=elasticnet_paramgrid, cv=3, refit=True, n_jobs=1).fit(x_T_FPctF2, y_T_FPct).best_params_
ElasticNetFPctF3_params = GridSearchCV(estimator=elasticnet, param_grid=elasticnet_paramgrid, cv=3, refit=True, n_jobs=1).fit(x_T_FPctF3, y_T_FPct).best_params_
ElasticNetFPctF4_params = GridSearchCV(estimator=elasticnet, param_grid=elasticnet_paramgrid, cv=3, refit=True, n_jobs=1).fit(x_T_FPctF4, y_T_FPct).best_params_

ElasticNetCOMbn_params   = GridSearchCV(estimator=elasticnet, param_grid=elasticnet_paramgrid, cv=3, refit=True, n_jobs=1).fit(x_T_COMbn, y_Tc).best_params_
ElasticNetCOMbnF1_params = GridSearchCV(estimator=elasticnet, param_grid=elasticnet_paramgrid, cv=3, refit=True, n_jobs=1).fit(x_T_COMbnF1, y_Tc).best_params_
ElasticNetCOMbnF2_params = GridSearchCV(estimator=elasticnet, param_grid=elasticnet_paramgrid, cv=3, refit=True, n_jobs=1).fit(x_T_COMbnF2, y_Tc).best_params_
ElasticNetCOMbnF3_params = GridSearchCV(estimator=elasticnet, param_grid=elasticnet_paramgrid, cv=3, refit=True, n_jobs=1).fit(x_T_COMbnF3, y_Tc).best_params_
ElasticNetCOMbnF4_params = GridSearchCV(estimator=elasticnet, param_grid=elasticnet_paramgrid, cv=3, refit=True, n_jobs=1).fit(x_T_COMbnF4, y_Tc).best_params_

ElasticNetCOMct_params   = GridSearchCV(estimator=elasticnet, param_grid=elasticnet_paramgrid, cv=3, refit=True, n_jobs=1).fit(x_T_COMct, y_Tc).best_params_
ElasticNetCOMctF1_params = GridSearchCV(estimator=elasticnet, param_grid=elasticnet_paramgrid, cv=3, refit=True, n_jobs=1).fit(x_T_COMctF1, y_Tc).best_params_
ElasticNetCOMctF2_params = GridSearchCV(estimator=elasticnet, param_grid=elasticnet_paramgrid, cv=3, refit=True, n_jobs=1).fit(x_T_COMctF2, y_Tc).best_params_
ElasticNetCOMctF3_params = GridSearchCV(estimator=elasticnet, param_grid=elasticnet_paramgrid, cv=3, refit=True, n_jobs=1).fit(x_T_COMctF3, y_Tc).best_params_
ElasticNetCOMctF4_params = GridSearchCV(estimator=elasticnet, param_grid=elasticnet_paramgrid, cv=3, refit=True, n_jobs=1).fit(x_T_COMctF4, y_Tc).best_params_

print('\nELASTICNET GRID SEARCH:',
      '\n> ElasticNetPC:',ElasticNetPC_params, '\n> ElasticNetPCF1:',ElasticNetPCF1_params, '\n> ElasticNetPCF2:',ElasticNetPCF2_params, '\n> ElasticNetPCF3:',ElasticNetPCF3_params, '\n> ElasticNetPCF4:',ElasticNetPCF4_params,
      '\n> ElasticNetSKbn:',ElasticNetSKbn_params, '\n> ElasticNetSKbnF1:',ElasticNetSKbnF1_params, '\n> ElasticNetSKbnF2:',ElasticNetSKbnF2_params, '\n> ElasticNetSKbnF3:',ElasticNetSKbnF3_params, '\n> ElasticNetSKbnF4:',ElasticNetSKbnF4_params,
      '\n> ElasticNetSKct:',ElasticNetSKct_params, '\n> ElasticNetSKctF1:',ElasticNetSKctF1_params, '\n> ElasticNetSKctF2:',ElasticNetSKctF2_params, '\n> ElasticNetSKctF3:',ElasticNetSKctF3_params, '\n> ElasticNetSKctF4:',ElasticNetSKctF4_params,
      '\n> ElasticNetFPbn:',ElasticNetFPbn_params, '\n> ElasticNetFPbnF1:',ElasticNetFPbnF1_params, '\n> ElasticNetFPbnF2:',ElasticNetFPbnF2_params, '\n> ElasticNetFPbnF3:',ElasticNetFPbnF3_params, '\n> ElasticNetFPbnF4:',ElasticNetFPbnF4_params,
      '\n> ElasticNetFPct:',ElasticNetFPct_params, '\n> ElasticNetFPctF1:',ElasticNetFPctF1_params, '\n> ElasticNetFPctF2:',ElasticNetFPctF2_params, '\n> ElasticNetFPctF3:',ElasticNetFPctF3_params, '\n> ElasticNetFPctF4:',ElasticNetFPctF4_params,
      '\n> ElasticNetCOMbn:',ElasticNetCOMbn_params, '\n> ElasticNetCOMbnF1:',ElasticNetCOMbnF1_params, '\n> ElasticNetCOMbnF2:',ElasticNetCOMbnF2_params, '\n> ElasticNetCOMbnF3:',ElasticNetCOMbnF3_params, '\n> ElasticNetCOMbnF4:',ElasticNetCOMbnF4_params,
      '\n> ElasticNetCOMct:',ElasticNetCOMct_params, '\n> ElasticNetCOMctF1:',ElasticNetCOMctF1_params, '\n> ElasticNetCOMctF2:',ElasticNetCOMctF2_params, '\n> ElasticNetCOMctF3:',ElasticNetCOMctF3_params, '\n> ElasticNetCOMctF4:',ElasticNetCOMctF4_params)
print('(%.2fs)' % (time()-start5c))

#   >>> Output
#   ELASTICNET GRID SEARCH: 
#   > ElasticNetPC: {'alpha': 0.001, 'l1_ratio': 0.1} 
#   > ElasticNetPCF1: {'alpha': 0.001, 'l1_ratio': 0.1} 
#   > ElasticNetPCF2: {'alpha': 0.001, 'l1_ratio': 0.1} 
#   > ElasticNetPCF3: {'alpha': 0.001, 'l1_ratio': 0.5} 
#   > ElasticNetPCF4: {'alpha': 0.001, 'l1_ratio': 0.3} 
#   > ElasticNetSKbn: {'alpha': 0.001, 'l1_ratio': 0.1} 
#   > ElasticNetSKbnF1: {'alpha': 0.001, 'l1_ratio': 0.1} 
#   > ElasticNetSKbnF2: {'alpha': 0.001, 'l1_ratio': 0.1} 
#   > ElasticNetSKbnF3: {'alpha': 0.001, 'l1_ratio': 0.1} 
#   > ElasticNetSKbnF4: {'alpha': 0.001, 'l1_ratio': 0.7} 
#   > ElasticNetSKct: {'alpha': 0.001, 'l1_ratio': 0.1} 
#   > ElasticNetSKctF1: {'alpha': 0.001, 'l1_ratio': 0.1} 
#   > ElasticNetSKctF2: {'alpha': 0.001, 'l1_ratio': 0.1} 
#   > ElasticNetSKctF3: {'alpha': 0.001, 'l1_ratio': 0.9} 
#   > ElasticNetSKctF4: {'alpha': 0.001, 'l1_ratio': 0.9} 
#   > ElasticNetFPbn: {'alpha': 0.001, 'l1_ratio': 0.5} 
#   > ElasticNetFPbnF1: {'alpha': 0.001, 'l1_ratio': 0.5} 
#   > ElasticNetFPbnF2: {'alpha': 0.001, 'l1_ratio': 0.7} 
#   > ElasticNetFPbnF3: {'alpha': 0.001, 'l1_ratio': 0.9} 
#   > ElasticNetFPbnF4: {'alpha': 0.001, 'l1_ratio': 0.9} 
#   > ElasticNetFPct: {'alpha': 0.001, 'l1_ratio': 0.1} 
#   > ElasticNetFPctF1: {'alpha': 0.001, 'l1_ratio': 0.1} 
#   > ElasticNetFPctF2: {'alpha': 0.001, 'l1_ratio': 0.5} 
#   > ElasticNetFPctF3: {'alpha': 0.001, 'l1_ratio': 0.9} 
#   > ElasticNetFPctF4: {'alpha': 0.001, 'l1_ratio': 0.9} 
#   > ElasticNetCOMbn: {'alpha': 0.001, 'l1_ratio': 0.1} 
#   > ElasticNetCOMbnF1: {'alpha': 0.001, 'l1_ratio': 0.1} 
#   > ElasticNetCOMbnF2: {'alpha': 0.001, 'l1_ratio': 0.1} 
#   > ElasticNetCOMbnF3: {'alpha': 0.001, 'l1_ratio': 0.1} 
#   > ElasticNetCOMbnF4: {'alpha': 0.001, 'l1_ratio': 0.1} 
#   > ElasticNetCOMct: {'alpha': 0.001, 'l1_ratio': 0.1} 
#   > ElasticNetCOMctF1: {'alpha': 0.001, 'l1_ratio': 0.1} 
#   > ElasticNetCOMctF2: {'alpha': 0.001, 'l1_ratio': 0.1} 
#   > ElasticNetCOMctF3: {'alpha': 0.001, 'l1_ratio': 0.1} 
#   > ElasticNetCOMctF4: {'alpha': 0.001, 'l1_ratio': 0.1}
#   (3509.74s)



# SGD-minimised l1/l2-regularised multilinear regression
sgd = SGDRegressor(random_state=seed)
sgd_paramgrid = {'loss': ['squared_loss', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'],
		 'learning_rate': ['constant', 'optimal', 'invscaling'],
		 'penalty': ['none', 'l2', 'l1', 'elasticnet']}

start5d = time()

SgdPC_params   = GridSearchCV(estimator=sgd, param_grid=sgd_paramgrid, cv=3, refit=True, n_jobs=1).fit(x_T_PC, y_T_PC).best_params_
SgdPCF1_params = GridSearchCV(estimator=sgd, param_grid=sgd_paramgrid, cv=3, refit=True, n_jobs=1).fit(x_T_PCF1, y_T_PC).best_params_
SgdPCF2_params = GridSearchCV(estimator=sgd, param_grid=sgd_paramgrid, cv=3, refit=True, n_jobs=1).fit(x_T_PCF2, y_T_PC).best_params_
SgdPCF3_params = GridSearchCV(estimator=sgd, param_grid=sgd_paramgrid, cv=3, refit=True, n_jobs=1).fit(x_T_PCF3, y_T_PC).best_params_
SgdPCF4_params = GridSearchCV(estimator=sgd, param_grid=sgd_paramgrid, cv=3, refit=True, n_jobs=1).fit(x_T_PCF4, y_T_PC).best_params_

SgdSKbn_params   = GridSearchCV(estimator=sgd, param_grid=sgd_paramgrid, cv=3, refit=True, n_jobs=1).fit(x_T_SKbn, y_T_SKbn).best_params_
SgdSKbnF1_params = GridSearchCV(estimator=sgd, param_grid=sgd_paramgrid, cv=3, refit=True, n_jobs=1).fit(x_T_SKbnF1, y_T_SKbn).best_params_
SgdSKbnF2_params = GridSearchCV(estimator=sgd, param_grid=sgd_paramgrid, cv=3, refit=True, n_jobs=1).fit(x_T_SKbnF2, y_T_SKbn).best_params_
SgdSKbnF3_params = GridSearchCV(estimator=sgd, param_grid=sgd_paramgrid, cv=3, refit=True, n_jobs=1).fit(x_T_SKbnF3, y_T_SKbn).best_params_
SgdSKbnF4_params = GridSearchCV(estimator=sgd, param_grid=sgd_paramgrid, cv=3, refit=True, n_jobs=1).fit(x_T_SKbnF4, y_T_SKbn).best_params_

SgdSKct_params   = GridSearchCV(estimator=sgd, param_grid=sgd_paramgrid, cv=3, refit=True, n_jobs=1).fit(x_T_SKct, y_T_SKct).best_params_
SgdSKctF1_params = GridSearchCV(estimator=sgd, param_grid=sgd_paramgrid, cv=3, refit=True, n_jobs=1).fit(x_T_SKctF1, y_T_SKct).best_params_
SgdSKctF2_params = GridSearchCV(estimator=sgd, param_grid=sgd_paramgrid, cv=3, refit=True, n_jobs=1).fit(x_T_SKctF2, y_T_SKct).best_params_
SgdSKctF3_params = GridSearchCV(estimator=sgd, param_grid=sgd_paramgrid, cv=3, refit=True, n_jobs=1).fit(x_T_SKctF3, y_T_SKct).best_params_
SgdSKctF4_params = GridSearchCV(estimator=sgd, param_grid=sgd_paramgrid, cv=3, refit=True, n_jobs=1).fit(x_T_SKctF4, y_T_SKct).best_params_

SgdFPbn_params   = GridSearchCV(estimator=sgd, param_grid=sgd_paramgrid, cv=3, refit=True, n_jobs=1).fit(x_T_FPbn, y_T_FPbn).best_params_
SgdFPbnF1_params = GridSearchCV(estimator=sgd, param_grid=sgd_paramgrid, cv=3, refit=True, n_jobs=1).fit(x_T_FPbnF1, y_T_FPbn).best_params_
SgdFPbnF2_params = GridSearchCV(estimator=sgd, param_grid=sgd_paramgrid, cv=3, refit=True, n_jobs=1).fit(x_T_FPbnF2, y_T_FPbn).best_params_
SgdFPbnF3_params = GridSearchCV(estimator=sgd, param_grid=sgd_paramgrid, cv=3, refit=True, n_jobs=1).fit(x_T_FPbnF3, y_T_FPbn).best_params_
SgdFPbnF4_params = GridSearchCV(estimator=sgd, param_grid=sgd_paramgrid, cv=3, refit=True, n_jobs=1).fit(x_T_FPbnF4, y_T_FPbn).best_params_

SgdFPct_params   = GridSearchCV(estimator=sgd, param_grid=sgd_paramgrid, cv=3, refit=True, n_jobs=1).fit(x_T_FPct, y_T_FPct).best_params_
SgdFPctF1_params = GridSearchCV(estimator=sgd, param_grid=sgd_paramgrid, cv=3, refit=True, n_jobs=1).fit(x_T_FPctF1, y_T_FPct).best_params_
SgdFPctF2_params = GridSearchCV(estimator=sgd, param_grid=sgd_paramgrid, cv=3, refit=True, n_jobs=1).fit(x_T_FPctF2, y_T_FPct).best_params_
SgdFPctF3_params = GridSearchCV(estimator=sgd, param_grid=sgd_paramgrid, cv=3, refit=True, n_jobs=1).fit(x_T_FPctF3, y_T_FPct).best_params_
SgdFPctF4_params = GridSearchCV(estimator=sgd, param_grid=sgd_paramgrid, cv=3, refit=True, n_jobs=1).fit(x_T_FPctF4, y_T_FPct).best_params_

SgdCOMbn_params   = GridSearchCV(estimator=sgd, param_grid=sgd_paramgrid, cv=3, refit=True, n_jobs=1).fit(x_T_COMbn, y_Tc).best_params_
SgdCOMbnF1_params = GridSearchCV(estimator=sgd, param_grid=sgd_paramgrid, cv=3, refit=True, n_jobs=1).fit(x_T_COMbnF1, y_Tc).best_params_
SgdCOMbnF2_params = GridSearchCV(estimator=sgd, param_grid=sgd_paramgrid, cv=3, refit=True, n_jobs=1).fit(x_T_COMbnF2, y_Tc).best_params_
SgdCOMbnF3_params = GridSearchCV(estimator=sgd, param_grid=sgd_paramgrid, cv=3, refit=True, n_jobs=1).fit(x_T_COMbnF3, y_Tc).best_params_
SgdCOMbnF4_params = GridSearchCV(estimator=sgd, param_grid=sgd_paramgrid, cv=3, refit=True, n_jobs=1).fit(x_T_COMbnF4, y_Tc).best_params_

SgdCOMct_params   = GridSearchCV(estimator=sgd, param_grid=sgd_paramgrid, cv=3, refit=True, n_jobs=1).fit(x_T_COMct, y_Tc).best_params_
SgdCOMctF1_params = GridSearchCV(estimator=sgd, param_grid=sgd_paramgrid, cv=3, refit=True, n_jobs=1).fit(x_T_COMctF1, y_Tc).best_params_
SgdCOMctF2_params = GridSearchCV(estimator=sgd, param_grid=sgd_paramgrid, cv=3, refit=True, n_jobs=1).fit(x_T_COMctF2, y_Tc).best_params_
SgdCOMctF3_params = GridSearchCV(estimator=sgd, param_grid=sgd_paramgrid, cv=3, refit=True, n_jobs=1).fit(x_T_COMctF3, y_Tc).best_params_
SgdCOMctF4_params = GridSearchCV(estimator=sgd, param_grid=sgd_paramgrid, cv=3, refit=True, n_jobs=1).fit(x_T_COMctF4, y_Tc).best_params_

print('\nSGD GRID SEARCH:',
      '\n> SgdPC:',SgdPC_params, '\n> SgdPCF1:',SgdPCF1_params, '\n> SgdPCF2:',SgdPCF2_params, '\n> SgdPCF3:',SgdPCF3_params, '\n> SgdPCF4:',SgdPCF4_params,
      '\n> SgdSKbn:',SgdSKbn_params, '\n> SgdSKbnF1:',SgdSKbnF1_params, '\n> SgdSKbnF2:',SgdSKbnF2_params, '\n> SgdSKbnF3:',SgdSKbnF3_params, '\n> SgdSKbnF4:',SgdSKbnF4_params,
      '\n> SgdSKct:',SgdSKct_params, '\n> SgdSKctF1:',SgdSKctF1_params, '\n> SgdSKctF2:',SgdSKctF2_params, '\n> SgdSKctF3:',SgdSKctF3_params, '\n> SgdSKctF4:',SgdSKctF4_params,
      '\n> SgdFPbn:',SgdFPbn_params, '\n> SgdFPbnF1:',SgdFPbnF1_params, '\n> SgdFPbnF2:',SgdFPbnF2_params, '\n> SgdFPbnF3:',SgdFPbnF3_params, '\n> SgdFPbnF4:',SgdFPbnF4_params,
      '\n> SgdFPct:',SgdFPct_params, '\n> SgdFPctF1:',SgdFPctF1_params, '\n> SgdFPctF2:',SgdFPctF2_params, '\n> SgdFPctF3:',SgdFPctF3_params, '\n> SgdFPctF4:',SgdFPctF4_params,
      '\n> SgdCOMbn:',SgdCOMbn_params, '\n> SgdCOMbnF1:',SgdCOMbnF1_params, '\n> SgdCOMbnF2:',SgdCOMbnF2_params, '\n> SgdCOMbnF3:',SgdCOMbnF3_params, '\n> SgdCOMbnF4:',SgdCOMbnF4_params,
      '\n> SgdCOMct:',SgdCOMct_params, '\n> SgdCOMctF1:',SgdCOMctF1_params, '\n> RSgdCOMctF2:',SgdCOMctF2_params, '\n> SgdCOMctF3:',SgdCOMctF3_params, '\n> SgdCOMctF4:',SgdCOMctF4_params)
print('(%.2fs)' % (time()-start5d))

#   >>> Output
#   SGD GRID SEARCH: 
#   > SgdPC: {'learning_rate': 'invscaling', 'loss': 'squared_epsilon_insensitive', 'penalty': 'none'} 
#   > SgdPCF1: {'learning_rate': 'invscaling', 'loss': 'squared_epsilon_insensitive', 'penalty': 'l1'} 
#   > SgdPCF2: {'learning_rate': 'invscaling', 'loss': 'squared_epsilon_insensitive', 'penalty': 'l1'} 
#   > SgdPCF3: {'learning_rate': 'constant', 'loss': 'squared_loss', 'penalty': 'l1'} 
#   > SgdPCF4: {'learning_rate': 'optimal', 'loss': 'huber', 'penalty': 'l2'} 
#   > SgdSKbn: {'learning_rate': 'invscaling', 'loss': 'squared_epsilon_insensitive', 'penalty': 'none'} 
#   > SgdSKbnF1: {'learning_rate': 'invscaling', 'loss': 'squared_epsilon_insensitive', 'penalty': 'none'} 
#   > SgdSKbnF2: {'learning_rate': 'invscaling', 'loss': 'squared_epsilon_insensitive', 'penalty': 'none'} 
#   > SgdSKbnF3: {'learning_rate': 'invscaling', 'loss': 'squared_epsilon_insensitive', 'penalty': 'none'} 
#   > SgdSKbnF4: {'learning_rate': 'invscaling', 'loss': 'squared_epsilon_insensitive', 'penalty': 'none'} 
#   > SgdSKct: {'learning_rate': 'invscaling', 'loss': 'squared_epsilon_insensitive', 'penalty': 'none'} 
#   > SgdSKctF1: {'learning_rate': 'invscaling', 'loss': 'squared_epsilon_insensitive', 'penalty': 'none'} 
#   > SgdSKctF2: {'learning_rate': 'invscaling', 'loss': 'squared_epsilon_insensitive', 'penalty': 'none'} 
#   > SgdSKctF3: {'learning_rate': 'invscaling', 'loss': 'squared_epsilon_insensitive', 'penalty': 'none'} 
#   > SgdSKctF4: {'learning_rate': 'constant', 'loss': 'squared_loss', 'penalty': 'l1'} 
#   > SgdFPbn: {'learning_rate': 'invscaling', 'loss': 'squared_epsilon_insensitive', 'penalty': 'none'} 
#   > SgdFPbnF1: {'learning_rate': 'invscaling', 'loss': 'squared_epsilon_insensitive', 'penalty': 'none'} 
#   > SgdFPbnF2: {'learning_rate': 'invscaling', 'loss': 'squared_epsilon_insensitive', 'penalty': 'l1'} 
#   > SgdFPbnF3: {'learning_rate': 'invscaling', 'loss': 'squared_loss', 'penalty': 'l1'} 
#   > SgdFPbnF4: {'learning_rate': 'invscaling', 'loss': 'squared_loss', 'penalty': 'l1'} 
#   > SgdFPct: {'learning_rate': 'constant', 'loss': 'squared_epsilon_insensitive', 'penalty': 'none'} 
#   > SgdFPctF1: {'learning_rate': 'optimal', 'loss': 'huber', 'penalty': 'none'} 
#   > SgdFPctF2: {'learning_rate': 'optimal', 'loss': 'huber', 'penalty': 'none'} 
#   > SgdFPctF3: {'learning_rate': 'optimal', 'loss': 'huber', 'penalty': 'none'} 
#   > SgdFPctF4: {'learning_rate': 'constant', 'loss': 'squared_loss', 'penalty': 'l1'} 
#   > SgdCOMbn: {'learning_rate': 'invscaling', 'loss': 'squared_loss', 'penalty': 'none'} 
#   > SgdCOMbnF1: {'learning_rate': 'invscaling', 'loss': 'squared_loss', 'penalty': 'none'} 
#   > SgdCOMbnF2: {'learning_rate': 'invscaling', 'loss': 'squared_loss', 'penalty': 'none'} 
#   > SgdCOMbnF3: {'learning_rate': 'invscaling', 'loss': 'squared_epsilon_insensitive', 'penalty': 'none'} 
#   > SgdCOMbnF4: {'learning_rate': 'invscaling', 'loss': 'squared_epsilon_insensitive', 'penalty': 'none'} 
#   > SgdCOMct: {'learning_rate': 'invscaling', 'loss': 'squared_loss', 'penalty': 'none'} 
#   > SgdCOMctF1: {'learning_rate': 'invscaling', 'loss': 'squared_loss', 'penalty': 'none'} 
#   > RSgdCOMctF2: {'learning_rate': 'invscaling', 'loss': 'squared_loss', 'penalty': 'none'} 
#   > SgdCOMctF3: {'learning_rate': 'invscaling', 'loss': 'squared_epsilon_insensitive', 'penalty': 'none'} 
#   > SgdCOMctF4: {'learning_rate': 'constant', 'loss': 'squared_loss', 'penalty': 'l1'}
#   (3479.39s)



print('\n(%.2fs)' % (time()-start5))
'''


#=============================================================================
# (6) ALGORITHM TRAINING AND MODEL PREDICTIONS
#=============================================================================
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, SGDRegressor



print('\n\n\n########## ALGORITHM TRAINING ##########')
start6 = time()



# Multilinear regression
start6a = time()

MlrPC   = LinearRegression(n_jobs=-1).fit(x_T_PC, y_T_PC).predict(x_V_PC)
MlrPCF1 = LinearRegression(n_jobs=-1).fit(x_T_PCF1, y_T_PC).predict(x_V_PCF1)
MlrPCF2 = LinearRegression(n_jobs=-1).fit(x_T_PCF2, y_T_PC).predict(x_V_PCF2)
MlrPCF3 = LinearRegression(n_jobs=-1).fit(x_T_PCF3, y_T_PC).predict(x_V_PCF3)
MlrPCF4 = LinearRegression(n_jobs=-1).fit(x_T_PCF4, y_T_PC).predict(x_V_PCF4)

MlrSKbn   = Lasso(alpha=0, random_state=seed).fit(x_T_SKbn, y_T_SKbn).predict(x_V_SKbn)
MlrSKbnF1 = Lasso(alpha=0, random_state=seed).fit(x_T_SKbnF1, y_T_SKbn).predict(x_V_SKbnF1)
MlrSKbnF2 = LinearRegression(n_jobs=-1).fit(x_T_SKbnF2, y_T_SKbn).predict(x_V_SKbnF2)
MlrSKbnF3 = LinearRegression(n_jobs=-1).fit(x_T_SKbnF3, y_T_SKbn).predict(x_V_SKbnF3)
MlrSKbnF4 = LinearRegression(n_jobs=-1).fit(x_T_SKbnF4, y_T_SKbn).predict(x_V_SKbnF4)

MlrSKct   = Lasso(alpha=0, random_state=seed).fit(x_T_SKct, y_T_SKct).predict(x_V_SKct)
MlrSKctF1 = LinearRegression(n_jobs=-1).fit(x_T_SKctF1, y_T_SKct).predict(x_V_SKctF1)
MlrSKctF2 = LinearRegression(n_jobs=-1).fit(x_T_SKctF2, y_T_SKct).predict(x_V_SKctF2)
MlrSKctF3 = LinearRegression(n_jobs=-1).fit(x_T_SKctF3, y_T_SKct).predict(x_V_SKctF3)
MlrSKctF4 = LinearRegression(n_jobs=-1).fit(x_T_SKctF4, y_T_SKct).predict(x_V_SKctF4)

MlrFPbn   = LinearRegression(n_jobs=-1).fit(x_T_FPbn, y_T_FPbn).predict(x_V_FPbn)
MlrFPbnF1 = LinearRegression(n_jobs=-1).fit(x_T_FPbnF1, y_T_FPbn).predict(x_V_FPbnF1)
MlrFPbnF2 = LinearRegression(n_jobs=-1).fit(x_T_FPbnF2, y_T_FPbn).predict(x_V_FPbnF2)
MlrFPbnF3 = LinearRegression(n_jobs=-1).fit(x_T_FPbnF3, y_T_FPbn).predict(x_V_FPbnF3)
MlrFPbnF4 = LinearRegression(n_jobs=-1).fit(x_T_FPbnF4, y_T_FPbn).predict(x_V_FPbnF4)

MlrFPct   = LinearRegression(n_jobs=-1).fit(x_T_FPct, y_T_FPct).predict(x_V_FPct)
MlrFPctF1 = LinearRegression(n_jobs=-1).fit(x_T_FPctF1, y_T_FPct).predict(x_V_FPctF1)
MlrFPctF2 = LinearRegression(n_jobs=-1).fit(x_T_FPctF2, y_T_FPct).predict(x_V_FPctF2)
MlrFPctF3 = LinearRegression(n_jobs=-1).fit(x_T_FPctF3, y_T_FPct).predict(x_V_FPctF3)
MlrFPctF4 = LinearRegression(n_jobs=-1).fit(x_T_FPctF4, y_T_FPct).predict(x_V_FPctF4)

MlrCOMbn   = Lasso(alpha=0, random_state=seed).fit(x_T_COMbn, y_Tc).predict(x_V_COMbn)
MlrCOMbnF1 = Lasso(alpha=0, random_state=seed).fit(x_T_COMbnF1, y_Tc).predict(x_V_COMbnF1)
MlrCOMbnF2 = LinearRegression(n_jobs=-1).fit(x_T_COMbnF2, y_Tc).predict(x_V_COMbnF2)
MlrCOMbnF3 = LinearRegression(n_jobs=-1).fit(x_T_COMbnF3, y_Tc).predict(x_V_COMbnF3)
MlrCOMbnF4 = LinearRegression(n_jobs=-1).fit(x_T_COMbnF4, y_Tc).predict(x_V_COMbnF4)

MlrCOMct   = Lasso(alpha=0, random_state=seed).fit(x_T_COMct, y_Tc).predict(x_V_COMct)
MlrCOMctF1 = LinearRegression(n_jobs=-1).fit(x_T_COMctF1, y_Tc).predict(x_V_COMctF1)
MlrCOMctF2 = LinearRegression(n_jobs=-1).fit(x_T_COMctF2, y_Tc).predict(x_V_COMctF2)
MlrCOMctF3 = LinearRegression(n_jobs=-1).fit(x_T_COMctF3, y_Tc).predict(x_V_COMctF3)
MlrCOMctF4 = LinearRegression(n_jobs=-1).fit(x_T_COMctF4, y_Tc).predict(x_V_COMctF4)

print('> Multilinear regression trained (%.2fs)' % (time()-start6a))



# Lasso (l1-regularised) multilinear regression
start6b = time()

class CustomLasso:
	
	def __init__(self, alpha, selection):
		self.alpha = alpha
		self.selection = selection

	def fit_predict(self, x_train, y_train, x_predict):
		self.x_train = x_train
		self.y_train = y_train
		self.x_predict = x_predict
		
		def lasso(random_state):
			return Lasso(random_state=random_state, alpha=self.alpha, selection=self.selection, max_iter=3000)
		
		self.model1 = lasso(69).fit(self.x_train, self.y_train)
		self.model2 = lasso(135).fit(self.x_train, self.y_train)
		self.model3 = lasso(346).fit(self.x_train, self.y_train)
		self.model4 = lasso(545).fit(self.x_train, self.y_train)
		self.model5 = lasso(978).fit(self.x_train, self.y_train)
		
		self.prediction1 = self.model1.predict(self.x_predict)
		self.prediction2 = self.model2.predict(self.x_predict)
		self.prediction3 = self.model3.predict(self.x_predict)
		self.prediction4 = self.model4.predict(self.x_predict)
		self.prediction5 = self.model5.predict(self.x_predict)
		
		self.final_prediction = []
		for pred1x, pred2x, pred3x, pred4x, pred5x in zip(self.prediction1, self.prediction2, self.prediction3, self.prediction4, self.prediction5):
			self.final_prediction.append(np.mean([pred1x, pred2x, pred3x, pred4x, pred5x]))
		
		return np.asarray(self.final_prediction)

	

LassoPC   = CustomLasso(0.001, 'random').fit_predict(x_T_PC, y_T_PC, x_V_PC)
LassoPCF1 = CustomLasso(0.001, 'cyclic').fit_predict(x_T_PCF1, y_T_PC, x_V_PCF1)
LassoPCF2 = CustomLasso(0.001, 'cyclic').fit_predict(x_T_PCF2, y_T_PC, x_V_PCF2)
LassoPCF3 = CustomLasso(0.001, 'random').fit_predict(x_T_PCF3, y_T_PC, x_V_PCF3)
LassoPCF4 = CustomLasso(0.001, 'cyclic').fit_predict(x_T_PCF4, y_T_PC, x_V_PCF4)

LassoSKbn   = CustomLasso(0.001, 'cyclic').fit_predict(x_T_SKbn, y_T_SKbn, x_V_SKbn)
LassoSKbnF1 = CustomLasso(0.001, 'cyclic').fit_predict(x_T_SKbnF1, y_T_SKbn, x_V_SKbnF1)
LassoSKbnF2 = CustomLasso(0.001, 'cyclic').fit_predict(x_T_SKbnF2, y_T_SKbn, x_V_SKbnF2)
LassoSKbnF3 = CustomLasso(0.001, 'cyclic').fit_predict(x_T_SKbnF3, y_T_SKbn, x_V_SKbnF3)
LassoSKbnF4 = CustomLasso(0.001, 'cyclic').fit_predict(x_T_SKbnF4, y_T_SKbn, x_V_SKbnF4)

LassoSKct   = CustomLasso(0.001, 'cyclic').fit_predict(x_T_SKct, y_T_SKct, x_V_SKct)
LassoSKctF1 = CustomLasso(0.001, 'cyclic').fit_predict(x_T_SKctF1, y_T_SKct, x_V_SKctF1)
LassoSKctF2 = CustomLasso(0.001, 'cyclic').fit_predict(x_T_SKctF2, y_T_SKct, x_V_SKctF2)
LassoSKctF3 = CustomLasso(0.001, 'random').fit_predict(x_T_SKctF3, y_T_SKct, x_V_SKctF3)
LassoSKctF4 = CustomLasso(0.001, 'random').fit_predict(x_T_SKctF4, y_T_SKct, x_V_SKctF4)

LassoFPbn   = CustomLasso(0.001, 'random').fit_predict(x_T_FPbn, y_T_FPbn, x_V_FPbn)
LassoFPbnF1 = CustomLasso(0.001, 'random').fit_predict(x_T_FPbnF1, y_T_FPbn, x_V_FPbnF1)
LassoFPbnF2 = CustomLasso(0.001, 'random').fit_predict(x_T_FPbnF2, y_T_FPbn, x_V_FPbnF2)
LassoFPbnF3 = CustomLasso(0.001, 'random').fit_predict(x_T_FPbnF3, y_T_FPbn, x_V_FPbnF3)
LassoFPbnF4 = CustomLasso(0.001, 'cyclic').fit_predict(x_T_FPbnF4, y_T_FPbn, x_V_FPbnF4)

LassoFPct   = CustomLasso(0.001, 'cyclic').fit_predict(x_T_FPct, y_T_FPct, x_V_FPct)
LassoFPctF1 = CustomLasso(0.001, 'random').fit_predict(x_T_FPctF1, y_T_FPct, x_V_FPctF1)
LassoFPctF2 = CustomLasso(0.001, 'cyclic').fit_predict(x_T_FPctF2, y_T_FPct, x_V_FPctF2)
LassoFPctF3 = CustomLasso(0.001, 'random').fit_predict(x_T_FPctF3, y_T_FPct, x_V_FPctF3)
LassoFPctF4 = CustomLasso(0.001, 'random').fit_predict(x_T_FPctF4, y_T_FPct, x_V_FPctF4)

LassoCOMbn   = CustomLasso(0.001, 'cyclic').fit_predict(x_T_COMbn, y_Tc, x_V_COMbn)
LassoCOMbnF1 = CustomLasso(0.001, 'random').fit_predict(x_T_COMbnF1, y_Tc, x_V_COMbnF1)
LassoCOMbnF2 = CustomLasso(0.001, 'cyclic').fit_predict(x_T_COMbnF2, y_Tc, x_V_COMbnF2)
LassoCOMbnF3 = CustomLasso(0.001, 'cyclic').fit_predict(x_T_COMbnF3, y_Tc, x_V_COMbnF3)
LassoCOMbnF4 = CustomLasso(0.001, 'cyclic').fit_predict(x_T_COMbnF4, y_Tc, x_V_COMbnF4)

LassoCOMct   = CustomLasso(0.001, 'cyclic').fit_predict(x_T_COMct, y_Tc, x_V_COMct)
LassoCOMctF1 = CustomLasso(0.001, 'random').fit_predict(x_T_COMctF1, y_Tc, x_V_COMctF1)
LassoCOMctF2 = CustomLasso(0.001, 'random').fit_predict(x_T_COMctF2, y_Tc, x_V_COMctF2)
LassoCOMctF3 = CustomLasso(0.001, 'cyclic').fit_predict(x_T_COMctF3, y_Tc, x_V_COMctF3)
LassoCOMctF4 = CustomLasso(0.001, 'cyclic').fit_predict(x_T_COMctF4, y_Tc, x_V_COMctF4)

print('> Lasso multilinear regression trained (%.2fs)' % (time()-start6b))



# Ridge (l2-regularised) multilinear regression
start6c = time()

class CustomRidge:
	
	def __init__(self, alpha, solver):
		self.alpha = alpha
		self.solver = solver

	def fit_predict(self, x_train, y_train, x_predict):
		self.x_train = x_train
		self.y_train = y_train
		self.x_predict = x_predict
		
		def ridge(random_state):
			return Ridge(random_state=random_state, alpha=self.alpha, solver=self.solver)
		
		self.model1 = ridge(69).fit(self.x_train, self.y_train)
		self.model2 = ridge(135).fit(self.x_train, self.y_train)
		self.model3 = ridge(346).fit(self.x_train, self.y_train)
		self.model4 = ridge(545).fit(self.x_train, self.y_train)
		self.model5 = ridge(978).fit(self.x_train, self.y_train)
		
		self.prediction1 = self.model1.predict(self.x_predict)
		self.prediction2 = self.model2.predict(self.x_predict)
		self.prediction3 = self.model3.predict(self.x_predict)
		self.prediction4 = self.model4.predict(self.x_predict)
		self.prediction5 = self.model5.predict(self.x_predict)
		
		self.final_prediction = []
		for pred1x, pred2x, pred3x, pred4x, pred5x in zip(self.prediction1, self.prediction2, self.prediction3, self.prediction4, self.prediction5):
			self.final_prediction.append(np.mean([pred1x, pred2x, pred3x, pred4x, pred5x]))
		
		return np.asarray(self.final_prediction)



RidgePC   = CustomRidge(0.1, 'sag').fit_predict(x_T_PC, y_T_PC, x_V_PC)
RidgePCF1 = CustomRidge(0.1, 'auto').fit_predict(x_T_PCF1, y_T_PC, x_V_PCF1)
RidgePCF2 = CustomRidge(0.1, 'auto').fit_predict(x_T_PCF2, y_T_PC, x_V_PCF2)
RidgePCF3 = CustomRidge(10, 'svd').fit_predict(x_T_PCF3, y_T_PC, x_V_PCF3)
RidgePCF4 = CustomRidge(10, 'sag').fit_predict(x_T_PCF4, y_T_PC, x_V_PCF4)

RidgeSKbn   = CustomRidge(1, 'saga').fit_predict(x_T_SKbn, y_T_SKbn, x_V_SKbn)
RidgeSKbnF1 = CustomRidge(1, 'lsqr').fit_predict(x_T_SKbnF1, y_T_SKbn, x_V_SKbnF1)
RidgeSKbnF2 = CustomRidge(1, 'saga').fit_predict(x_T_SKbnF2, y_T_SKbn, x_V_SKbnF2)
RidgeSKbnF3 = CustomRidge(1, 'saga').fit_predict(x_T_SKbnF3, y_T_SKbn, x_V_SKbnF3)
RidgeSKbnF4 = CustomRidge(1, 'sparse_cg').fit_predict(x_T_SKbnF4, y_T_SKbn, x_V_SKbnF4)

RidgeSKct   = CustomRidge(1, 'auto').fit_predict(x_T_SKct, y_T_SKct, x_V_SKct)
RidgeSKctF1 = CustomRidge(0.1, 'sag').fit_predict(x_T_SKctF1, y_T_SKct, x_V_SKctF1)
RidgeSKctF2 = CustomRidge(0.1, 'sag').fit_predict(x_T_SKctF2, y_T_SKct, x_V_SKctF2)
RidgeSKctF3 = CustomRidge(0.001, 'auto').fit_predict(x_T_SKctF3, y_T_SKct, x_V_SKctF3)
RidgeSKctF4 = CustomRidge(0.001, 'sag').fit_predict(x_T_SKctF4, y_T_SKct, x_V_SKctF4)

RidgeFPbn   = CustomRidge(10, 'lsqr').fit_predict(x_T_FPbn, y_T_FPbn, x_V_FPbn)
RidgeFPbnF1 = CustomRidge(10, 'saga').fit_predict(x_T_FPbnF1, y_T_FPbn, x_V_FPbnF1)
RidgeFPbnF2 = CustomRidge(100, 'saga').fit_predict(x_T_FPbnF2, y_T_FPbn, x_V_FPbnF2)
RidgeFPbnF3 = CustomRidge(100, 'saga').fit_predict(x_T_FPbnF3, y_T_FPbn, x_V_FPbnF3)
RidgeFPbnF4 = CustomRidge(100, 'saga').fit_predict(x_T_FPbnF4, y_T_FPbn, x_V_FPbnF4)

RidgeFPct   = CustomRidge(1, 'saga').fit_predict(x_T_FPct, y_T_FPct, x_V_FPct)
RidgeFPctF1 = CustomRidge(1, 'lsqr').fit_predict(x_T_FPctF1, y_T_FPct, x_V_FPctF1)
RidgeFPctF2 = CustomRidge(1, 'sparse_cg').fit_predict(x_T_FPctF2, y_T_FPct, x_V_FPctF2)
RidgeFPctF3 = CustomRidge(1, 'lsqr').fit_predict(x_T_FPctF3, y_T_FPct, x_V_FPctF3)
RidgeFPctF4 = CustomRidge(10, 'sparse_cg').fit_predict(x_T_FPctF4, y_T_FPct, x_V_FPctF4)

RidgeCOMbn   = CustomRidge(1, 'auto').fit_predict(x_T_COMbn, y_Tc, x_V_COMbn)
RidgeCOMbnF1 = CustomRidge(1, 'auto').fit_predict(x_T_COMbnF1, y_Tc, x_V_COMbnF1)
RidgeCOMbnF2 = CustomRidge(1, 'svd').fit_predict(x_T_COMbnF2, y_Tc, x_V_COMbnF2)
RidgeCOMbnF3 = CustomRidge(1, 'saga').fit_predict(x_T_COMbnF3, y_Tc, x_V_COMbnF3)
RidgeCOMbnF4 = CustomRidge(10, 'lsqr').fit_predict(x_T_COMbnF4, y_Tc, x_V_COMbnF4)

RidgeCOMct   = CustomRidge(1, 'svd').fit_predict(x_T_COMct, y_Tc, x_V_COMct)
RidgeCOMctF1 = CustomRidge(1, 'svd').fit_predict(x_T_COMctF1, y_Tc, x_V_COMctF1)
RidgeCOMctF2 = CustomRidge(1, 'svd').fit_predict(x_T_COMctF2, y_Tc, x_V_COMctF2)
RidgeCOMctF3 = CustomRidge(0.1, 'auto').fit_predict(x_T_COMctF3, y_Tc, x_V_COMctF3)
RidgeCOMctF4 = CustomRidge(1, 'auto').fit_predict(x_T_COMctF4, y_Tc, x_V_COMctF4)

print('> Ridge multilinear regression trained (%.2fs)' % (time()-start6c))



# ElasticNet (l1/l2-regularised) multilinear regression
start6d = time()

class CustomElasticNet:
	
	def __init__(self, alpha, l1_ratio):
		self.alpha = alpha
		self.l1_ratio = l1_ratio

	def fit_predict(self, x_train, y_train, x_predict):
		self.x_train = x_train
		self.y_train = y_train
		self.x_predict = x_predict
		
		def elasticnet(random_state):
			return ElasticNet(random_state=random_state, alpha=self.alpha, l1_ratio=self.l1_ratio, max_iter=5000, selection='random')
		
		self.model1 = elasticnet(69).fit(self.x_train, self.y_train)
		self.model2 = elasticnet(135).fit(self.x_train, self.y_train)
		self.model3 = elasticnet(346).fit(self.x_train, self.y_train)
		self.model4 = elasticnet(545).fit(self.x_train, self.y_train)
		self.model5 = elasticnet(978).fit(self.x_train, self.y_train)
		
		self.prediction1 = self.model1.predict(self.x_predict)
		self.prediction2 = self.model2.predict(self.x_predict)
		self.prediction3 = self.model3.predict(self.x_predict)
		self.prediction4 = self.model4.predict(self.x_predict)
		self.prediction5 = self.model5.predict(self.x_predict)
		
		self.final_prediction = []
		for pred1x, pred2x, pred3x, pred4x, pred5x in zip(self.prediction1, self.prediction2, self.prediction3, self.prediction4, self.prediction5):
			self.final_prediction.append(np.mean([pred1x, pred2x, pred3x, pred4x, pred5x]))
		
		return np.asarray(self.final_prediction)



ElasticNetPC   = CustomElasticNet(0.001, 0.1).fit_predict(x_T_PC, y_T_PC, x_V_PC)
ElasticNetPCF1 = CustomElasticNet(0.001, 0.1).fit_predict(x_T_PCF1, y_T_PC, x_V_PCF1)
ElasticNetPCF2 = CustomElasticNet(0.001, 0.1).fit_predict(x_T_PCF2, y_T_PC, x_V_PCF2)
ElasticNetPCF3 = CustomElasticNet(0.001, 0.5).fit_predict(x_T_PCF3, y_T_PC, x_V_PCF3)
ElasticNetPCF4 = CustomElasticNet(0.001, 0.3).fit_predict(x_T_PCF4, y_T_PC, x_V_PCF4)

ElasticNetSKbn   = CustomElasticNet(0.001, 0.1).fit_predict(x_T_SKbn, y_T_SKbn, x_V_SKbn)
ElasticNetSKbnF1 = CustomElasticNet(0.001, 0.1).fit_predict(x_T_SKbnF1, y_T_SKbn, x_V_SKbnF1)
ElasticNetSKbnF2 = CustomElasticNet(0.001, 0.1).fit_predict(x_T_SKbnF2, y_T_SKbn, x_V_SKbnF2)
ElasticNetSKbnF3 = CustomElasticNet(0.001, 0.1).fit_predict(x_T_SKbnF3, y_T_SKbn, x_V_SKbnF3)
ElasticNetSKbnF4 = CustomElasticNet(0.001, 0.7).fit_predict(x_T_SKbnF4, y_T_SKbn, x_V_SKbnF4)

ElasticNetSKct   = CustomElasticNet(0.001, 0.1).fit_predict(x_T_SKct, y_T_SKct, x_V_SKct)
ElasticNetSKctF1 = CustomElasticNet(0.001, 0.1).fit_predict(x_T_SKctF1, y_T_SKct, x_V_SKctF1)
ElasticNetSKctF2 = CustomElasticNet(0.001, 0.1).fit_predict(x_T_SKctF2, y_T_SKct, x_V_SKctF2)
ElasticNetSKctF3 = CustomElasticNet(0.001, 0.9).fit_predict(x_T_SKctF3, y_T_SKct, x_V_SKctF3)
ElasticNetSKctF4 = CustomElasticNet(0.001, 0.9).fit_predict(x_T_SKctF4, y_T_SKct, x_V_SKctF4)

ElasticNetFPbn   = CustomElasticNet(0.001, 0.5).fit_predict(x_T_FPbn, y_T_FPbn, x_V_FPbn)
ElasticNetFPbnF1 = CustomElasticNet(0.001, 0.5).fit_predict(x_T_FPbnF1, y_T_FPbn, x_V_FPbnF1)
ElasticNetFPbnF2 = CustomElasticNet(0.001, 0.7).fit_predict(x_T_FPbnF2, y_T_FPbn, x_V_FPbnF2)
ElasticNetFPbnF3 = CustomElasticNet(0.001, 0.9).fit_predict(x_T_FPbnF3, y_T_FPbn, x_V_FPbnF3)
ElasticNetFPbnF4 = CustomElasticNet(0.001, 0.9).fit_predict(x_T_FPbnF4, y_T_FPbn, x_V_FPbnF4)

ElasticNetFPct   = CustomElasticNet(0.001, 0.1).fit_predict(x_T_FPct, y_T_FPct, x_V_FPct)
ElasticNetFPctF1 = CustomElasticNet(0.001, 0.1).fit_predict(x_T_FPctF1, y_T_FPct, x_V_FPctF1)
ElasticNetFPctF2 = CustomElasticNet(0.001, 0.5).fit_predict(x_T_FPctF2, y_T_FPct, x_V_FPctF2)
ElasticNetFPctF3 = CustomElasticNet(0.001, 0.9).fit_predict(x_T_FPctF3, y_T_FPct, x_V_FPctF3)
ElasticNetFPctF4 = CustomElasticNet(0.001, 0.9).fit_predict(x_T_FPctF4, y_T_FPct, x_V_FPctF4)

ElasticNetCOMbn   = CustomElasticNet(0.001, 0.1).fit_predict(x_T_COMbn, y_Tc, x_V_COMbn)
ElasticNetCOMbnF1 = CustomElasticNet(0.001, 0.1).fit_predict(x_T_COMbnF1, y_Tc, x_V_COMbnF1)
ElasticNetCOMbnF2 = CustomElasticNet(0.001, 0.1).fit_predict(x_T_COMbnF2, y_Tc, x_V_COMbnF2)
ElasticNetCOMbnF3 = CustomElasticNet(0.001, 0.1).fit_predict(x_T_COMbnF3, y_Tc, x_V_COMbnF3)
ElasticNetCOMbnF4 = CustomElasticNet(0.001, 0.1).fit_predict(x_T_COMbnF4, y_Tc, x_V_COMbnF4)

ElasticNetCOMct   = CustomElasticNet(0.001, 0.1).fit_predict(x_T_COMct, y_Tc, x_V_COMct)
ElasticNetCOMctF1 = CustomElasticNet(0.001, 0.1).fit_predict(x_T_COMctF1, y_Tc, x_V_COMctF1)
ElasticNetCOMctF2 = CustomElasticNet(0.001, 0.1).fit_predict(x_T_COMctF2, y_Tc, x_V_COMctF2)
ElasticNetCOMctF3 = CustomElasticNet(0.001, 0.1).fit_predict(x_T_COMctF3, y_Tc, x_V_COMctF3)
ElasticNetCOMctF4 = CustomElasticNet(0.001, 0.1).fit_predict(x_T_COMctF4, y_Tc, x_V_COMctF4)

print('> ElasticNet multilinear regression trained (%.2fs)' % (time()-start6d))



# SGD-minimised l1/l2-regularised multilinear regression
start6e = time()

class CustomSGD:
	
	def __init__(self, loss, learning_rate, penalty):
		self.loss = loss
		self.learning_rate = learning_rate
		self.penalty = penalty

	def fit_predict(self, x_train, y_train, x_predict):
		self.x_train = x_train
		self.y_train = y_train
		self.x_predict = x_predict
		
		def sgd(random_state):
			return SGDRegressor(random_state=random_state, loss=self.loss, learning_rate=self.learning_rate, penalty=self.penalty)
		
		self.model1 = sgd(69).fit(self.x_train, self.y_train)
		self.model2 = sgd(135).fit(self.x_train, self.y_train)
		self.model3 = sgd(346).fit(self.x_train, self.y_train)
		self.model4 = sgd(545).fit(self.x_train, self.y_train)
		self.model5 = sgd(978).fit(self.x_train, self.y_train)
		
		self.prediction1 = self.model1.predict(self.x_predict)
		self.prediction2 = self.model2.predict(self.x_predict)
		self.prediction3 = self.model3.predict(self.x_predict)
		self.prediction4 = self.model4.predict(self.x_predict)
		self.prediction5 = self.model5.predict(self.x_predict)
		
		self.final_prediction = []
		for pred1x, pred2x, pred3x, pred4x, pred5x in zip(self.prediction1, self.prediction2, self.prediction3, self.prediction4, self.prediction5):
			self.final_prediction.append(np.mean([pred1x, pred2x, pred3x, pred4x, pred5x]))
		
		return np.asarray(self.final_prediction)



SgdPC   = CustomSGD('squared_epsilon_insensitive', 'invscaling', 'none').fit_predict(x_T_PC, y_T_PC, x_V_PC)
SgdPCF1 = CustomSGD('squared_epsilon_insensitive', 'invscaling', 'l1').fit_predict(x_T_PCF1, y_T_PC, x_V_PCF1)
SgdPCF2 = CustomSGD('squared_epsilon_insensitive', 'invscaling', 'l1').fit_predict(x_T_PCF2, y_T_PC, x_V_PCF2)
SgdPCF3 = CustomSGD('squared_loss', 'constant', 'l1').fit_predict(x_T_PCF3, y_T_PC, x_V_PCF3)
SgdPCF4 = CustomSGD('huber', 'optimal', 'l2').fit_predict(x_T_PCF4, y_T_PC, x_V_PCF4)

SgdSKbn   = CustomSGD('squared_epsilon_insensitive', 'invscaling', 'none').fit_predict(x_T_SKbn, y_T_SKbn, x_V_SKbn)
SgdSKbnF1 = CustomSGD('squared_epsilon_insensitive', 'invscaling', 'none').fit_predict(x_T_SKbnF1, y_T_SKbn, x_V_SKbnF1)
SgdSKbnF2 = CustomSGD('squared_epsilon_insensitive', 'invscaling', 'none').fit_predict(x_T_SKbnF2, y_T_SKbn, x_V_SKbnF2)
SgdSKbnF3 = CustomSGD('squared_epsilon_insensitive', 'invscaling', 'none').fit_predict(x_T_SKbnF3, y_T_SKbn, x_V_SKbnF3)
SgdSKbnF4 = CustomSGD('squared_epsilon_insensitive', 'invscaling', 'none').fit_predict(x_T_SKbnF4, y_T_SKbn, x_V_SKbnF4)

SgdSKct   = CustomSGD('squared_epsilon_insensitive', 'invscaling', 'none').fit_predict(x_T_SKct, y_T_SKct, x_V_SKct)
SgdSKctF1 = CustomSGD('squared_epsilon_insensitive', 'invscaling', 'none').fit_predict(x_T_SKctF1, y_T_SKct, x_V_SKctF1)
SgdSKctF2 = CustomSGD('squared_epsilon_insensitive', 'invscaling', 'none').fit_predict(x_T_SKctF2, y_T_SKct, x_V_SKctF2)
SgdSKctF3 = CustomSGD('squared_epsilon_insensitive', 'invscaling', 'none').fit_predict(x_T_SKctF3, y_T_SKct, x_V_SKctF3)
SgdSKctF4 = CustomSGD('squared_loss', 'constant', 'l1').fit_predict(x_T_SKctF4, y_T_SKct, x_V_SKctF4)

SgdFPbn   = CustomSGD('squared_epsilon_insensitive', 'invscaling', 'none').fit_predict(x_T_FPbn, y_T_FPbn, x_V_FPbn)
SgdFPbnF1 = CustomSGD('squared_epsilon_insensitive', 'invscaling', 'none').fit_predict(x_T_FPbnF1, y_T_FPbn, x_V_FPbnF1)
SgdFPbnF2 = CustomSGD('squared_epsilon_insensitive', 'invscaling', 'l1').fit_predict(x_T_FPbnF2, y_T_FPbn, x_V_FPbnF2)
SgdFPbnF3 = CustomSGD('squared_loss', 'invscaling', 'l1').fit_predict(x_T_FPbnF3, y_T_FPbn, x_V_FPbnF3)
SgdFPbnF4 = CustomSGD('squared_loss', 'invscaling', 'l1').fit_predict(x_T_FPbnF4, y_T_FPbn, x_V_FPbnF4)

SgdFPct   = CustomSGD('squared_epsilon_insensitive', 'constant', 'none').fit_predict(x_T_FPct, y_T_FPct, x_V_FPct)
SgdFPctF1 = CustomSGD('huber', 'optimal', 'none').fit_predict(x_T_FPctF1, y_T_FPct, x_V_FPctF1)
SgdFPctF2 = CustomSGD('huber', 'optimal', 'none').fit_predict(x_T_FPctF2, y_T_FPct, x_V_FPctF2)
SgdFPctF3 = CustomSGD('huber', 'optimal', 'none').fit_predict(x_T_FPctF3, y_T_FPct, x_V_FPctF3)
SgdFPctF4 = CustomSGD('squared_loss', 'constant', 'l1').fit_predict(x_T_FPctF4, y_T_FPct, x_V_FPctF4)

SgdCOMbn   = CustomSGD('squared_loss', 'invscaling', 'none').fit_predict(x_T_COMbn, y_Tc, x_V_COMbn)
SgdCOMbnF1 = CustomSGD('squared_loss', 'invscaling', 'none').fit_predict(x_T_COMbnF1, y_Tc, x_V_COMbnF1)
SgdCOMbnF2 = CustomSGD('squared_loss', 'invscaling', 'none').fit_predict(x_T_COMbnF2, y_Tc, x_V_COMbnF2)
SgdCOMbnF3 = CustomSGD('squared_epsilon_insensitive', 'invscaling', 'none').fit_predict(x_T_COMbnF3, y_Tc, x_V_COMbnF3)
SgdCOMbnF4 = CustomSGD('squared_epsilon_insensitive', 'invscaling', 'none').fit_predict(x_T_COMbnF4, y_Tc, x_V_COMbnF4)

SgdCOMct   = CustomSGD('squared_loss', 'invscaling', 'none').fit_predict(x_T_COMct, y_Tc, x_V_COMct)
SgdCOMctF1 = CustomSGD('squared_loss', 'invscaling', 'none').fit_predict(x_T_COMctF1, y_Tc, x_V_COMctF1)
SgdCOMctF2 = CustomSGD('squared_loss', 'invscaling', 'none').fit_predict(x_T_COMctF2, y_Tc, x_V_COMctF2)
SgdCOMctF3 = CustomSGD('squared_epsilon_insensitive', 'invscaling', 'none').fit_predict(x_T_COMctF3, y_Tc, x_V_COMctF3)
SgdCOMctF4 = CustomSGD('squared_loss', 'constant', 'l1').fit_predict(x_T_COMctF4, y_Tc, x_V_COMctF4)

print('SGD multilinear regression trained (%.2fs)' % (time()-start6e))



print('(%.2fs)' % (time()-start6))



#=============================================================================
# (7) MODEL BENCHMARKING
#=============================================================================
print('\n\n\n########## MODEL BENCHMARKING ##########')
start7 = time()



def RMSE(true_logP, predicted_logP):
	mean_squared_error = np.square(np.subtract(true_logP, predicted_logP)).mean()
	return np.sqrt(mean_squared_error)

def CI(data, confidence=0.95): # Confidence interval using t-distribution; adapted from https://stackoverflow.com/questions/15033511
	n, m, se = len(data), np.mean(data), sp.stats.sem(data)
	return se * sp.stats.t.ppf((1 + confidence) / 2, n-1)



# Multilinear regression
MlrPC_rmse   = RMSE(y_V_PC, MlrPC)
MlrPCF1_rmse = RMSE(y_V_PC, MlrPCF1)
MlrPCF2_rmse = RMSE(y_V_PC, MlrPCF2)
MlrPCF3_rmse = RMSE(y_V_PC, MlrPCF3)
MlrPCF4_rmse = RMSE(y_V_PC, MlrPCF4)

MlrSKbn_rmse   = RMSE(y_V_SKbn, MlrSKbn)
MlrSKbnF1_rmse = RMSE(y_V_SKbn, MlrSKbnF1)
MlrSKbnF2_rmse = RMSE(y_V_SKbn, MlrSKbnF2)
MlrSKbnF3_rmse = RMSE(y_V_SKbn, MlrSKbnF3)
MlrSKbnF4_rmse = RMSE(y_V_SKbn, MlrSKbnF4)

MlrSKct_rmse   = RMSE(y_V_SKct, MlrSKct)
MlrSKctF1_rmse = RMSE(y_V_SKct, MlrSKctF1)
MlrSKctF2_rmse = RMSE(y_V_SKct, MlrSKctF2)
MlrSKctF3_rmse = RMSE(y_V_SKct, MlrSKctF3)
MlrSKctF4_rmse = RMSE(y_V_SKct, MlrSKctF4)

MlrFPbn_rmse   = RMSE(y_V_FPbn, MlrFPbn)
MlrFPbnF1_rmse = RMSE(y_V_FPbn, MlrFPbnF1)
MlrFPbnF2_rmse = RMSE(y_V_FPbn, MlrFPbnF2)
MlrFPbnF3_rmse = RMSE(y_V_FPbn, MlrFPbnF3)
MlrFPbnF4_rmse = RMSE(y_V_FPbn, MlrFPbnF4)

MlrFPct_rmse   = RMSE(y_V_FPct, MlrFPct)
MlrFPctF1_rmse = RMSE(y_V_FPct, MlrFPctF1)
MlrFPctF2_rmse = RMSE(y_V_FPct, MlrFPctF2)
MlrFPctF3_rmse = RMSE(y_V_FPct, MlrFPctF3)
MlrFPctF4_rmse = RMSE(y_V_FPct, MlrFPctF4)

MlrCOMbn_rmse   = RMSE(y_V_PC, MlrCOMbn)
MlrCOMbnF1_rmse = RMSE(y_V_PC, MlrCOMbnF1)
MlrCOMbnF2_rmse = RMSE(y_V_PC, MlrCOMbnF2)
MlrCOMbnF3_rmse = RMSE(y_V_PC, MlrCOMbnF3)
MlrCOMbnF4_rmse = RMSE(y_V_PC, MlrCOMbnF4)

MlrCOMct_rmse   = RMSE(y_V_PC, MlrCOMct)
MlrCOMctF1_rmse = RMSE(y_V_PC, MlrCOMctF1)
MlrCOMctF2_rmse = RMSE(y_V_PC, MlrCOMctF2)
MlrCOMctF3_rmse = RMSE(y_V_PC, MlrCOMctF3)
MlrCOMctF4_rmse = RMSE(y_V_PC, MlrCOMctF4)

print('> Multilinear regression validation error calculated')
Mlr_rmse = np.array([[MlrPC_rmse, MlrPCF1_rmse, MlrPCF2_rmse, MlrPCF3_rmse, MlrPCF4_rmse, 
		      MlrSKbn_rmse, MlrSKbnF1_rmse, MlrSKbnF2_rmse, MlrSKbnF3_rmse, MlrSKbnF4_rmse,
		      MlrSKct_rmse, MlrSKctF1_rmse, MlrSKctF2_rmse, MlrSKctF3_rmse, MlrSKctF4_rmse, 
		      MlrFPbn_rmse, MlrFPbnF1_rmse, MlrFPbnF2_rmse, MlrFPbnF3_rmse, MlrFPbnF4_rmse,
		      MlrFPct_rmse, MlrFPctF1_rmse, MlrFPctF2_rmse, MlrFPctF3_rmse, MlrFPctF4_rmse, 
		      MlrCOMbn_rmse, MlrCOMbnF1_rmse, MlrCOMbnF2_rmse, MlrCOMbnF3_rmse, MlrCOMbnF4_rmse,
		      MlrCOMct_rmse, MlrCOMctF1_rmse, MlrCOMctF2_rmse, MlrCOMctF3_rmse, MlrCOMctF4_rmse]])



# Lasso (l1-regularised) multilinear regression
LassoPC_rmse   = RMSE(y_V_PC, LassoPC)
LassoPCF1_rmse = RMSE(y_V_PC, LassoPCF1)
LassoPCF2_rmse = RMSE(y_V_PC, LassoPCF2)
LassoPCF3_rmse = RMSE(y_V_PC, LassoPCF3)
LassoPCF4_rmse = RMSE(y_V_PC, LassoPCF4)

LassoSKbn_rmse   = RMSE(y_V_SKbn, LassoSKbn)
LassoSKbnF1_rmse = RMSE(y_V_SKbn, LassoSKbnF1)
LassoSKbnF2_rmse = RMSE(y_V_SKbn, LassoSKbnF2)
LassoSKbnF3_rmse = RMSE(y_V_SKbn, LassoSKbnF3)
LassoSKbnF4_rmse = RMSE(y_V_SKbn, LassoSKbnF4)

LassoSKct_rmse   = RMSE(y_V_SKct, LassoSKct)
LassoSKctF1_rmse = RMSE(y_V_SKct, LassoSKctF1)
LassoSKctF2_rmse = RMSE(y_V_SKct, LassoSKctF2)
LassoSKctF3_rmse = RMSE(y_V_SKct, LassoSKctF3)
LassoSKctF4_rmse = RMSE(y_V_SKct, LassoSKctF4)

LassoFPbn_rmse   = RMSE(y_V_FPbn, LassoFPbn)
LassoFPbnF1_rmse = RMSE(y_V_FPbn, LassoFPbnF1)
LassoFPbnF2_rmse = RMSE(y_V_FPbn, LassoFPbnF2)
LassoFPbnF3_rmse = RMSE(y_V_FPbn, LassoFPbnF3)
LassoFPbnF4_rmse = RMSE(y_V_FPbn, LassoFPbnF4)

LassoFPct_rmse   = RMSE(y_V_FPct, LassoFPct)
LassoFPctF1_rmse = RMSE(y_V_FPct, LassoFPctF1)
LassoFPctF2_rmse = RMSE(y_V_FPct, LassoFPctF2)
LassoFPctF3_rmse = RMSE(y_V_FPct, LassoFPctF3)
LassoFPctF4_rmse = RMSE(y_V_FPct, LassoFPctF4)

LassoCOMbn_rmse   = RMSE(y_V_PC, LassoCOMbn)
LassoCOMbnF1_rmse = RMSE(y_V_PC, LassoCOMbnF1)
LassoCOMbnF2_rmse = RMSE(y_V_PC, LassoCOMbnF2)
LassoCOMbnF3_rmse = RMSE(y_V_PC, LassoCOMbnF3)
LassoCOMbnF4_rmse = RMSE(y_V_PC, LassoCOMbnF4)

LassoCOMct_rmse   = RMSE(y_V_PC, LassoCOMct)
LassoCOMctF1_rmse = RMSE(y_V_PC, LassoCOMctF1)
LassoCOMctF2_rmse = RMSE(y_V_PC, LassoCOMctF2)
LassoCOMctF3_rmse = RMSE(y_V_PC, LassoCOMctF3)
LassoCOMctF4_rmse = RMSE(y_V_PC, LassoCOMctF4)

print('> Lasso regression validation error calculated')
Lasso_rmse = np.array([[LassoPC_rmse, LassoPCF1_rmse, LassoPCF2_rmse, LassoPCF3_rmse, LassoPCF4_rmse, 
			LassoSKbn_rmse, LassoSKbnF1_rmse, LassoSKbnF2_rmse, LassoSKbnF3_rmse, LassoSKbnF4_rmse,
			LassoSKct_rmse, LassoSKctF1_rmse, LassoSKctF2_rmse, LassoSKctF3_rmse, LassoSKctF4_rmse, 
			LassoFPbn_rmse, LassoFPbnF1_rmse, LassoFPbnF2_rmse, LassoFPbnF3_rmse, LassoFPbnF4_rmse,
			LassoFPct_rmse, LassoFPctF1_rmse, LassoFPctF2_rmse, LassoFPctF3_rmse, LassoFPctF4_rmse, 
			LassoCOMbn_rmse, LassoCOMbnF1_rmse, LassoCOMbnF2_rmse, LassoCOMbnF3_rmse, LassoCOMbnF4_rmse,
			LassoCOMct_rmse, LassoCOMctF1_rmse, LassoCOMctF2_rmse, LassoCOMctF3_rmse, LassoCOMctF4_rmse]])



# Ridge (l2-regularised) multilinear regression
RidgePC_rmse   = RMSE(y_V_PC, RidgePC)
RidgePCF1_rmse = RMSE(y_V_PC, RidgePCF1)
RidgePCF2_rmse = RMSE(y_V_PC, RidgePCF2)
RidgePCF3_rmse = RMSE(y_V_PC, RidgePCF3)
RidgePCF4_rmse = RMSE(y_V_PC, RidgePCF4)

RidgeSKbn_rmse   = RMSE(y_V_SKbn, RidgeSKbn)
RidgeSKbnF1_rmse = RMSE(y_V_SKbn, RidgeSKbnF1)
RidgeSKbnF2_rmse = RMSE(y_V_SKbn, RidgeSKbnF2)
RidgeSKbnF3_rmse = RMSE(y_V_SKbn, RidgeSKbnF3)
RidgeSKbnF4_rmse = RMSE(y_V_SKbn, RidgeSKbnF4)

RidgeSKct_rmse   = RMSE(y_V_SKct, RidgeSKct)
RidgeSKctF1_rmse = RMSE(y_V_SKct, RidgeSKctF1)
RidgeSKctF2_rmse = RMSE(y_V_SKct, RidgeSKctF2)
RidgeSKctF3_rmse = RMSE(y_V_SKct, RidgeSKctF3)
RidgeSKctF4_rmse = RMSE(y_V_SKct, RidgeSKctF4)

RidgeFPbn_rmse   = RMSE(y_V_FPbn, RidgeFPbn)
RidgeFPbnF1_rmse = RMSE(y_V_FPbn, RidgeFPbnF1)
RidgeFPbnF2_rmse = RMSE(y_V_FPbn, RidgeFPbnF2)
RidgeFPbnF3_rmse = RMSE(y_V_FPbn, RidgeFPbnF3)
RidgeFPbnF4_rmse = RMSE(y_V_FPbn, RidgeFPbnF4)

RidgeFPct_rmse   = RMSE(y_V_FPct, RidgeFPct)
RidgeFPctF1_rmse = RMSE(y_V_FPct, RidgeFPctF1)
RidgeFPctF2_rmse = RMSE(y_V_FPct, RidgeFPctF2)
RidgeFPctF3_rmse = RMSE(y_V_FPct, RidgeFPctF3)
RidgeFPctF4_rmse = RMSE(y_V_FPct, RidgeFPctF4)

RidgeCOMbn_rmse   = RMSE(y_V_PC, RidgeCOMbn)
RidgeCOMbnF1_rmse = RMSE(y_V_PC, RidgeCOMbnF1)
RidgeCOMbnF2_rmse = RMSE(y_V_PC, RidgeCOMbnF2)
RidgeCOMbnF3_rmse = RMSE(y_V_PC, RidgeCOMbnF3)
RidgeCOMbnF4_rmse = RMSE(y_V_PC, RidgeCOMbnF4)

RidgeCOMct_rmse   = RMSE(y_V_PC, RidgeCOMct)
RidgeCOMctF1_rmse = RMSE(y_V_PC, RidgeCOMctF1)
RidgeCOMctF2_rmse = RMSE(y_V_PC, RidgeCOMctF2)
RidgeCOMctF3_rmse = RMSE(y_V_PC, RidgeCOMctF3)
RidgeCOMctF4_rmse = RMSE(y_V_PC, RidgeCOMctF4)

print('> Ridge regression validation error calculated')
Ridge_rmse = np.array([[RidgePC_rmse, RidgePCF1_rmse, RidgePCF2_rmse, RidgePCF3_rmse, RidgePCF4_rmse, 
			RidgeSKbn_rmse, RidgeSKbnF1_rmse, RidgeSKbnF2_rmse, RidgeSKbnF3_rmse, RidgeSKbnF4_rmse,
			RidgeSKct_rmse, RidgeSKctF1_rmse, RidgeSKctF2_rmse, RidgeSKctF3_rmse, RidgeSKctF4_rmse, 
			RidgeFPbn_rmse, RidgeFPbnF1_rmse, RidgeFPbnF2_rmse, RidgeFPbnF3_rmse, RidgeFPbnF4_rmse,
			RidgeFPct_rmse, RidgeFPctF1_rmse, RidgeFPctF2_rmse, RidgeFPctF3_rmse, RidgeFPctF4_rmse, 
			RidgeCOMbn_rmse, RidgeCOMbnF1_rmse, RidgeCOMbnF2_rmse, RidgeCOMbnF3_rmse, RidgeCOMbnF4_rmse,
			RidgeCOMct_rmse, RidgeCOMctF1_rmse, RidgeCOMctF2_rmse, RidgeCOMctF3_rmse, RidgeCOMctF4_rmse]])




# ElasticNet (l1/l2-regularised) multilinear regression
ElasticNetPC_rmse   = RMSE(y_V_PC, ElasticNetPC)
ElasticNetPCF1_rmse = RMSE(y_V_PC, ElasticNetPCF1)
ElasticNetPCF2_rmse = RMSE(y_V_PC, ElasticNetPCF2)
ElasticNetPCF3_rmse = RMSE(y_V_PC, ElasticNetPCF3)
ElasticNetPCF4_rmse = RMSE(y_V_PC, ElasticNetPCF4)

ElasticNetSKbn_rmse   = RMSE(y_V_SKbn, ElasticNetSKbn)
ElasticNetSKbnF1_rmse = RMSE(y_V_SKbn, ElasticNetSKbnF1)
ElasticNetSKbnF2_rmse = RMSE(y_V_SKbn, ElasticNetSKbnF2)
ElasticNetSKbnF3_rmse = RMSE(y_V_SKbn, ElasticNetSKbnF3)
ElasticNetSKbnF4_rmse = RMSE(y_V_SKbn, ElasticNetSKbnF4)

ElasticNetSKct_rmse   = RMSE(y_V_SKct, ElasticNetSKct)
ElasticNetSKctF1_rmse = RMSE(y_V_SKct, ElasticNetSKctF1)
ElasticNetSKctF2_rmse = RMSE(y_V_SKct, ElasticNetSKctF2)
ElasticNetSKctF3_rmse = RMSE(y_V_SKct, ElasticNetSKctF3)
ElasticNetSKctF4_rmse = RMSE(y_V_SKct, ElasticNetSKctF4)

ElasticNetFPbn_rmse   = RMSE(y_V_FPbn, ElasticNetFPbn)
ElasticNetFPbnF1_rmse = RMSE(y_V_FPbn, ElasticNetFPbnF1)
ElasticNetFPbnF2_rmse = RMSE(y_V_FPbn, ElasticNetFPbnF2)
ElasticNetFPbnF3_rmse = RMSE(y_V_FPbn, ElasticNetFPbnF3)
ElasticNetFPbnF4_rmse = RMSE(y_V_FPbn, ElasticNetFPbnF4)

ElasticNetFPct_rmse   = RMSE(y_V_FPct, ElasticNetFPct)
ElasticNetFPctF1_rmse = RMSE(y_V_FPct, ElasticNetFPctF1)
ElasticNetFPctF2_rmse = RMSE(y_V_FPct, ElasticNetFPctF2)
ElasticNetFPctF3_rmse = RMSE(y_V_FPct, ElasticNetFPctF3)
ElasticNetFPctF4_rmse = RMSE(y_V_FPct, ElasticNetFPctF4)

ElasticNetCOMbn_rmse   = RMSE(y_V_PC, ElasticNetCOMbn)
ElasticNetCOMbnF1_rmse = RMSE(y_V_PC, ElasticNetCOMbnF1)
ElasticNetCOMbnF2_rmse = RMSE(y_V_PC, ElasticNetCOMbnF2)
ElasticNetCOMbnF3_rmse = RMSE(y_V_PC, ElasticNetCOMbnF3)
ElasticNetCOMbnF4_rmse = RMSE(y_V_PC, ElasticNetCOMbnF4)

ElasticNetCOMct_rmse   = RMSE(y_V_PC, ElasticNetCOMct)
ElasticNetCOMctF1_rmse = RMSE(y_V_PC, ElasticNetCOMctF1)
ElasticNetCOMctF2_rmse = RMSE(y_V_PC, ElasticNetCOMctF2)
ElasticNetCOMctF3_rmse = RMSE(y_V_PC, ElasticNetCOMctF3)
ElasticNetCOMctF4_rmse = RMSE(y_V_PC, ElasticNetCOMctF4)

print('> ElasticNet regression validation error calculated')
ElasticNet_rmse = np.array([[ElasticNetPC_rmse, ElasticNetPCF1_rmse, ElasticNetPCF2_rmse, ElasticNetPCF3_rmse, ElasticNetPCF4_rmse, 
			     ElasticNetSKbn_rmse, ElasticNetSKbnF1_rmse, ElasticNetSKbnF2_rmse, ElasticNetSKbnF3_rmse, ElasticNetSKbnF4_rmse,
			     ElasticNetSKct_rmse, ElasticNetSKctF1_rmse, ElasticNetSKctF2_rmse, ElasticNetSKctF3_rmse, ElasticNetSKctF4_rmse, 
			     ElasticNetFPbn_rmse, ElasticNetFPbnF1_rmse, ElasticNetFPbnF2_rmse, ElasticNetFPbnF3_rmse, ElasticNetFPbnF4_rmse,
			     ElasticNetFPct_rmse, ElasticNetFPctF1_rmse, ElasticNetFPctF2_rmse, ElasticNetFPctF3_rmse, ElasticNetFPctF4_rmse, 
			     ElasticNetCOMbn_rmse, ElasticNetCOMbnF1_rmse, ElasticNetCOMbnF2_rmse, ElasticNetCOMbnF3_rmse, ElasticNetCOMbnF4_rmse,
			     ElasticNetCOMct_rmse, ElasticNetCOMctF1_rmse, ElasticNetCOMctF2_rmse, ElasticNetCOMctF3_rmse, ElasticNetCOMctF4_rmse]])



# SGD-minimised l1/l2-regularised multilinear regression
SgdPC_rmse   = RMSE(y_V_PC, SgdPC)
SgdPCF1_rmse = RMSE(y_V_PC, SgdPCF1)
SgdPCF2_rmse = RMSE(y_V_PC, SgdPCF2)
SgdPCF3_rmse = RMSE(y_V_PC, SgdPCF3)
SgdPCF4_rmse = RMSE(y_V_PC, SgdPCF4)

SgdSKbn_rmse   = RMSE(y_V_SKbn, SgdSKbn)
SgdSKbnF1_rmse = RMSE(y_V_SKbn, SgdSKbnF1)
SgdSKbnF2_rmse = RMSE(y_V_SKbn, SgdSKbnF2)
SgdSKbnF3_rmse = RMSE(y_V_SKbn, SgdSKbnF3)
SgdSKbnF4_rmse = RMSE(y_V_SKbn, SgdSKbnF4)

SgdSKct_rmse   = RMSE(y_V_SKct, SgdSKct)
SgdSKctF1_rmse = RMSE(y_V_SKct, SgdSKctF1)
SgdSKctF2_rmse = RMSE(y_V_SKct, SgdSKctF2)
SgdSKctF3_rmse = RMSE(y_V_SKct, SgdSKctF3)
SgdSKctF4_rmse = RMSE(y_V_SKct, SgdSKctF4)

SgdFPbn_rmse   = RMSE(y_V_FPbn, SgdFPbn)
SgdFPbnF1_rmse = RMSE(y_V_FPbn, SgdFPbnF1)
SgdFPbnF2_rmse = RMSE(y_V_FPbn, SgdFPbnF2)
SgdFPbnF3_rmse = RMSE(y_V_FPbn, SgdFPbnF3)
SgdFPbnF4_rmse = RMSE(y_V_FPbn, SgdFPbnF4)

SgdFPct_rmse   = RMSE(y_V_FPct, SgdFPct)
SgdFPctF1_rmse = RMSE(y_V_FPct, SgdFPctF1)
SgdFPctF2_rmse = RMSE(y_V_FPct, SgdFPctF2)
SgdFPctF3_rmse = RMSE(y_V_FPct, SgdFPctF3)
SgdFPctF4_rmse = RMSE(y_V_FPct, SgdFPctF4)

SgdCOMbn_rmse   = RMSE(y_V_PC, SgdCOMbn)
SgdCOMbnF1_rmse = RMSE(y_V_PC, SgdCOMbnF1)
SgdCOMbnF2_rmse = RMSE(y_V_PC, SgdCOMbnF2)
SgdCOMbnF3_rmse = RMSE(y_V_PC, SgdCOMbnF3)
SgdCOMbnF4_rmse = RMSE(y_V_PC, SgdCOMbnF4)

SgdCOMct_rmse   = RMSE(y_V_PC, SgdCOMct)
SgdCOMctF1_rmse = RMSE(y_V_PC, SgdCOMctF1)
SgdCOMctF2_rmse = RMSE(y_V_PC, SgdCOMctF2)
SgdCOMctF3_rmse = RMSE(y_V_PC, SgdCOMctF3)
SgdCOMctF4_rmse = RMSE(y_V_PC, SgdCOMctF4)

print('> SGD multilinear regression validation error calculated')
Sgd_rmse = np.array([[SgdPC_rmse, SgdPCF1_rmse, SgdPCF2_rmse, SgdPCF3_rmse, SgdPCF4_rmse, 
		      SgdSKbn_rmse, SgdSKbnF1_rmse, SgdSKbnF2_rmse, SgdSKbnF3_rmse, SgdSKbnF4_rmse,
		      SgdSKct_rmse, SgdSKctF1_rmse, SgdSKctF2_rmse, SgdSKctF3_rmse, SgdSKctF4_rmse, 
		      SgdFPbn_rmse, SgdFPbnF1_rmse, SgdFPbnF2_rmse, SgdFPbnF3_rmse, SgdFPbnF4_rmse,
		      SgdFPct_rmse, SgdFPctF1_rmse, SgdFPctF2_rmse, SgdFPctF3_rmse, SgdFPctF4_rmse, 
		      SgdCOMbn_rmse, SgdCOMbnF1_rmse, SgdCOMbnF2_rmse, SgdCOMbnF3_rmse, SgdCOMbnF4_rmse,
		      SgdCOMct_rmse, SgdCOMctF1_rmse, SgdCOMctF2_rmse, SgdCOMctF3_rmse, SgdCOMctF4_rmse]])



# Compile and process model validation results
Mlr_rmse = np.swapaxes(Mlr_rmse,0,1)
Lasso_rmse = np.swapaxes(Lasso_rmse,0,1)
Ridge_rmse = np.swapaxes(Ridge_rmse,0,1)
ElasticNet_rmse = np.swapaxes(ElasticNet_rmse,0,1)
Sgd_rmse = np.swapaxes(Sgd_rmse,0,1)
val_rmse = np.hstack((Mlr_rmse, Lasso_rmse, Ridge_rmse, ElasticNet_rmse, Sgd_rmse))
print(val_rmse)

val_rmse_avg = np.swapaxes(np.array([val_rmse.mean(axis=1)]), 0, 1)

val_rmse_ci = []
for row in val_rmse[:,0:5]:
	val_rmse_ci.append(CI(row))
val_rmse_ci = np.swapaxes(np.array([val_rmse_ci]), 0, 1)

val_rmse = np.hstack((val_rmse, val_rmse_avg, val_rmse_ci))

np.savetxt('ComparingMolRepsforLogP_BenchmarkRawResults.csv', val_rmse, delimiter=',')



print('(%.2fs)' % (time()-start7))



#=============================================================================
# (8) RESULT VISUALISATION
#=============================================================================
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap
import seaborn as sns



print('\n\n\n########## RESULT VISUALISATION ##########')
start8 = time()



class MidpointNormalize(matplotlib.colors.Normalize): #https://github.com/mwaskom/seaborn/issues/1309
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        matplotlib.colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))



# Labels
x_labels = ['OLS', 'Lasso', 'Ridge', 'ElasticNet', 'SGD']
y_labels = ['Physicochemical descriptors      [Full]', 'Physicochemical descriptors     [75%]', 'Physicochemical descriptors     [50%]', 'Physicochemical descriptors     [25%]', 'Physicochemical descriptors  [12.5%]',
			'Structural key (count)      [Full]', 'Structural key (count)     [75%]', 'Structural key (count)     [50%]', 'Structural key (count)     [25%]', 'Structural key (count)  [12.5%]',
			'Structural key (binary)      [Full]', 'Structural key (binary)     [75%]', 'Structural key (binary)     [50%]', 'Structural key (binary)     [25%]', 'Structural key (binary)  [12.5%]',
			'Circular fingerprint (count)      [Full]', 'Circular fingerprint (count)     [75%]', 'Circular fingerprint (count)     [50%]', 'Circular fingerprint (count)     [25%]', 'Circular fingerprint (count)  [12.5%]',
			'Circular fingerprint (binary)      [Full]', 'Circular fingerprint (binary)     [75%]', 'Circular fingerprint (binary)     [50%]', 'Circular fingerprint (binary)     [25%]', 'Circular fingerprint (binary)  [12.5%]',
			'Composite representation (count)      [Full]', 'Composite representation (count)     [75%]', 'Composite representation (count)     [50%]', 'Composite representation (count)     [25%]', 'Composite representation (count)  [12.5%]',
			'Composite representation (binary)      [Full]', 'Composite representation (binary)     [75%]', 'Composite representation (binary)     [50%]', 'Composite representation (binary)     [25%]', 'Composite representation (binary)  [12.5%]']


	
plt.figure(figsize=(20,16))



# Colourmap
n_grad = 8
oldcmap = cm.get_cmap('viridis_r', n_grad)
newcolors = oldcmap(np.linspace(0,1,n_grad)) # Values are a fraction of respective RGB values
print(newcolors)
newcolors[2,:] = newcolors[1,:]
newcolors[4,:] = np.array([64/255, 108/255, 166/255, 1])
print(newcolors)
newcmap = ListedColormap(newcolors)

# Heatmap
plt.subplot(1,2,1)
plt.title('(A) LogP RMSE across five algorithms', fontsize=23, fontweight='bold', position=(0.5,1.01))
sns.heatmap(data=val_rmse, annot=True, annot_kws={'fontsize':'xx-large'}, fmt='.3f', xticklabels=x_labels, yticklabels=y_labels, cmap=newcmap, norm=MidpointNormalize(midpoint=1.3), cbar=False)
plt.xticks(rotation=0)
plt.hlines(y=[5,10,15,20,25,30], xmin=0, xmax=5, colors='white', linewidth=6)
plt.xlabel('Linear regression algorithm', fontsize='xx-large') #bug re: position (x,Y): xlabel only obeys x. Use ax.x/yaxis.set_label_coords instead
plt.gca().xaxis.set_label_coords(0.52, -0.04)
plt.ylabel('Molecular representation', fontsize=23, fontweight='bold', rotation=0) #bug re: position(X,y): ylabel only obeys y; documented @ https://github.com/matplotlib/matplotlib/issues/7946
plt.gca().yaxis.set_label_coords(-0.425, 1.01)
plt.xticks(fontsize='xx-large')
plt.yticks(fontsize='xx-large', position=(-0.0425,0))

# Barplot
plt.subplot(1,2,2)
plt.title('(B) Average logP RMSE', fontsize=23, fontweight='bold', position=(0.5,1.01))
y_labels = np.arange(0,35,1)
plt.barh(width=val_rmse_avg[0:5], y=y_labels[0:5], color='lightgray', edgecolor='k', linewidth=2, hatch='\\')
plt.barh(width=val_rmse_avg[5:15], y=y_labels[5:15], color='lightgray', edgecolor='k', linewidth=2)
plt.barh(width=val_rmse_avg[15:25], y=y_labels[15:25], color='lightgray', edgecolor='k', linewidth=2, hatch='\\')
plt.barh(width=val_rmse_avg[25:35], y=y_labels[25:35], color='lightgray', edgecolor='k', linewidth=2)
plt.ylim(-0.5, 34.5)
plt.gca().invert_yaxis()
plt.errorbar(x=val_rmse_avg, y=y_labels, xerr=val_rmse_ci, ecolor='k', capsize=3, ls='none')
sns.despine(right=True)
plt.xticks(fontsize='xx-large')
plt.yticks([])
plt.xlabel('Root mean square error', fontsize='xx-large')
plt.gca().xaxis.set_label_coords(0.52, -0.04)



plt.tight_layout()
plt.savefig('ComparingMolRepsforLogP_BenchmarkVisualisedResults.png', dpi=600)



print('(%.2fs)' % (time()-start8))



print('\n\n\n********** Experiment complete **********     (%.2fs)' % (time()-start0))
