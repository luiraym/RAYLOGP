'''
Raymond Lui
rlui9522@uni.sydney.edu.au
Pharmacoinformatics Laboratory
Discipline of Pharmacology, School of Medical Sciences
Faculty of Medicine and Health, The University of Sydney
Sydney, New South Wales, 2006, Australia
'''

from time import time
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)



print('********** RAYLOGP **********     ')
start0 = time()



#=============================================================================
# (0) LOAD PACKAGES
#=============================================================================

import sys
print('\n\n> Python v%s' % sys.version)
import numpy as np
print('> NumPy v%s' % np.__version__)
import pandas as pd
print('> pandas v%s' % pd.__version__)
import sklearn
print('> scikit-learn v%s' % sklearn.__version__)



#=============================================================================
# (1) LOAD DATA
#=============================================================================
start1 = time()

# TRAINING SET
## Mansouri PHYSPROP database
Mansouri_data = pd.read_csv('Mansouri14045Train_DescriptorsFinal.csv')
Mansouri_descriptors = Mansouri_data.iloc[:,0:1438].values
Mansouri_logP = Mansouri_data.iloc[:,1438].values

# BENCHMARKING SET
## Martel UHPLC benchmarking dataset
Martel_data = pd.read_csv('Martel707Validation_DescriptorsFinal.csv')
Martel_descriptors = Martel_data.iloc[:,0:1438].values
Martel_logP = Martel_data.iloc[:,1438].values

# TEST SETS
## SAMPL6 LogP Prediction Challenge 2019 kinase inhibitor fragment-like small molecules
SAMPL6_data = pd.read_csv('SAMPL6_logP_DescriptorsFinal.csv')
SAMPL6_descriptors = SAMPL6_data.iloc[:,0:1438].values
SAMPL6_logP = pd.read_csv('SAMPL6_logP_DescriptorsFinal_w-logP.csv').iloc[:,1438].values



print('\n\n> Training set loaded: %s molecules, %s features' % (Mansouri_descriptors.shape[0], Mansouri_descriptors.shape[1]))
print('> Benchmarking set loaded: %s molecules, %s features' % (Martel_descriptors.shape[0], Martel_descriptors.shape[1]))
print('> Test set (SAMPL6) loaded: %s molecules, %s features' % (SAMPL6_descriptors.shape[0], SAMPL6_descriptors.shape[1]))



#=============================================================================
# (2) SCALE DATA
#=============================================================================
from sklearn.preprocessing import MinMaxScaler



start2 = time()

Scaler = MinMaxScaler().fit(Mansouri_descriptors)
Mansouri_descriptors = Scaler.transform(Mansouri_descriptors)
Martel_descriptors = Scaler.transform(Martel_descriptors)
SAMPL6_descriptors = Scaler.transform(SAMPL6_descriptors)

print('\n\n> Physicochemical data scaled in %.5f seconds' % (time()-start2))



#=============================================================================
# (3) RAYLOGP
#=============================================================================
from sklearn.linear_model import SGDRegressor



start3 = time()

def RAYLOGP(input_descriptors):
	
	loss =  'squared_epsilon_insensitive'
	learning_rate = 'invscaling' 
	penalty = 'none' 
	
	algorithm1 = SGDRegressor(loss=loss, learning_rate=learning_rate, penalty=penalty, random_state=69)
	algorithm2 = SGDRegressor(loss=loss, learning_rate=learning_rate, penalty=penalty, random_state=135)
	algorithm3 = SGDRegressor(loss=loss, learning_rate=learning_rate, penalty=penalty, random_state=346)
	algorithm4 = SGDRegressor(loss=loss, learning_rate=learning_rate, penalty=penalty, random_state=545)
	algorithm5 = SGDRegressor(loss=loss, learning_rate=learning_rate, penalty=penalty, random_state=978)

	model1 = algorithm1.fit(Mansouri_descriptors, Mansouri_logP)
	model2 = algorithm2.fit(Mansouri_descriptors, Mansouri_logP)
	model3 = algorithm3.fit(Mansouri_descriptors, Mansouri_logP)
	model4 = algorithm4.fit(Mansouri_descriptors, Mansouri_logP)
	model5 = algorithm5.fit(Mansouri_descriptors, Mansouri_logP)
	
	prediction1 = model1.predict(input_descriptors)
	prediction2 = model2.predict(input_descriptors)
	prediction3 = model3.predict(input_descriptors)
	prediction4 = model4.predict(input_descriptors)
	prediction5 = model5.predict(input_descriptors)
	
	final_prediction = []
	for seed1, seed2, seed3, seed4, seed5 in zip(prediction1, prediction2, prediction3, prediction4, prediction5):
		final_prediction.append(np.mean([seed1, seed2, seed3, seed4, seed5]))
	
	return np.asarray(final_prediction)

print('> RAYLOGP trained in %.5f seconds' % (time()-start3))



#=============================================================================
# (4) MAKE PREDICTIONS
#=============================================================================
def RMSE(true_logP, predicted_logP):
        mean_squared_error = np.square(np.subtract(true_logP, predicted_logP)).mean()
        return np.sqrt(mean_squared_error)



# Martel UHPLC benchmarking dataset
start4a = time()
RAYLOGP_Martel = RAYLOGP(Martel_descriptors)
df = pd.DataFrame({'RAYLOGP':RAYLOGP_Martel})
df.to_csv('RAYLOGP_Martel_predictions.csv', index=False)
print('\n\n> RAYLOGP predicted in %.2f seconds' % (time()-start4a))
print(RMSE(Martel_logP, RAYLOGP_Martel))

# SAMPL6 LogP Prediction Challenge 2019 kinase inhibitor fragment-like small molecules
start4b = time()
RAYLOGP_SAMPL6 = RAYLOGP(SAMPL6_descriptors)
df = pd.DataFrame({'RAYLOGP':RAYLOGP_SAMPL6})
df.to_csv('RAYLOGP_SAMPL6_predictions.csv', index=False)
print('> SAMPL6 logP predicted in %.2f seconds' % (time()-start4b))
print(RMSE(SAMPL6_logP, RAYLOGP_SAMPL6))



print('\n\n********** RAYLOGP execution complete **********', '(Duration: %s)'%(time()-start0))
