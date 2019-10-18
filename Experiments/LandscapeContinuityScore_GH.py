'''
30 JUNE 2019 20:47:37

Raymond Lui
rlui9522@uni.sydney.edu.au
Pharmacoinformatics Laboratory
Discipline of Pharmacology, School of Medical Sciences
Faculty of Medicine and Health, The University of Sydney
Sydney, New South Wales, 2006, Australia
'''


import os
import sys
import itertools
import statistics
import numpy as np
from time import time
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D




def Landscaper(
	features,
	response,
	response_name="",
	landscape_type="2D",
	scatter_plot=False,
	scatter_labels=None,
	DPI=600,
	verbose=False
	):

# Documentation:
# features = An array of the molecular features as described by e.g. descriptors or fingeprints; 
# response = An array of the endpoints being modelled e.g. biological activity or chemical property;
# response_name = The string name of the endpoint being modelled e.g. "IC50" or "LogP";
# landscape_type = Create a "2D" flat landscape or "3D" rotating landscape;
# scatter_plot = 
# scatter_labels = 
# DPI = Dots per inch of saved landscape image (default 600 dpi).
	
	
	
	# LOAD AND CHECK DATA
	def AreEqual(array1, array2):
		if len(array1) != len(array2):
			sys.exit('\n\nLandscaperError: No. of feature and response instances do not match; ensure both arrays are the same length.\n\n')
		else:
			None
	
	start1 = time()
	AreEqual(features, response)
	
	if verbose == True:
		print('\n\n==================== Landscaper ====================')
		print('[###------] %s INSTANCES, %s FEATURES LOADED (%.3fs)' % (features.shape[0], features.shape[1], (time()-start1)))
	else:
		None
	
	
	
	# TRANSFORM FEATURE SPACE TO TWO DIMENSIONS
	start2 = time()
	
	if features.shape[1] > 2:
		features = PCA(n_components=2, random_state=18).fit_transform(features)
		if verbose == True:
			print('[######---] FEATURES TRANSFORMED (%.3fs)' % (time()-start2))
			print('      *%s instances with %s principal components' % (features.shape[0],features.shape[1]))
		else:
			None
	elif features.shape[1] < 2:
		sys.exit('\n\LandscaperError: No. of features less than 2 are not currently supported; ensure there are two or more features.\n\n')
	else:
		print('[######---] FEATURES READY TO PLOT')
	
	
		
	# PLOT LANDSCAPE
	start3 = time()
	
	from matplotlib import cm
	from matplotlib.colors import ListedColormap
	n_grad = 7
	oldcmap = cm.get_cmap('viridis_r', n_grad)
	newcolors = oldcmap(np.linspace(0,1,n_grad)) #Values are a fraction of respective RGB values
	print(newcolors)
	print(newcolors)
	newcmap = ListedColormap(newcolors)

	if landscape_type == '2D':
		plt.figure()
		surf = plt.tricontourf(features[:,0], features[:,1], response, cmap=newcmap)
		plt.xlim((min(features[:,0])-0.5), max((features[:,0])+0.5)) #Switch min- and max+ to flip axis
		plt.ylim((min(features[:,1])-0.5), max((features[:,1])+0.5))
		plt.xlabel('Principal Component 1')
		plt.ylabel('Principal Component 2')
		cb = plt.colorbar(surf, fraction=0.03, label=response_name)
		cb.ax.yaxis.set_ticks_position('right')
		cb.ax.yaxis.set_label_position('right')
		if scatter_plot == True:
			plt.plot(features[:,0], features[:,1], 'ko', ms=1)
			if scatter_labels == None:
				None
			else:
				if len(scatter_labels) != len(features):
					sys.exit('\n\LandscapeError: No. of instances and labels do not match; ensure both arrays are the same length.\n\n')
				else:
					scatter_labels = np.asarray(scatter_labels)
					for label_right,x,y in zip(scatter_labels[[0,1,2,4,7]], features[[0,1,2,4,7],0], features[[0,1,2,4,7],1]):
						plt.annotate(label_right, xy=(x,y), fontsize='xx-small', ha='right')
					for label_left,x,y in zip(scatter_labels[[3,6,5,8,9,10]], features[[3,6,5,8,9,10],0], features[[3,6,5,8,9,10],1]):
						plt.annotate(label_left, xy=(x,y), fontsize='xx-small', ha='left')
		else:
			None
		plt.savefig('2DLandscape_%s.png'%response_name, dpi=DPI)
	
	elif landscape_type == '3D':
		ax = plt.axes(projection='3d')
		ax.plot_trisurf(features[:,0], features[:,1], response, cmap=newcmap)
		ax.set_xlabel('Principal Component 1')
		ax.set_ylabel('Principal Component 2')
		ax.set_zlabel(response_name)
		
		path = os.getcwd()+'/3DLandscapes/'
		try:
			os.mkdir(path)
		except FileExistsError:
			sys.exit('\nLandscapeError: A folder named "3DLandscapes" exists in the current directory; please delete or relocate')
		else:
			None
		for angle in range(0,360,3):
			ax.view_init(azim=angle)
			filename = path+'3DLandscape_%s_angle'%response_name+str(angle)+'.png'
			ax.figure.savefig(filename, dpi=DPI)
	
	else:
		raise ValueError('landscape_type not supported; select one from "2D" or "3D".')
	
	if verbose == True:
		print('[#########] %s LANDSCAPE PLOTTED AND SAVED (%.3fs)' % (landscape_type, (time()-start3)))
		print('=======================================================\n\n')
	else:
		None
	
	
	
	return None






def ContinuityIndex(
	features,
	response,
	variable_type="",
	similarity_metric="dice",
	verbose=False
	):

# Documentation: 
# features = An array of the molecular features as described by e.g. descriptors or fingeprints; 
# response = An array of the endpoints being modelled e.g. biological activity or chemical property; 
# variable_type = "binary" or "continuous" feature variables. If continuous, ensure data is scaled appropriately;
# similarity_metric = "dice", "tanimoto", or "cosine" coefficients.
	
	
	
	# LOAD AND CHECK DATA
	def AreEqual(Array1, Array2):
		if len(Array1) != len(Array2):
			sys.exit('\n\nContinuityIndexError: No. of feature and response instances do not match; ensure both arrays are the same length.\n\n')
		else:
			None
	
	start1 = time()
	AreEqual(features, response)
	
	if verbose == True:
		print('\n\n====================== ContinuityScore ======================')
		print('[##--------] %s INSTANCES, %s FEATURES LOADED (%.3fs)' % (features.shape[0], features.shape[1], (time()-start1)))
	else:
		None
	
	
	
	# GENERATE PAIRWISE COMBINATIONS
	start2 = time()
	
	pairwise_features = list(itertools.combinations(features,2))
	pairwise_features = np.asarray(pairwise_features)
	
	pairwise_response = list(itertools.combinations(response,2))
	pairwise_response = np.asarray(pairwise_response)
	
	if verbose == True:
		if len(pairwise_features) == len(pairwise_response):
			print('[####------] %s PAIRWISE COMBINATIONS GENERATED (%.3fs)' % (pairwise_features.shape[0], (time()-start2)))
		else:
			raise ValueError('No. of feature and response pairwise combinations do not match; ensure both arrays are the same length')
	else:
		None
	
	
	
	# DEFINE FEATURE VARIABLE TYPE
	if variable_type == 'binary':
		def One(molecule):
			return sum(1 for i in molecule if i == 1)
		def Both(molecule1, molecule2):
			return sum(1 for i,j in zip(molecule1,molecule2) if i == j == 1)
	
	elif variable_type == 'continuous':
		def One(molecule):
			return sum(i*i for i in molecule)
		def Both(molecule1, molecule2):
			return sum(i*j for i,j in zip(molecule1,molecule2))
	
	else:
		raise ValueError('variable_type not supported; select one from "Binary" or "Continuous".')
	
	
	
	# DEFINE SIMILARITY METRICS
	def Dice(molecule1, molecule2):
		return (2*Both(molecule1,molecule2)) / (One(molecule1) + One(molecule2))
	
	def Tanimoto(molecule1, molecule2):
		return (Both(molecule1,molecule2)) / (One(molecule1) + One(molecule2) - Both(molecule1,molecule2))
	
	def Cosine(Molecule1, Molecule2):
		return (Both(molecule1,molecule2)) / np.sqrt(One(molecule1) * One(molecule2))   
	
	
	
	# COMPUTE SIMILARITY
	start3 = time()
	
	pairwise_similarity = []
	for i in pairwise_features:
		if similarity_metric == 'dice':
			pairwise_similarity.append(Dice(i[0],i[1]))
		elif similarity_metric == 'tanimoto':
			pairwise_similarity.append(Tanimoto(i[0],i[1]))
		elif similarity_metric == 'cosine':
			pairwise_similarity.append(Cosine(i[0],i[1]))
		else:
			raise ValueError('similarity_metric not supported; select one from "dice", "tanimoto", or "cosine".')
	
	if verbose == True:
		print('[######----] %s %s SIMILARITY COMPUTED (%.3fs)' % (variable_type.upper(), similarity_metric.upper(), (time()-start3)))
		print('      *Average similarity: %.5f' % statistics.mean(pairwise_similarity))
	else:
		None
	
	
	
	# COMPUTE RESPONSE DIFFERENCE
	def ResponseDifference(molecule1, molecule2):
		return np.absolute(molecule1 - molecule2)
	def Scaler(data):
		scaled_data = []
		for i in data:
			scaled_data.append((i - data.min()) / (data.max() - data.min()))
		return scaled_data
	
	start4 = time()
	
	pairwise_response_difference = []
	for i in pairwise_response:
		pairwise_response_difference.append(float((ResponseDifference(i[0], i[1]))))
	pairwise_response_difference = Scaler(np.asarray(pairwise_response_difference))
	
	if verbose == True:
		print('[########--] RESPONSE DIFFERENCE COMPUTED (%.3fs)' % (time()-start4))
		print('      *Average response difference: %.5f' % statistics.mean(pairwise_response_difference))
	else:
		None
	
	
	
	# COMPUTE CONTINUITY SCORE
	def ContinuityScore(similarity, delta_response):
		return similarity * (1 - delta_response)
	
	start5 = time()
	
	pairwise_continuity = []
	for similarity, delta_response in zip(pairwise_similarity, pairwise_response_difference):
		pairwise_continuity.append(ContinuityScore(similarity, delta_response))
	output = statistics.mean(pairwise_continuity)
	
	if verbose == True:
		print('[##########] CONTINUITY INDEX COMPUTED (%.3fs)' % (time()-start5))
		print('      *Continuity index: %.5f' % output)
		print('================================================================\n\n')
	else:
		None
	
	
	
	return output






#=============================================================================
# MAIN
#=============================================================================
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load data
Martel_descriptors = pd.read_csv('Martel707Validation_DescriptorsFinal.csv').iloc[:,0:1438].values
Martel_structuralkey = pd.read_csv('Martel707Validation_StructuralKeyFinal.csv').iloc[:,0:1354].values
Martel_skct = pd.read_csv('Martel707Validation_StructuralKeyFinal_counts.csv').iloc[:,0:1354].values
Martel_fingerprint = pd.read_csv('Martel707Validation_MorganFP1024Final.csv').iloc[:,0:1024].values
Martel_fpct = pd.read_csv('Martel707Validation_MorganFP1024Final_counts.csv').iloc[:,0:1024].values
Martel_logP = pd.read_csv('Martel707Validation_DescriptorsFinal.csv').iloc[:,1438].values

# Scale continuous data
Martel_descriptors = MinMaxScaler().fit(pd.read_csv('Mansouri14045Train_DescriptorsFinal.csv').iloc[:,0:1438].values).transform(Martel_descriptors)
Martel_skct = MinMaxScaler().fit(pd.read_csv('Mansouri14050Train_StructuralKeyFinal_counts.csv').iloc[:,0:1354].values).transform(Martel_skct)
Martel_fpct = MinMaxScaler().fit(pd.read_csv('Mansouri13710Train_MorganFP1024Final_counts.csv').iloc[:,0:1024].values).transform(Martel_fpct)

# Compute landscape and continuity score
features = Martel_descriptors
response = Martel_logP

print('Martel physicochemical descriptors')
Landscaper(features, response, response_name='LogP', landscape_type='2D')
index = ContinuityIndex(features, response, variable_type='continuous', verbose=True)

print('Landscape saved. Continuity index = ', index)
