#iPython notebook for Indeed Data Science assignment, converted to a python script
#This script reads in a csv file of feature and labeled data fits a model to that data 

import string
import re
import csv
import numpy as np
import pandas as pd
import sklearn
from sklearn import cross_validation
from sklearn.cross_validation import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn.linear_model import Ridge
from sklearn import pipeline
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge


class LinFitTransformer(BaseEstimator, RegressorMixin): 
	'''
	Class to make a linear fit (LinearRegression) and prediction for a dataset's 'fieldname' feature
	'''   
 
	def __init__(self, fieldname):
		self.fieldname = fieldname

	def fit(self, X, y):   
		self.X = X[self.fieldname].values.reshape(-1,1)
		self.y = y
		self.model = LinearRegression()
		self.model.fit(self.X, self.y)
		return self
        
	def predict(self, X):
		self.X = X[self.fieldname].values.reshape(-1,1)
		self.prediction = self.model.predict(self.X)
		return self.prediction


class CategTransformer(BaseEstimator, RegressorMixin):
	'''
	Class to make a 'categorical fit' to the data by simply setting the predicted value to the average of all label values of that category 'fieldname' feature
    'means' (self.means) is a small dataframe containing the mean labeled value for every row with the same 'fieldname' category 
	'new_df' is the joined dataframe of original data + corresponding values of the mean data from the 'means' dataframe.  The mean values are returned as the prediction.   
	'''   
 
	def __init__(self, fieldname):
		self.fieldname = fieldname

	def fit(self, X, y):   
		self.df = X
		self.X = X[self.fieldname].values.reshape(-1,1)
		self.y = y    
		self.means = self.df.groupby([self.fieldname])['target'].mean()
		self.new_df = self.df.join(self.means, on=self.fieldname,rsuffix='_mean')
		return self
        
	def predict(self, X):
		self.df = X
		self.X = X[self.fieldname].values.reshape(-1,1)
		self.new_df = self.df.join(self.means, on=self.fieldname,rsuffix='_mean')

		#prediction column will be called 'target_mean' if 'target' was already a column
        #in the left data frame
		try:
			self.prediction = self.new_df['target_mean'].values.reshape(-1,1)
		except:
			self.prediction = self.new_df['target'].values.reshape(-1,1)
		return self.prediction



class FullModelTransformer(BaseEstimator, TransformerMixin):
	'''
	general class to make fits and transform based on other classes;     
	''' 

	def __init__(self, X):
		self.est = X
       
	def fit(self, X, y):
		self.est.fit(X, y)
		return self

	def transform(self, X):
		return self.est.predict(X)


def make_plot(train_df, blind_df):
	'''
	function to plot data from pandas dataframes
	input: two dataframes
	output: none, but two plots are generated within same directory as file
	'''

	import matplotlib
	import matplotlib.pyplot as plt

	#x and y plot numpy arrays are set to the actual and predicted values of the labeled data
	x1 = train_df['target'].values.reshape(-1,1)
	y1 = train_df['target_pred'].values.reshape(-1,1)

	#make another numpy arrays of predictions from the blind data
	y2 = blind_df['target'].values.reshape(-1,1)

	#arrays use to generate line of x=y
	xline = [0,100,200,300]
	yline = xline

	#make scatter plot
	plt.cla()
	plt.clf()
	plt.plot(x1[0], y1[0], 'ko', label='train target')
	plt.plot(x1, y1, 'ko')
	plt.plot(xline[0], yline[0], linewidth=5.0, color='b', label='line of perfect fit')
	plt.plot(xline, yline, color='b',linewidth=5.0)
	plt.xlabel('actual target')
	plt.ylabel('predicted target')
	plt.xlim([0,350])
	plt.ylim([0,350])
	plt.legend()
	plt.savefig('target.png')

	#make histogram
	plt.cla()
	plt.clf()
	plt.hist(x1, normed=1, alpha=0.2, label='train actual target')
	plt.hist(y1, normed=1, alpha=0.2, label='test predicted target')
	plt.xlabel('target')
	plt.ylabel('% of job listings')
	plt.legend()
	plt.savefig('target_hist.png')

	return


def main():
	'''
	The main function reads in data, sets up the union features
	'''
	
	train_file = 'train_data.csv'
	blind_file = 'test_features_2013-03-07.csv'

	#read csv files into a dataframe
	train_df = pd.read_csv(train_file)
	blind_df  = pd.read_csv(blind_file)

	#do print-out checks of the file
	print(train_df.shape)
	print(blind_df.shape)
	print(train_df.head(5))

	#fix column names in the blind_df column to be all lowercase to match with train_df
	for col in blind_df.columns:
		blind_df.rename(columns={col: col.lower()}, inplace=True)

	print(blind_df.head(5))

	#take out the zero-target values from the train data
	X_total = train_df[train_df['target'] > 0]

	#set the predicted y values to be the 'target' column from the train dataframe
	y_total = X_total['target']

	print(X_total.shape)
	print(y_total.shape)

	#divide train data into 'train' set and validation test set, where validation set is 20% of data from the training dataframe 
	X_train, X_test, y_train, y_test = cross_validation.train_test_split(X_total, y_total, test_size=0.2)

	#make a dictionary of weights for each step in the model pipeline
	transform_dict = {'job type':1, 'years exp': 1, 'location': 1,
                  'degree': 1, 'major':1, 'industry':1}

	#combine features into one large set 
	all_features = pipeline.FeatureUnion([
     ('job type', FullModelTransformer(CategTransformer('jobtype'))),
     ('years exp', FullModelTransformer(LinFitTransformer('yearsexperience'))),
     ('location', FullModelTransformer(LinFitTransformer('milesfrommetropolis'))),
     ('degree', FullModelTransformer(CategTransformer('degree'))),
     ('major', FullModelTransformer(CategTransformer('major'))),
     ('industry', FullModelTransformer(CategTransformer('industry')))
        ],
     transformer_weights=transform_dict)

	#make a pipeline which performs fitting to all features and then fits those predictions to a an overall model (n nearest neighbors)
	k_union = pipeline.Pipeline([
       ("features", all_features),
     ('modelfit', KNeighborsRegressor(n_neighbors=3))
 #("linreg", LinearRegression(fit_intercept=True))
    ])
	
	#fit the train data
	k_union.fit(X_train, y_train.values.reshape(-1,1))
	#print the R^2 score of the fit
	print k_union.score(X_train, y_train.values.reshape(-1,1))

	#fit the validation test data
	k_union.fit(X_test, y_test.values.reshape(-1,1))
	#print the R^2 score of the fit
	print k_union.score(X_test, y_test.values.reshape(-1,1))

	#predict on the blind data
	result = k_union.predict(blind_df)

	#add the prediction result as a column in the blind dataframe
	blind_df['target'] = result
	#write out resulting dataframe to a new csv file
	header = ["jobid", "target"]
	#blind_df.to_csv('test_target.csv', columns = header, index=False)

	#predict on the entire input dataset
	result = k_union.predict(X_total)
	X_total['target_pred'] = result

	#write resuling dataframe to new csv file
	header = ["jobid", "target_pred"]
	X_total.to_csv('train_target_pred.csv', columns = header, index=False)

	#send results to plot
	make_plot(X_total, blind_df)

	return

if __name__ == "__main__":
    main()


