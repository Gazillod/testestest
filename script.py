from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score
from sklearn import preprocessing
from sklearn import utils
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn import svm
from sklearn import neighbors
from sklearn import ensemble
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.utils import to_categorical
import numpy as np
from ast import literal_eval

import csv
with open('1feature.csv','w') as new_csv:
	fieldnames = ['feature']+['model']+['Accuracy']
	csv_writer = csv.DictWriter(new_csv, fieldnames=fieldnames)
	csv_writer.writeheader()

	feature = []
	label = []
	dataset = {}
	nam = ['size','time','deltatime','diffdelta','doublediff','protocol','bandwidth','ndestination','delta_ndestination']
	for i in range(0,9,1):
		#for j in range(i+1,9,1):
			#for k in range(j+1,9,1):
				with open('dataset.csv', 'r') as csv_file:
					csv_reader = csv.DictReader(csv_file)
					fieldnames = csv_reader.__next__()
					
				
					for row in csv_reader:
						column = [int(row['size']),float(row['time']),float(row['deltatime']),float(row['diffdelta']),float(row['doublediff']),int(row['protocol']),float(row['bandwidth']),int(row['ndestination']),int(row['delta_ndestination'])]
						feature.append([column[i]])
						label.append(row['label'])
					dataset['packet'] = feature
					dataset['label'] = label
					file_write = open('dataset.txt','w')
					fea_name = nam[i]
					file_write.write(str(dataset))
				# 	file_write.close()
				# 	csv_file.close()
					
					
				# 	with open('dataset.txt','r') as file:
				# 		reader = file.read()
				# 		data = literal_eval(reader)
						


				# 		data = data
				# 		target = data['label']
				# 		packet = data['packet']
				# 		packet_train,packet_test,target_train,target_test = train_test_split(packet,target,test_size = 0.25, random_state = 5)

				# 		classifiers = [neighbors.KNeighborsClassifier(),
				# 						svm.LinearSVC(random_state=5),
				# 						tree.DecisionTreeClassifier(criterion='gini', random_state=5),
				# 						ensemble.RandomForestClassifier(criterion='gini', random_state=5)]
				
				# 		model_name = ['K-nearest neighbors','Support vector machine with linear kernel',
				# 					'Decision tree','Random Forest']

				# 		for x in range(0,4):
				# 			clf = classifiers[x]
				# 			clf.fit(packet_train,target_train)
				# 			predict = clf.predict(packet_test)
				# 			joblib.dump(clf, fea_name+'learn'+model_name[x]+'.plk')

				# 			Acc = accuracy_score(target_test,predict)
				# 			csv_writer.writerow(dict(feature=fea_name, model=model_name[x], Accuracy=Acc))
				# 		file.close()
				# new_csv.close()
							
					
					
					