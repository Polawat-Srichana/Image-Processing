#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#pip install imbalanced-learn


# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import imblearn
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import seaborn as sns
sns.set()


# In[2]:


# Load data into  dataframe
data=pd.read_csv('data2022.student.csv')


# In[3]:


data


# In[4]:


data.describe()


# In[5]:


# Data Types
print(data.dtypes)
# Drop ID  
ID = data.pop('ID')
num_data = data.select_dtypes(exclude='object')


# In[6]:


# 1 unique values
for attr in data.columns:
  if data[attr].unique().shape[0] == 1:
    data.drop(attr, axis=1, inplace=True)


# In[7]:


# Check the duplicate
duplicate = data.duplicated(subset=None, keep='first')

# Check duplicate rows
duplicate_rows = np.where(duplicate ==True)[0]
# Check duplicate columns
duplicate_column = []
for i in range(data.shape[1]):
  for j in range(i+1, data.shape[1]):
    # Compare each value 
    equality = data.iloc[:,i] == data.iloc[:,j]
    equal_values = np.array(np.where(equality==True))
    if( equal_values.size == data[data.columns[i]].size ):
      print(f'{data.columns[j]}  duplicate with column  {data.columns[i]}')
      duplicate_column.append(data.columns[j])    
     


# In[8]:


#duplicate row
duplicate_rows


# In[9]:


data.drop(duplicate_column, axis=1, inplace=True)


# In[10]:


#drop duplicate
data.drop_duplicates(inplace=True)


# In[11]:


# Missing data
for attr in data.columns[1:]:
  missing  = data.loc[:,attr].isna()
  missing_data = np.array(np.where(missing==True))
  if (missing_data.size > 0):
    print(f'{attr} has {missing_data.size} missing data')

print()



# In[12]:


# Drop column C12 and C24
data.drop(['C12', 'C24'], axis=1, inplace=True)


# In[13]:


# Fill in the missing values in C3,C30 which are float24 and C2 & C3 which are abjective
data['C30'].fillna(int(float(data['C30'].mode()[0])), inplace=True)
for attr in ['C2', 'C23']:
  data[attr].fillna(data[attr].mode()[0], inplace=True)


# In[14]:


#Check Correlation
sns.heatmap(data.iloc[:,:].corr())


# In[15]:


#C27 and C3 has Correlation at 1 so we drop C27
data.drop(['C3'], axis=1, inplace=True)


# In[16]:


#Correlation plot
sns.heatmap(data.iloc[:,:].corr())


# In[17]:


# scale data
scaledata = data[['C5', 'C10','C16','C28']]
#Scaling
datascale = StandardScaler().fit_transform(scaledata)
data[scaledata.columns] = datascale

category_list = list(data.select_dtypes(include='object').columns)
# Label Encoding the categorical data
for attr in category_list:
  data_categorical = data.loc[:,attr]
  # Getting the Label Encoder and converting the data
  le = LabelEncoder()
  data_categorical = le.fit_transform(data_categorical)
  data[attr] = data_categorical


# In[18]:


data


# In[19]:


data.describe()


# In[20]:


# Plot tha data distribution
for attr in data.columns[1:]:
    graph = sns.distplot(data[attr])
plt.show()
plt.savefig(f'preparationprocess{attr}plot.png')
plt.close()


# In[21]:


train_data = data.iloc[:900]
test_data = data.iloc[900:]

# SMOTE
sm = SMOTE(random_state=42)
un =RandomUnderSampler(random_state=42) 
# Getting the class
y = train_data.Class
X = train_data.drop('Class', axis=1)
X_columns = X.columns
# Applying SMOTE
X, y = sm.fit_resample(X, y)
X, y = un.fit_resample(X, y)
testY = test_data.Class 
testX = test_data.drop('Class', axis=1)
X = pd.DataFrame(X)
X.columns = X_columns
y = pd.DataFrame(y)
y.columns = ['Class']
# Plotting the attributes
#for attr in X.columns:
   #graph = sns.distplot(X[attr])
   #plt.show()
   #plt.savefig(f'SMOTE{attr}plot.png')
   #plt.close()

# Splpit data
trainX, ValidX, trainY, ValidY = train_test_split(X, y, test_size=0.1, stratify=y)


# Convert data
trainX = trainX.to_numpy()
trainY = trainY.to_numpy()
trainY = trainY.ravel()




# In[28]:


data.Class.hist()


# In[29]:


y.Class.hist()


# In[47]:


# Funtion to train the models
def train_model(model, trainX, trainY, ValidX, ValidY):
  model.fit(trainX, trainY)
  predict_validate = model.predict(ValidX)
  accuracy = accuracy_score(ValidY, predict_validate)
  f_measure = f1_score(predict_validate, ValidY)
  c_matrix = confusion_matrix(predict_validate, ValidY)
  return accuracy, f_measure, c_matrix

# Function to do Cross Validation of models
def crossValidation(model, trainX, y_train):
  # Defining Stratified K Fold
  skf = StratifiedKFold(n_splits=10,shuffle=True)
  accuracies = []
  f_measures = []
  c_matrices = []

  for train, valid in skf.split(trainX, trainY):
    # Selecting the training and validation data
    trainX_split = trainX[train]
    trainY_split = trainY[train]
    ValidX_split = trainX[valid]
    ValidY_split = trainY[valid]
    accuracy, f_measure, c_matrix = train_model(model, trainX_split, trainY_split, ValidX_split, ValidY_split)
    accuracies.append(accuracy)
    f_measures.append(f_measure)
    c_matrices.append(c_matrix)
  return np.mean(accuracies), np.mean(f_measures), c_matrices
#Plotting
def get_plot(values, title, x_label, y_label, fileName):
  plt.plot(values)
  plt.title(title)
  plt.xlabel(x_label)
  plt.ylabel(y_label)
  plt.savefig(fileName)
  plt.close()

# Function to get the model with best values of accuracy, F Measure and Precision
def best_model(models, trainX, trainY, name):
  best_f_measure, best_accuracy, = 0, 0
  best_c_matrices = []
  accuracies, f_measures = [], []
  for i, model in enumerate(models):
    accuracy, f_measure, c_matrices = crossValidation(model, trainX, trainY)
    accuracies.append(accuracy)
    f_measures.append(f_measure)
    if f_measure > best_f_measure:
      best_accuracy = accuracy
      best_f_measure = f_measure
      best_c_matrices = c_matrices
      position = i+1
    #if(len(models) > 1):
        #get_plot(accuracies, f'Accuracy of {name} Models', 'Model = x+1', 'Accuracy', name+ 'Accuracy.png')
        #get_plot(f_measures, f'F Measure of {name} Models', 'Model = x+1', 'F Measure', name+'FMeasure.png')

  return best_accuracy, best_f_measure, position, best_c_matrices



# In[25]:


#KNN hyperparamter from gridsearch
#kn=KNeighborsClassifier()
#param_grid = {'n_neighbors':[1,2,3,4,5,None],'n_jobs':[-1,1,2,None],'weights':['distance',None],'p':[1,2,3,4,None]}
#KN=GridSearchCV(estimator=kn,param_grid=param_grid)
#accuracy, f_measure, position, c_matrices = best_model([KN], trainX, trainY, 'KNN')
#accuracy, f_measure = '{:.1f}%'.format(accuracy*100), '{:.1f}%'.format(f_measure*100)
#print('\nThe best KNN on cross validation is')
#print(f'Accuracy: {accuracy}, f Measure: {f_measure}, Model {position}')
#for matrix in c_matrices:
  #print(matrix)



# In[26]:


#Check the best param for KNN
#KN.best_params_
#{'n_jobs': -1, 'n_neighbors': 2, 'p': 1, 'weights': None}


# In[48]:


#KNN hyper paramter
kn1= KNeighborsClassifier(n_jobs= -1, n_neighbors= 3, p= 1,weights= 'distance')
kn2= KNeighborsClassifier(n_jobs= -1, n_neighbors= 2, p= 1,weights= None)
kn3= KNeighborsClassifier(n_jobs= 2, n_neighbors= 3, p= 3,weights= 'distance')
kn=[kn1,kn2,kn3]
# accuracy, F-Measure, confusion matrix 
accuracy, f_measure, position, c_matrices = best_model(kn, trainX, trainY, 'KNN')
accuracy, f_measure = '{:.1f}%'.format(accuracy*100), '{:.1f}%'.format(f_measure*100)
print('\nThe best KNN on cross validation is')
print(f'Accuracy: {accuracy}, f Measure: {f_measure}, Model {position}')
for matrix in c_matrices:
  print(matrix)


# In[49]:


# Naive Bayes hyperameter 
gnb1= GaussianNB()
gnb2= GaussianNB(var_smoothing=0.0003)
gnb3= GaussianNB(var_smoothing=3e-9)
gnb=[gnb1,gnb2,gnb3]
# accuracy, F-Measure, confusion matrix 
accuracy, f_measure, position, c_matrices = best_model(gnb, trainX, trainY, 'GNB')
accuracy, f_measure = '{:.1f}%'.format(accuracy*100), '{:.1f}%'.format(f_measure*100)
print('\nThe best NV on cross validation is')
print(f'Accuracy: {accuracy}, f Measure: {f_measure}, Model {position}')
for matrix in c_matrices:
  print(matrix)


# In[39]:


#Decision Tree hyperparameter from gridsearch
#dt=DecisionTreeClassifier(random_state=42)
#param_grid = {'max_features':['auto','sqrt','log2',None],'max_depth':[9,10,12],'criterion':['gini','entropy',None],'min_samples_split':[9,10,20,None],'splitter':['random',None]}
#DT=GridSearchCV(estimator=dt,param_grid=param_grid)
# Getting the accuracy, F Measure, model and confusion matrix and displaying it
#accuracy, f_measure, position, c_matrices = best_model([DT], trainX, trainY, 'DT')
#accuracy, f_measure = '{:.1f}%'.format(accuracy*100), '{:.1f}%'.format(f_measure*100)
#print('\nThe best DT on cross validation is')
#print(f'Accuracy: {accuracy}, f Measure: {f_measure}, Model {position}')
#for matrix in c_matrices:
  #print(matrix)



# In[40]:


#Check the best param for DT from gridsearch
#DT.best_params_
#{random_state=42}


# In[50]:


#  Decision Tree hyperparameter
dt1= DecisionTreeClassifier(criterion ='entropy',max_depth= 12,max_features= None,min_samples_split= 9,splitter='random',random_state=42)
dt2= DecisionTreeClassifier(criterion ='gini',max_depth= 9,max_features= 'auto',min_samples_split= 10,splitter='random',random_state=42)
dt3=DecisionTreeClassifier(random_state=42)
DT=[dt1,dt2,dt3]
# accuracy, F-Measure, confusion matrix 
accuracy, f_measure, position, c_matrices = best_model(DT, trainX, trainY, 'DT')
accuracy, f_measure = '{:.1f}%'.format(accuracy*100), '{:.1f}%'.format(f_measure*100)
print('\nThe best DT on cross validation is')
print(f'Accuracy: {accuracy}, f Measure: {f_measure}, Model {position}')
for matrix in c_matrices:
  print(matrix)


# In[51]:


# Gradient Boosting hyperparameter from gridsearch
#gb=GradientBoostingClassifier(random_state=42)
#param_grid = {'n_estimators':[400],'learning_rate':[0.5],'max_features':['auto','sqrt','log2',None],'max_depth':[20],'min_samples_split':[20],'criterion':['friedman_mse']}
#GB=GridSearchCV(estimator=gb,param_grid=param_grid)
#accuracy, f_measure, position, c_matrices = best_model([GB], trainX, trainY, 'GB')
#accuracy, f_measure = '{:.1f}%'.format(accuracy*100), '{:.1f}%'.format(f_measure*100)
#print('\nThe best XGB on cross validation is')
#print(f'Accuracy: {accuracy}, f Measure: {f_measure}, Model {position}')
#for matrix in c_matrices:
  #print(matrix)


# In[52]:


#Check the best param for GB from gridsearch
#GB.best_params_
#{'criterion': 'friedman_mse',
 #'learning_rate': 0.5,
# 'max_depth': 20,
 #'max_features': 'auto',
 #'min_samples_split': 20,
 #'n_estimators': 400}


# In[51]:


#  Gradient Boosting hyperparameter
gb1= GradientBoostingClassifier(n_estimators=1000,learning_rate=0.05,max_features='auto',max_depth=20,min_samples_split=20,criterion='friedman_mse',random_state=42)
gb2= GradientBoostingClassifier(n_estimators=100,learning_rate=0.05,max_features='auto',max_depth=20,min_samples_split=20,criterion='squared_error',random_state=42)
gb3=GradientBoostingClassifier(random_state=42)
GB=[gb1,gb2,gb3]
# accuracy, F-Measure, confusion matrix 
accuracy, f_measure, position, c_matrices = best_model(GB, trainX, trainY, 'GB')
accuracy, f_measure = '{:.1f}%'.format(accuracy*100), '{:.1f}%'.format(f_measure*100)
print('\nThe best GB on cross validation is')
print(f'Accuracy: {accuracy}, f Measure: {f_measure}, Model {position}')
for matrix in c_matrices:
  print(matrix)


# In[37]:


# Define the Random Forest models

#rf=RandomForestClassifier(random_state=42)
#param_grid = {'n_estimators':[100,140,180,None],'max_features':['auto','sqrt','log2',None],'max_depth':[12,16,18,None],'criterion':['gini','entropy',None],'n_jobs':[-1],'bootstrap':[False,True,None],'min_samples_split':[12,14,16,18]}
#RF=GridSearchCV(estimator=rf,param_grid=param_grid)
# Getting the accuracy, F Measure, model and confusion matrix and displaying it
#accuracy, f_measure, position, c_matrices = best_model([RF], trainX, trainY, 'RF')
#accuracy, f_measure = '{:.1f}%'.format(accuracy*100), '{:.1f}%'.format(f_measure*100)
#print('\nThe best RF on cross validation is')
#print(f'Accuracy: {accuracy}, f Measure: {f_measure}, Model {position}')
#for matrix in c_matrices:
  #print(matrix)


# In[89]:


#Check the best param for RF
#RF.best_params_
#{'bootstrap': False,
 #'criterion': 'entropy',
 #'max_depth': 18,
 #'max_features': 'auto',
 #'min_samples_split': 12,
 #'n_estimators': 100,
 #'n_jobs': -1}
#random state = 42


# In[52]:


#  Random Forest hyperparameter
rf1= RandomForestClassifier(n_estimators=100, max_depth=7, random_state=0)
rf2= RandomForestClassifier(n_estimators=100,max_features='sqrt',max_depth=12,criterion='entropy',n_jobs=-1,bootstrap=True,min_samples_split=12,random_state=42)
rf3=RandomForestClassifier(n_estimators=200, min_samples_split=3, random_state=42,max_features= 'auto',criterion= 'entropy',bootstrap=False,max_depth=18,n_jobs= -1)
RF=[rf1,rf2,rf3]
# accuracy, F-Measure, confusion matrix 
accuracy, f_measure, position, c_matrices = best_model(RF, trainX, trainY, 'RF')
accuracy, f_measure = '{:.1f}%'.format(accuracy*100), '{:.1f}%'.format(f_measure*100)
print('\nThe best RF on cross validation is')
print(f'Accuracy: {accuracy}, f Measure: {f_measure}, Model {position}')
for matrix in c_matrices:
  print(matrix)


# In[53]:


# Classifiaction of test data on all the classifiers

print('\nClassifier on Validation data\n')
# KNN Model
kn = KNeighborsClassifier(n_jobs= -1, n_neighbors= 2, p= 1,weights= None)
# Getting the accuracy, F Measure and confusion matrix and displaying it
accuracy, f_measure, c_matrix = train_model(kn, trainX, trainY, ValidX, ValidY)
print('\nKNN Model: Accuracy: {:.1f}%'.format(accuracy*100), 'F Measure: {:.1f}%'.format(f_measure*100))
print(c_matrix)

# Decison Tree Model
dt = DecisionTreeClassifier(random_state=42)
# Getting the accuracy, F Measure and confusion matrix and displaying it
accuracy, f_measure, c_matrix = train_model(dt, trainX, trainY,ValidX, ValidY)
print('Decison Tree Model: Accuracy: {:.1f}%'.format(accuracy*100), 'F Measure: {:.1f}%'.format(f_measure*100))
print(c_matrix)


# Navie Bayes Model
gnb = GaussianNB(var_smoothing=0.0003)
# Getting the accuracy, F Measure and confusion matrix and displaying it
accuracy, f_measure, c_matrix = train_model(gnb, trainX, trainY,ValidX, ValidY)
print('Naive Bayes Model: Accuracy: {:.1f}%'.format(accuracy*100), 'F Measure: {:.1f}%'.format(f_measure*100))
print(c_matrix)

# Random Forest Model
rf= RandomForestClassifier(n_estimators=200, min_samples_split=3, random_state=42,max_features= 'auto',criterion= 'entropy',bootstrap=False,max_depth=18,n_jobs= -1)
# Getting the accuracy, F Measure and confusion matrix and displaying it
accuracy, f_measure, c_matrix = train_model(rf, trainX, trainY, ValidX, ValidY)
print('Random Forest Model: Accuracy: {:.1f}%'.format(accuracy*100), 'F Measure: {:.1f}%'.format(f_measure*100))
print(c_matrix)

# Gradient Boosting Model
gb = GradientBoostingClassifier(n_estimators=1000,learning_rate=0.05,max_features='auto',max_depth=20,min_samples_split=20,criterion='friedman_mse',random_state=42)
accuracy, f_measure, c_matrix = train_model(gb, trainX, trainY, ValidX, ValidY)
print('Gradient Boosting Model: Accuracy: {:.1f}%'.format(accuracy*100), 'F Measure: {:.1f}%'.format(f_measure*100))
print(c_matrix)


# In[56]:


# Classification of test data and obtain plot

print("\nComparing the best classifiers\n")

print('Classifaction of the validation data\n')

accuracies, f_measures = [], []

# Random Forest Model
rf= RandomForestClassifier(n_estimators=200, min_samples_split=3, random_state=42,max_features= 'auto',criterion= 'entropy',bootstrap=False,max_depth=18,n_jobs= -1)
# Getting the accuracy, F Measure and confusion matrix and displaying it
accuracy, f_measure, c_matrix = train_model(rf, trainX, trainY, ValidX, ValidY)
accuracies.append(accuracy)
f_measures.append(f_measure)
print('Random Forest Model: Accuracy: {:.1f}%'.format(accuracy*100), 'F Measure: {:.1f}%'.format(f_measure*100))
print(c_matrix)

# Gradient Boosting Model
gb = GradientBoostingClassifier(n_estimators=1000,learning_rate=0.05,max_features='auto',max_depth=20,min_samples_split=20,criterion='friedman_mse',random_state=42)
accuracy, f_measure, c_matrix = train_model(gb, trainX, trainY, ValidX, ValidY)
accuracies.append(accuracy)
f_measures.append(f_measure)
print('Gradient Boosting Model: Accuracy: {:.1f}%'.format(accuracy*100), 'F Measure: {:.1f}%'.format(f_measure*100))
print(c_matrix)
# KNN Model
kn = KNeighborsClassifier(n_jobs= -1, n_neighbors= 2, p= 1,weights= None)
# Getting the accuracy, F Measure and confusion matrix and displaying it
accuracy, f_measure, c_matrix = train_model(kn, trainX, trainY, ValidX, ValidY)
accuracies.append(accuracy)
f_measures.append(f_measure)
print('\nKNN Model: Accuracy: {:.1f}%'.format(accuracy*100), 'F Measure: {:.1f}%'.format(f_measure*100))
print(c_matrix)

# Decison Tree Model
dt = DecisionTreeClassifier(random_state=42)
# Getting the accuracy, F Measure and confusion matrix and displaying it
accuracy, f_measure, c_matrix = train_model(dt, trainX, trainY,ValidX, ValidY)
accuracies.append(accuracy)
f_measures.append(f_measure)
print('Decison Tree Model: Accuracy: {:.1f}%'.format(accuracy*100), 'F Measure: {:.1f}%'.format(f_measure*100))
print(c_matrix)


# Navie Bayes Model
gnb = GaussianNB(var_smoothing=0.0003)
# Getting the accuracy, F Measure and confusion matrix and displaying it
accuracy, f_measure, c_matrix = train_model(gnb, trainX, trainY,ValidX, ValidY)
accuracies.append(accuracy)
f_measures.append(f_measure)
print('Naive Bayes Model: Accuracy: {:.1f}%'.format(accuracy*100), 'F Measure: {:.1f}%'.format(f_measure*100))
print(c_matrix)


# Plotting the accuracies and F Measures
# Plotting the accuracies and F Measures
#get_plot(accuracies, 'Accuracy of models', 'Model = x+1', 'Accuracy', 'Accuracy.png')
#get_plot(f_measures, 'F Measure of models', 'Model = x+1', 'F Measure', 'F1.png')




# In[60]:


# Cross validation on all classifier and obtain plot

print("\nCross Validation of  the best classifiers\n")

accuracies, f_measures = [], []


# Define the random forest Model
rf= RandomForestClassifier(n_estimators=200, min_samples_split=3, random_state=42,max_features= 'auto',criterion= 'entropy',bootstrap=False,max_depth=18,n_jobs= -1)
# Getting the accuracy, F Measure and confusion matrix and displaying it
accuracy, f_measure, position, c_matrices = best_model([rf], trainX, trainY, 'RF')
accuracies.append(accuracy)
f_measures.append(f_measure)
print('\nRF : Best Accuracy: {:.1f}%'.format(accuracy*100), 'Best F Measure: {:.1f}%'.format(f_measure*100))
print('Confusion Matrix')
for matrix in c_matrices:
  print(matrix)

# Define the Gradient boosting
gb = GradientBoostingClassifier(n_estimators=1000,learning_rate=0.05,max_features='auto',max_depth=20,min_samples_split=20,criterion='friedman_mse',random_state=42)
# Getting the accuracy, F Measure and confusion matrix and displaying it
accuracy, f_measure, position, c_matrices = best_model([gb], trainX, trainY, 'GB')
accuracies.append(accuracy)
f_measures.append(f_measure)
print('\nGB: Best Accuracy: {:.1f}%'.format(accuracy*100), 'Best F Measure: {:.1f}%'.format(f_measure*100))
print('Confusion Matrix')
for matrix in c_matrices:
  print(matrix)

kn = KNeighborsClassifier(n_jobs= -1, n_neighbors= 2, p= 1,weights= None)
# Getting the accuracy, F Measure and confusion matrix and displaying it
accuracy, f_measure, position, c_matrices = best_model([kn], trainX, trainY, 'kn')
accuracies.append(accuracy)
f_measures.append(f_measure)
print('\nKNN : Best Accuracy: {:.1f}%'.format(accuracy*100), 'Best F Measure: {:.1f}%'.format(f_measure*100))
print('Confusion Matrix')
for matrix in c_matrices:
  print(matrix)

# Decison Tree Model
dt = DecisionTreeClassifier(random_state=42)
# Getting the accuracy, F Measure and confusion matrix and displaying it
accuracy, f_measure, position, c_matrices = best_model([dt], trainX, trainY,'dt')
accuracies.append(accuracy)
f_measures.append(f_measure)
print('\nDT : Best Accuracy: {:.1f}%'.format(accuracy*100), 'Best F Measure: {:.1f}%'.format(f_measure*100))
print('Confusion Matrix')
for matrix in c_matrices:
  print(matrix)

# Navie Bayes Model
gnb = GaussianNB(var_smoothing=0.0003)
# Getting the accuracy, F Measure and confusion matrix and displaying it
accuracy, f_measure, position, c_matrices = best_model([gnb], trainX, trainY,'gnb')
accuracies.append(accuracy)
f_measures.append(f_measure)
print('\nNB: Best Accuracy: {:.1f}%'.format(accuracy*100), 'Best F Measure: {:.1f}%'.format(f_measure*100))
print('Confusion Matrix')
for matrix in c_matrices:
  print(matrix)


# Plotting the accuracies and F Measures
get_plot(accuracies, 'Accuracy on Cross Validation', 'Model = x+1', 'Accuracy', 'crossValidAccuracy.png')
get_plot(f_measures, 'F Measure on Cross Validation', 'Model = x+1', 'F Measure', 'crossValidF1.png')


# In[61]:


# Predict output
predict_1 = rf.predict(testX)
predict_2 = gb.predict(testX)
# Preparing the output data
output = pd.DataFrame(columns=['ID', 'Predict1', 'Predict2'])
output.ID = ID[1000:]
output.Predict1 = predict_1
output.Predict2 = predict_2
# Saving in csv utf-8
output.to_csv('predict.csv', encoding='utf-8', index=False)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




