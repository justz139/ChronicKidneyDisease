Importing required libraries:

import pandas as pd
import numpy as np
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt

#Database loading and pre-processing
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import pandas as pd # Importing pandas with alias 'pd'

dataset=(r'/Chronic_Kidney_Dsease_data.csv')

Data preprocessing / EDA:

kidney_dataframe = pd.read_csv(dataset)

print('Shape of dataset: ' + str(kidney_dataframe.shape))
print('Total number of data = ' + str(kidney_dataframe.shape[0]))
print('Total number of attributes = ' + str(kidney_dataframe.shape[1]))


from google.colab import drive
drive.mount('/content/drive')

kidney_dataframe = kidney_dataframe.replace('Confidential', np.nan)

# Identifying columns
confidential_columns = [col for col in kidney_dataframe.columns
                       if kidney_dataframe[col].isnull().any()]

# one-hot encoding
for col in confidential_columns:
    # Create dummy variables
    dummies = pd.get_dummies(kidney_dataframe[col], prefix=col)

    # Dropping original column and add dummies
    kidney_dataframe = kidney_dataframe.drop(columns=[col])
    kidney_dataframe = pd.concat([kidney_dataframe, dummies], axis=1)

#setting the features and the target variables
target_class = kidney_dataframe['Diagnosis']
feature_classes = kidney_dataframe.drop(columns=['Diagnosis'])

# Impute missing values only on numerical columns
numerical_features = feature_classes.select_dtypes(include=np.number)
knn_missing_values_imputer = KNNImputer(n_neighbors=3)
imputed_features = pd.DataFrame(knn_missing_values_imputer.fit_transform(numerical_features),
                               columns=numerical_features.columns,
                               index=numerical_features.index)

# Update feature_classes with imputed values
feature_classes.update(imputed_features)

#Scaling
standard_scaler = StandardScaler()
scaled_features = standard_scaler.fit_transform(feature_classes.select_dtypes(include=np.number))
scaled_features = pd.DataFrame(scaled_features, columns=numerical_features.columns,
                               index=numerical_features.index)

# Update feature_classes with scaled values
feature_classes.update(scaled_features)

#Encoding target class
target_encoder = preprocessing.LabelEncoder()
target_class = target_encoder.fit_transform(target_class)
target_class1 = pd.DataFrame(target_class, columns=['Diagnosis']) #0 for kidney disease, 1 for no ckd

#target_class1

kidney_dataframe.head()
kidney_dataframe.describe()
kidney_dataframe.info()

Split the dataset:

train_features, test_features, train_target, test_target = train_test_split(feature_classes, target_class, train_size = 0.7, test_size=0.3, random_state=0)
print('\nAfter Pre-processing:')
print('Size of train dataset: ' + str(train_target.shape[0]))
print('Size of test dataset: ' + str(test_target.shape[0]))

#Target class Visualization
#0 - Kidney disease, 1 - no disease
sns.countplot(pd.concat([feature_classes, target_class1], axis=1, sort=False)['Diagnosis'], label = "Count")
plt.title('Bar chart for the number of observations relating to the target variable ')
plt.xlabel('Target_class')
plt.ylabel('Number of Observations')
plt.show()

Two Models:
#K-Nearest Neighbour
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import metrics

knn_model = KNeighborsClassifier()

knn_parameters_grid = {'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                       'weights': ['uniform', 'distance'],
                       'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                       'n_jobs': [1, -1]}

knn_grid_search = GridSearchCV(knn_model, knn_parameters_grid, scoring='accuracy')
# fit the data to the grid
knn_grid_search.fit(train_features, train_target)

print('Best parameters:\n ' + str(knn_grid_search.best_params_))

print('Best model after gridsearch:\n ' + str(knn_grid_search.best_estimator_))

knn_prediction = knn_grid_search.predict(test_features)

print('\nPrecision: ' + str(metrics.precision_score(test_target, knn_prediction)))
print('Accuracy: ' + str(metrics.accuracy_score(test_target, knn_prediction)))
print('Recall: ' + str(metrics.recall_score(test_target, knn_prediction)))
print('F1 score: ' + str(metrics.f1_score(test_target, knn_prediction)))

print('\nClassification Report:\n' + str(metrics.classification_report(test_target, knn_prediction)))

sns.heatmap(metrics.confusion_matrix(test_target, knn_prediction), annot=True)
plt.show()


#Decision Tree
from sklearn.tree import DecisionTreeClassifier

dt_model = DecisionTreeClassifier(random_state=0)

dt_parameters_grid = {'criterion': ['gini', 'entropy'],
                      'splitter': ['best', 'random'],
                      'min_samples_leaf': [1, 2, 3, 4, 5],
                      'max_features': ['auto', 'sqrt', 'log2']}

dt_grid_search = GridSearchCV(dt_model, dt_parameters_grid, scoring='accuracy')

dt_grid_search.fit(train_features, train_target)

print('Best parameters:\n ' + str(dt_grid_search.best_params_))
print('\n Best model after gridsearch is:\n ' + str(dt_grid_search.best_score_))

dt_prediction = dt_grid_search.predict(test_features)

print('\nPrecision: ' + str(metrics.precision_score(test_target, dt_prediction)))
print('Accuracy: ' + str(metrics.accuracy_score(test_target, dt_prediction)))
print('Recall: ' + str(metrics.recall_score(test_target, dt_prediction)))
print('F1 score: ' + str(metrics.f1_score(test_target, dt_prediction)))

print('\nClassification Report:\n' + str(metrics.classification_report(test_target, dt_prediction)))

print('\nConfusion Matrix:\n' + str(metrics.confusion_matrix(test_target, dt_prediction)))

sns.heatmap(metrics.confusion_matrix(test_target, dt_prediction), annot=True)
plt.show()

a=(train_features.loc[377]).values.reshape(1, -1)

a=np.asarray(a)
print(a)
a.reshape(1,-1)
print(type(a))
d = dt_grid_search.predict(a)
d[0]

b = (-0.94380604,  0.61132386,  0.96971691,  1.28675457, -1.25918547,  0.33703081,
  -1.13289044, -0.64367869, -0.84443929, -0.89942745,  1.05497466,  0.97849686,
  -0.40522898, -0.65681534,  1.69860116, -0.34340141, -0.51615135,  1.22700162,
   0.15485157, -0.59458621, -1.1871683,  -1.45049115,  1.0634567,   1.2698507,
   0.19593581,  1.4321779,  -0.90923736, -1.04010343,  1.3755787,   1.10615636,
   1.20274661, -1.64787841, -1.38201817,  0.08318189,  0.81736602,  1.54457576,
   1.46357284, -0.58022937,  1.27475488, -0.50395263, -0.50301208,  1.08637241,
   0.46080013, -0.23644032,  0.21032199,  1.36767293, -0.21454077, -0.33899757,
  -0.49547502,  1.42796391, -1.58609155, -0.33035966)
ab=(-0.94380604,  0.61132386,  0.96971691,  1.28675457, -1.25918547,  0.33703081,
  -1.13289044, -0.64367869, -0.84443929, -0.89942745,  1.05497466,  0.97849686,
  -0.40522898, -0.65681534,  1.69860116, -0.34340141, -0.51615135,  1.22700162,
   0.15485157, -0.59458621, -1.1871683,  -1.45049115,  1.0634567,   1.2698507,
   0.19593581,  1.4321779,  -0.90923736, -1.04010343,  1.3755787,   1.10615636,
   1.20274661, -1.64787841, -1.38201817,  0.08318189,  0.81736602,  1.54457576,
   1.46357284, -0.58022937,  1.27475488, -0.50395263, -0.50301208,  1.08637241,
   0.46080013, -0.23644032,  0.21032199,  1.36767293, -0.21454077, -0.33899757,
  -0.49547502,  1.42796391, -1.58609155, -0.33035966)

b = np.asarray(b)
#print(type(b))
b = b.reshape(1,-1)
d = dt_grid_search.predict(b)
d=[0]
if d[0]==0:
  print("Person has Chronic Kidney Disease")
else:
  print("Great! person is fine.")
