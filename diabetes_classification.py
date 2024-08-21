import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

data = pd.read_csv('diabetes.csv')

data.head()

data.info(verbose=True)

data.describe()

data_copy = data.copy(deep=True)

data_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = data_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.nan)

data_copy.isnull().sum()

data.hist(figsize=(20, 20))

data_copy['Glucose'].fillna(data_copy['Glucose'].mean(), inplace=True)
data_copy['BloodPressure'].fillna(data_copy['BloodPressure'].mean(), inplace=True)
data_copy['SkinThickness'].fillna(data_copy['SkinThickness'].median(), inplace=True)
data_copy['Insulin'].fillna(data_copy['Insulin'].median(), inplace=True)
data_copy['BMI'].fillna(data_copy['BMI'].median(), inplace=True)

data_copy.hist(figsize=(20, 20))

sns.countplot(x='Outcome', data=data, palette= 'rainbow', hue='Outcome')

sns.pairplot(data_copy, hue='Outcome')

sns.heatmap(data_copy.corr(), annot=True)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = pd.DataFrame(sc.fit_transform(data_copy.drop(['Outcome'], axis=1)),
                 columns=['Pregnancies', 'Glucose', 'BloodPressure',
                          'SkinThickness', 'Insulin', 'BMI',
                          'DiabetesPedigreeFunction', 'Age'])
y = data_copy['Outcome']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Display the new class distribution
print(pd.Series(y_train_smote).value_counts())

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

params_grid = {'n_neighbors': np.arange(1, 100)}
knn = KNeighborsClassifier()
knn_cv = GridSearchCV(knn, params_grid, cv=5)
knn_cv.fit(X, y)

print("Best Score:", str(knn_cv.best_score_))
print("Best Params:", str(knn_cv.best_params_))

knn = KNeighborsClassifier(n_neighbors=knn_cv.best_params_['n_neighbors'])
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

from sklearn.metrics import confusion_matrix
cnf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
plt.title('Confusion matrix')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

from sklearn.metrics import classification_report
print('Classification Report for KNN:\n\n', classification_report(y_test, y_pred))

from sklearn.linear_model import LogisticRegression
lg = LogisticRegression()
lg.fit(X_train, y_train)

y_pred = lg.predict(X_test)

scaler = StandardScaler().fit(data_copy.drop(['Outcome'], axis=1))
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(knn, 'knn_model.pkl')
joblib.dump(lg, 'lg_model.pkl')

cnf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
plt.title('Confusion matrix')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

print('Classification Report for Logistic Regression:\n\n', classification_report(y_test, y_pred))

