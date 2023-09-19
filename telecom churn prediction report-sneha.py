

#Sneha Balyan

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("WA_Fn_UseC_Telco-Customer-Churn.csv")
df.head()
df.shape
df.dtypes
df.info()
df.isnull().sum()
df.duplicated().sum()
df.duplicated().value_counts()
df.columns.values

# Removing the missing values
df.TotalCharges = pd.to_numeric(df.TotalCharges, errors='coerce')
df.isnull().sum()
df.dropna(inplace = True)

'''
import warnings
warnings.filterwarnings('ignore')
sns.countplot(df["Churn"])
'''

df1 = df.iloc[:,1:]
df1.info()

for i in df1:
    plt.figure(i)
    sns.countplot(data=df1, x = i, hue='Churn')

# Label Encoding the Target Variable
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
tar_var = ['Churn']
df1[tar_var] = df1[tar_var].apply(LabelEncoder().fit_transform)
df1.info()

# Converting all the categorical variables into dummy variables
df_dummies = pd.get_dummies(df1)
df_dummies.head()
df_dummies.info()

#Get Correlation of "Churn" with other variables:
plt.figure(figsize = (13,8))
df_dummies.corr()['Churn'].sort_values(ascending = False).plot(kind='bar')

plt.figure(figsize = (27,27))
matrix = df_dummies.corr()
sns.heatmap(matrix,cmap='coolwarm',fmt = '.0%',annot = True)
matrix

fig = plt.figure(figsize = (12,9))
sns.boxplot(data = df_dummies, linewidth = 1)
plt.xticks(rotation = ('vertical'))
plt.show()

df_dummies.TotalCharges.describe()

def outliers (df_dummies,ft):
    Q1 = df_dummies[ft].quantile(0.25)
    Q3 = df_dummies[ft].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1-1.5*IQR
    upper_bound = Q3+1.5*IQR
    
    ls = df_dummies.index[(df_dummies[ft] < lower_bound) | (df_dummies[ft] > upper_bound)]
    return ls

def remove (df_dummies,ls):
    ls = sorted(set(ls))
    df_dummies = df_dummies.drop(ls)    
    return df_dummies

df_dummies.columns.values

index_list=[]
for feature in ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges',
       'Churn', 'gender_Female', 'gender_Male', 'Partner_No',
       'Partner_Yes', 'Dependents_No', 'Dependents_Yes',
       'PhoneService_No', 'PhoneService_Yes', 'MultipleLines_No',
       'MultipleLines_No phone service', 'MultipleLines_Yes',
       'InternetService_DSL', 'InternetService_Fiber optic',
       'InternetService_No', 'OnlineSecurity_No',
       'OnlineSecurity_No internet service', 'OnlineSecurity_Yes',
       'OnlineBackup_No', 'OnlineBackup_No internet service',
       'OnlineBackup_Yes', 'DeviceProtection_No',
       'DeviceProtection_No internet service', 'DeviceProtection_Yes',
       'TechSupport_No', 'TechSupport_No internet service',
       'TechSupport_Yes', 'StreamingTV_No',
       'StreamingTV_No internet service', 'StreamingTV_Yes',
       'StreamingMovies_No', 'StreamingMovies_No internet service',
       'StreamingMovies_Yes', 'Contract_Month-to-month',
       'Contract_One year', 'Contract_Two year', 'PaperlessBilling_No',
       'PaperlessBilling_Yes', 'PaymentMethod_Bank transfer (automatic)',
       'PaymentMethod_Credit card (automatic)',
       'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check']:
    index_list.extend(outliers(df_dummies,feature))
   
df_new = remove(df_dummies,index_list)

df_new.shape

fig = plt.figure(figsize = (12,9))
sns.boxplot(data = df_new, linewidth = 1)
plt.xticks(rotation = ('vertical'))
plt.show()

df_new.info()

# feature selection
x = df_new.drop('Churn', axis = 1)
y = df_new['Churn'].values

# Scaling all the variables to a range of 0 to 1
from sklearn.preprocessing import MinMaxScaler
features = x.columns.values
scaler = MinMaxScaler(feature_range = (0,1))
scaler.fit(x)
x = pd.DataFrame(scaler.transform(x))
x.columns = features

#====================================================================

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

select_kbest_rank_feature = SelectKBest(score_func= chi2, k = 5)
kbest_feature = select_kbest_rank_feature.fit(x, y)

df_score = pd.DataFrame(kbest_feature.scores_,columns = ['Score'])
dfcolumns = pd.DataFrame(x.columns)

kbest_rank_feature_concat = pd.concat([dfcolumns,df_score], axis = 1)
kbest_rank_feature_concat.columns = ['features','k_score']
kbest_rank_feature_concat

print(kbest_rank_feature_concat.nlargest(45,'k_score'))


#drop columns through creating list
K_Best_drop_features=['SeniorCitizen','PhoneService_No','PhoneService_Yes',
                      'MultipleLines_No phone service','InternetService_No',
                      'OnlineSecurity_No internet service','OnlineBackup_No internet service',
                      'DeviceProtection_No internet service','TechSupport_No internet service',
                      'StreamingTV_No internet service','StreamingMovies_No internet service',
                      'Contract_Month-to-month','Contract_One year','Contract_Two year',
                      'PaymentMethod_Bank transfer (automatic)','PaymentMethod_Credit card (automatic)',
                      'PaymentMethod_Electronic check','PaymentMethod_Mailed check','MultipleLines_Yes',
                      'MultipleLines_No','gender_Female','gender_Male','StreamingMovies_No',
                      'DeviceProtection_No','StreamingMovies_Yes','Dependents_No']

df_new.drop(K_Best_drop_features,inplace=True,axis=1)
df_new.info()

plt.figure(figsize=(12,9))
matrix = df_new.corr()
sns.heatmap(df_new.corr(),annot = True,linewidth = 1,cmap = 'coolwarm',fmt = '.0%')
matrix

#===========================================================================
### MODEL DEVELOPMENT :-

# 1. logistic regression
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,train_size = 0.70,random_state=10)

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(x_train,y_train)

y_train_pred = logreg.predict(x_train)
y_test_pred = logreg.predict(x_test)

from sklearn.metrics import accuracy_score
print("Training Accuracy Score:",accuracy_score(y_train,y_train_pred).round(3))
print("Test Accuracy Score:",accuracy_score(y_test,y_test_pred).round(3))

#=====================================================================
# To get the weights of all the variables
weights = pd.Series(logreg.coef_[0],
                 index=x.columns.values)
print (weights.sort_values(ascending = False)[:20].plot(kind='bar'))

print(weights.sort_values(ascending = False)[-20:].plot(kind='bar'))

#========================================================================
# 2. confusion matrix
from sklearn.metrics import confusion_matrix,accuracy_score
cm1 = confusion_matrix(y_train,y_train_pred)
print("Accuracy Score:",accuracy_score(y_train,y_train_pred).round(2))
TN=cm1[0,0]
FP=cm1[0,1]
TNR=TN/(TN+FP)
print("Specificity Score:",TNR.round(2))
from sklearn.metrics import recall_score,f1_score,precision_score
print("Sensitivity score:",recall_score(y_train,y_train_pred).round(2))
print("Precision Score:",precision_score(y_train,y_train_pred).round(2))
print("F1 Score:",f1_score(y_train,y_train_pred).round(2))
cm1

from sklearn.metrics import confusion_matrix,accuracy_score
cm2 = confusion_matrix(y_test,y_test_pred)
print("Accuracy Score:",accuracy_score(y_test,y_test_pred).round(2))
TN=cm2[0,0]
FP=cm2[0,1]
TNR=TN/(TN+FP)
print("Specificity Score:",TNR.round(2))
from sklearn.metrics import recall_score,f1_score,precision_score
print("Sensitivity score:",recall_score(y_test,y_test_pred).round(2))
print("Precision Score:",precision_score(y_test,y_test_pred).round(2))
print("F1 Score:",f1_score(y_test,y_test_pred).round(2))
cm2

#=======================================================================
# 3. k fold method
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
kfold = KFold(n_splits=10)
logreg = LogisticRegression()
results = cross_val_score(logreg,x,y,cv = kfold,scoring='accuracy')
np.mean(results).round(3)
results

#=========================================================================
# 4. KNN
from sklearn.neighbors import KNeighborsClassifier
KNN = KNeighborsClassifier(n_neighbors=5,p=2)
KNN.fit(x_train,y_train)
KNeighborsClassifier()
y_train_pred = KNN.predict(x_train)
y_test_pred = KNN.predict(x_test)
train_acc = accuracy_score(y_train,y_train_pred)
test_acc = accuracy_score(y_test,y_test_pred)
print(train_acc)
print(test_acc)

#===========================================================================
# 5. Random Forest
from sklearn.ensemble import RandomForestClassifier
RFC = RandomForestClassifier(max_depth=5,
                             n_estimators=500,
                             max_samples=0.6,max_features=0.5,
                             bootstrap=True,random_state=100)

RFC.fit(x_train,y_train)

y_pred_train = RFC.predict(x_train)
y_pred_test = RFC.predict(x_test)

from sklearn.metrics import accuracy_score
ac1 = accuracy_score(y_train,y_pred_train)
print("Training Accuracy score" ,ac1.round(3))
ac2 = accuracy_score(y_test,y_pred_test)
print("Test Accuracy score" ,ac2.round(3))

#=====================================================================
# 6. Support Vecor Machine (SVM)

from sklearn.model_selection._split import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=10)

# model fitting
from sklearn.svm import SVC
svc_linear = SVC(kernel='linear')
svc_linear.fit(x_train, y_train)
y_pred_train = svc_linear.predict(x_train)
y_pred_test  = svc_linear.predict(x_test)

from sklearn import metrics
print("Training Accuracy :",metrics.accuracy_score(y_train, y_pred_train).round(3))
print("Test Accuracy:",metrics.accuracy_score(y_test, y_pred_test).round(3))

#========================================================================
# 7. Decision Tree

# fit the model with DT
from sklearn.tree import DecisionTreeClassifier

#dt = DecisionTreeClassifier(criterion="gini", max_depth=6)
dt = DecisionTreeClassifier(criterion="entropy", max_depth=6)

dt.fit(x_train,y_train)

y_pred_train = dt.predict(x_train)
y_pred_test = dt.predict(x_test)

from sklearn.metrics import accuracy_score
ac1 = accuracy_score(y_train,y_pred_train)
print("Training Accuracy score" ,ac1.round(3))
ac2 = accuracy_score(y_test,y_pred_test)
print("Test Accuracy score" ,ac2.round(3))

dt.tree_.max_depth # calculating the depth of the tree
dt.tree_.node_count # calculating the number of nodes

#==============================================================

# 8. Bagging method
from sklearn.ensemble import BaggingClassifier
dt = DecisionTreeClassifier(max_depth=5)
bag = BaggingClassifier(base_estimator=dt,
                        n_estimators=100,
                        max_samples=0.6,max_features=0.5,
                        bootstrap=False)

bag.fit(x_train,y_train)

y_pred_train = bag.predict(x_train)
y_pred_test = bag.predict(x_test)

from sklearn.metrics import accuracy_score
ac1 = accuracy_score(y_train,y_pred_train)
print("Training Accuracy score" ,ac1.round(3))
ac2 = accuracy_score(y_test,y_pred_test)
print("Test Accuracy score" ,ac2.round(3))

#======================================================================

# 9. ADA Boost
# AdaBoost Algorithm

from sklearn.ensemble import AdaBoostClassifier
model = AdaBoostClassifier()

# n_estimators = 50 (default value) 
# base_estimator = DecisionTreeClassifier (default value)

model.fit(x_train,y_train)
preds = model.predict(x_test)
metrics.accuracy_score(y_test, preds)

#=====================================================================================
# 10. XG Boost

from xgboost import XGBClassifier
model = XGBClassifier()
model.fit(x_train, y_train)
preds = model.predict(x_test)
metrics.accuracy_score(y_test, preds)
