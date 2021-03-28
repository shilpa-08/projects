#!/usr/bin/env python
# coding: utf-8

# ### About Dataset

# The HR_EMPLOYEE_ATTRITION dataset is the dataset giving various information about  employees and the attrition rate , the attrition of an employee
# is the loss for the company, as the companies invest time ,money and efforts  in training the employees , besides good employees are the 
# Assets of  the company, and more attrition in a company pretends that the company is not good that's why it is hard for an employee to work there,
# However there may be some another reasons of attriton, may be some personal reason of an employee.
# The Human Resources Department takes lots of efforts to reduce the attrition in the company.
# 
# Here in the dataset er have to prepare a  model through various attributes and  train it to determine whether the employee will continue the job or Not
# the answer of our model should be in the format of yes,No or 0,1.
# It means that it is Clasification type of Problem

# ### Importing necessary libraries

# In[123]:


import pandas as pd
import numpy as np


# In[124]:


#loading the dataset
df=pd.read_csv("HR_EMPLOYEE_ATTRITION.csv")


# In[125]:


df


# In[126]:


df.columns


# In[127]:


df.dtypes


# In[128]:


df.shape


# ### Checking Null Values

# In[129]:


df.isnull().sum()


# As all the columns have 0 values in front of them it means there are no null values in the dataset

# ### Visualizing Dataset

# In[130]:


import seaborn as sns
import matplotlib.pyplot as plt
import warnings 
warnings.filterwarnings('ignore')
sns.countplot(x=df['Attrition'],data=df)
print(df['Attrition'].value_counts())


# In[131]:


sns.countplot(x=df['BusinessTravel'],data=df)
print(df['BusinessTravel'].value_counts())


# there are 1043 employees who travels rarely, 277 employees travels frequently , and there are 150 employees who don't travel

# In[132]:


sns.countplot(x=df['Department'],data=df)
print(df['Department'].value_counts())


# Highest employees are engaged in Research & Development ,446 employees are engaged in sales department, and 63 employees are engaged in HR department

# In[133]:


sns.countplot(x=df['EducationField'],data=df)
print(df['EducationField'].value_counts())


# Highest number i.e., 606  employees are from Life Science field, 464 employees from Medical field, 159 employees from Marketing 
# ,132 from Technical degree, 82 from other and 27 from Human Resources

# In[134]:


sns.countplot(x=df['Gender'],data=df)
print(df['Gender'].value_counts())


# There are 882 Male Employees and 588 Female Employees

# In[135]:


sns.countplot(x=df['JobRole'],data=df)
print(df['JobRole'].value_counts())


# 326 employees are doing jobrole of Sales Executive, 
# 292  employees are doing jobrole  of Research Scientist ,
# 259 employees are Laboratory Technician,
# 145 are Manufacturing Director,
# 131 are Healthcare Representative, 
# 102 employees are Manager, 
# 83 of them are Sales Representative,
# 80 employees are  Research Director
# and 52 employees are doing Jobrole of Human Resources 

# In[136]:


sns.countplot(x=df['MaritalStatus'],data=df)
print(df['MaritalStatus'].value_counts())


# 673 Employees are Married , 470 employees are single and 327 are divorced employees

# In[137]:


sns.countplot(x=df['OverTime'],data=df)
print(df['OverTime'].value_counts())


# only 416 employees are doing Overtime and 1054 employees are not doing Overtime

# ##### Visualizing Continuous values

# In[139]:


df['Age'].plot.hist()
plt.xlabel('Age')


# more than 250 employees have their age between 30 to 40, minimum age is 20 and max age is 60 

# In[140]:


df['DailyRate'].plot.hist()
plt.xlabel('DailyRate')


# more than 160 employees have a daily rate of 600 , minimum daily rate is 200 and maximum is 1400

# In[141]:


df['DistanceFromHome'].plot.hist()
plt.xlabel('DistanceFromHome')


# almost 500 employees travel distance upto 5 kilometers, and almost 100 employees travels distance of 27 kms

# In[142]:


df['Education'].plot.hist()
plt.xlabel('Education')


# 600 employees have their education between point 3.0 and 50-70 employees have highest education of point 5.0

# In[143]:


df['EmployeeCount'].plot.hist()
plt.xlabel('EmployeeCount')


# there are 1470 employees 

# In[144]:


df['EmployeeNumber'].plot.hist()
plt.xlabel('EmployeeNumber')


# EmployeeNumber is the  unique number given to each employee it ranges from 0 to 2000

# In[145]:


df['EnvironmentSatisfaction'].plot.hist()
plt.xlabel('EnvironmentSatisfaction')


# more than 400 employees have Environment Satisfaction  of 3.0 and 4.0

# In[146]:


df['HourlyRate'].plot.hist()
plt.xlabel('HourlyRate')


# highest  hourly rate of employees is  100 and lowest is 60-70

# In[147]:


df['JobInvolvement'].plot.hist()
plt.xlabel('JobInvolvement')


# more than 800 employees have job involvement  of 3.0 

# In[148]:


df['JobLevel'].plot.hist()
plt.xlabel('JobLevel')


# more than 500 employees have Job level of points 1.0  ,   2.0

# In[149]:


df['JobSatisfaction'].plot.hist()
plt.xlabel('JobSatisfaction')


# more than 400 employees have job satifaction of 3.0 and 4.0

# In[150]:


df['MonthlyIncome'].plot.hist()
plt.xlabel('MonthlyIncome')


# more than 350 + employees have monthly income of 2500 to 5000 , around 50 employees get monthly income of 20000

# In[151]:


df['NumCompaniesWorked'].plot.hist()
plt.xlabel('NumCompaniesWorked')


# more than 500 employees have worked in 2 companies

# In[152]:


df['PerformanceRating'].plot.hist()
plt.xlabel('PerformanceRating')


# more than 1200 employees have Prformance Rating of 3.0 and 200 employees have of 4.0

# In[153]:


df['RelationshipSatisfaction'].plot.hist()
plt.xlabel('RelationshipSatisfaction')


# more than 400 employees have 3.0 and 4.0 relationship satisfaction

# In[154]:


df['StandardHours'].plot.hist()
plt.xlabel('Standard Hours')


#  all employees have same standard hours

# In[156]:


df['TotalWorkingYears'].plot.hist()
plt.xlabel('Total Working Years')


# more than 400 employees have working hour of 5-10 hours

# In[158]:


df['TrainingTimesLastYear'].plot.hist()
plt.xlabel('Traiining Time Last Year')


# more than 500 employees have been trained for 2 times last year 

# In[159]:


df['YearsAtCompany'].plot.hist()
plt.xlabel('Years at company')


# more than 400 workers have completed 5-7 years in a company

# In[160]:


df['YearsInCurrentRole'].plot.hist()
plt.xlabel('Years in Current Role')


# more than 500 employees have spend 0 to 2.5 years in current role

# In[162]:


df['YearsWithCurrManager'].plot.hist()
plt.xlabel('Years with current Manager')


# More than 500 workers have spent 0 to 2.5 years with the same manager

# ### Label Encoding 

# In[164]:



from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
for i in df.columns:
    if df[i].dtypes == object:
        df[i]=le.fit_transform(df[i])


# In[166]:


#checking the columns after the label encoding
df.dtypes


# ### statistical summary

# In[167]:


df.describe()


# Observations  :-
# 1)     DailyRate, Distancefromhome, TotalWorkingYears, YearAtCompany, YearsInCurrentRole , YearsSinceLastPromotion , YearsWithCurrManager have a 
# remarkable difference between their 75% and max showing the possibility of Outliers
# 
# 2) In DailyRate , EmployeeNumber the standard deviation is high , it means that this data are widely spread
# 
# 

# ### Checking Correlation

# In[168]:


dfcor=df.corr()


# In[169]:


plt.figure(figsize=(40,20))
sns.heatmap(dfcor,cmap='YlOrRd_r',annot=True)


# here it becomes unable to read the correlation data so we will statistically calculate it

# In[170]:


corr_matrix=df.corr()
corr_matrix['Attrition'].sort_values(ascending=False)


# now er can clearly identify the correlation of independent variable with target variable "class" there are around 20 variables 
# who has less then 0.01 correlation value (very weak relationship)
# 

# ### Checking Outliers

# In[171]:


for i in df.columns:
    plt.figure()
    sns.boxplot(df[i])


# Observation :-
# 
# MonthlyIncome, NumofCompaniesWorked, stockoptionlevel, totalworkingYears ,  TrainingTimesLastYear, YearsAtCompany, YearsInCurrentRole,
# YearsSinceLastPromotion, YearsWithCurrManager  columns are showing Outliers

# In[172]:


### Removing Outliers
from scipy.stats import zscore
z=np.abs(zscore(df))


# In[173]:


df_new=df[(z<3).all(axis=1)]


# In[174]:


df_new.shape , df.shape


# as per  the zscore all columns contains Outliers, so lets check it with Iqr

# In[175]:


#checking outliers using IQR
q1=df.quantile(0.25)
q3=df.quantile(0.75)
IQR=q3-q1
print(IQR)


# In[176]:


df_iqr_new=(df < (q1 - 1.5 * IQR)) | (df > (q3 + 1.5 * IQR)) 


# In[177]:


df_iqr_neww=df_iqr_new[~((df < (q1 - 1.5 * IQR)) | (df > (q3 + 1.5 * IQR))).any(axis=1)]


# In[178]:


df_iqr_neww.shape , df.shape


# In[179]:


1470-641


# here through IQR 829 columns are being removed as Outliers, such a big loss of data is not affordable so dropping the idea of removing outliers

# ### CheckingSkewness

# In[180]:


df.skew()


# as many columns have skew score more than 0.5 , so it is necessary to cure the skewness of dataset

# In[181]:


for i in df.columns:
    plt.figure()
    sns.distplot(df[i])


# Visually we can see  DistanceFromHome , MonthlyIncome ,  PercentSalaryHike ,  TotalWorkingYears, YearsAtCompany , YearsSinceLastPromotion , are rightly skewed ,and it should be treated

# ### separating the Data

# In[182]:


x=df.drop('Attrition',axis=1)


# In[183]:


x


# In[184]:


# applying power transform to remove the skewness of features columns
from sklearn.preprocessing  import power_transform

df_new=power_transform(x)

df_new=pd.DataFrame(df_new,columns=x.columns)


# In[185]:


x=df_new


# In[186]:


y=df['Attrition']
y


# In[188]:


#importing Necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score , confusion_matrix, classification_report


# ### Finding the Best Random State

# In[189]:



maxAccuracy=0
maxRandomState=0
for i in range(1,1000):
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=.30,random_state=i)
    lg=LogisticRegression()
    lg.fit(x_train,y_train)
    pred=lg.predict(x_test)
    accuracy=accuracy_score(y_test,pred)
    if accuracy>maxAccuracy:
        maxAccuracy=accuracy
        maxRandomState=i
print("Best Accuracy is ",maxAccuracy," on Random State",maxRandomState)


# ### Finding the best Model

# In[190]:


# Logistic Regression

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=.30,random_state=123)
lg=LogisticRegression()
lg.fit(x_train,y_train)
pred=lg.predict(x_test)
print(accuracy_score(y_test,pred))
print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))


# ###### LogisticRegression Model is giving accuracy score of 91 %

# In[191]:


from sklearn.tree import DecisionTreeClassifier

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=.30,random_state=123)
dtc=DecisionTreeClassifier()
dtc.fit(x_train,y_train)
dtc.fit(x_train,y_train)
pred=dtc.predict(x_test)
print(accuracy_score(y_test,pred))
print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))


# ###### DecisionTreeClassifier Model is giving accuracy of 76 %

# In[192]:


from sklearn.ensemble import RandomForestClassifier

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=.30,random_state=123)
rfc=RandomForestClassifier()
rfc.fit(x_train,y_train)
rfc.fit(x_train,y_train)
pred=rfc.predict(x_test)
print(accuracy_score(y_test,pred))
print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))


# ###### RandomForestClassifier model is giving accuracy of 88% 

# In[193]:


from sklearn.svm import SVC

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=.30,random_state=123)
svc=SVC()
svc.fit(x_train,y_train)
svc.fit(x_train,y_train)
pred=svc.predict(x_test)
print(accuracy_score(y_test,pred))
print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))


# ###### SupportVectorClass  Model  is giving accuracy of 89%

# In[194]:


from sklearn.naive_bayes import GaussianNB

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=.30,random_state=123)
gnb=GaussianNB()
gnb.fit(x_train,y_train)
gnb.fit(x_train,y_train)
predgnb=gnb.predict(x_test)
print(accuracy_score(y_test,predgnb))
print(confusion_matrix(y_test,predgnb))
print(classification_report(y_test,predgnb))


# ###### GaussianNB  model is giving accuracy of 83 %

# ### Cross Validation of Models

# finding the best CV

# In[195]:


from sklearn.model_selection import cross_val_score

for j in range(2,10):
    cvs = cross_val_score(lg,x,y,cv=j)
    cvsmean=cvs.mean()
    print(f" Cross_val_score at  {j } is {cvsmean}")


# at CV=8 the model is giving highest Cv score

# ### Checking Cross_Val_Score of all models with cv=8

# In[196]:


from sklearn.model_selection import cross_val_score

modellist=[lg,dtc,rfc,svc,gnb]
for i in modellist:
    cvs = cross_val_score(i,x,y,cv=8)
    cvsmean=cvs.mean()
    print(f" Cross_val_score of  {i} is {cvsmean}")


# In[224]:


# now finding the best with highest accuracy and cross _val_score

#         MODEL               ACCURACY(in %)            CROSS_VAL_SCORE (in %)           DIFFERENCE

# LogisticRegression             91                           87                           4 %
# DecisionTreeClassifier         76                           77                           1 %
# RandomForestClassifier         89                           85                           4 %
# SupportVectorClass             89                           86                           3 %
#   GaussianNB                   83                           82                           1 %


# as per the above observations , the difference between the accuracy and cross val score of DecisionTreeClassifier is 1% ,
# It means the model has learnt well 

# ### Hyper Parameter Tuning

# In[225]:


from sklearn.model_selection import GridSearchCV
 # creating parameter list to pass in GridSearchCV
parameters ={'max_depth': np.arange(2,15),
            'criterion':['gini','entropy']}


# In[226]:


CGV=GridSearchCV(DecisionTreeClassifier(),parameters,cv=8)


# In[227]:


CGV.fit(x_train,y_train)


# In[228]:


CGV.best_params_            #printing the best parameters found by GridSearchCV


# In[229]:


CGV_pred=CGV.best_estimator_.predict(x_test)


# In[230]:


accuracy_score(y_test,CGV_pred)


# ###### After Hyper Parameter Tuning of DecisionTreeModel we have got the Accuracy of    87 %

# ### Loading The model in .pkl 

# In[245]:


import pickle
filename='HR_EMPLOYEE_ATTRITION.pkl'
pickle.dump(CGV.best_estimator_,open(filename,'wb'))


# ### Conclusion

# ###### here we have successfully load the model in 'HR_EMPLOYEE_ATTRITION.pkl'

# ###### Checking the Working of the Model

# In[246]:


import pickle
loaded_model=pickle.load(open('HR_EMPLOYEE_ATTRITION.pkl','rb'))
result=loaded_model.score(x_test,y_test)

print(result)


# In[249]:


conclusion=pd.DataFrame([loaded_model.predict(x_test)[:],CGV_pred[:]],index=["Predicted","Original"])


# In[250]:


conclusion


# In[ ]:




