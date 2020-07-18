# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 15:50:37 2020

@author: SAMRAH SOHA
"""

import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

m_affair=pd.read_csv('E:/Assisgnments/Logistic Regression/Affairs.csv')
m_affair.gender=m_affair.gender.map({'male':1,'female':0})
m_affair.children=m_affair.children.map({'yes':1,'no':0})

#converting affair variable into binary
m_affair.loc[m_affair.affairs>0,'affairs']=1
 #eda 
m_affair.describe()  #  box plot values mean, median, sd, range of all variables
sb.pairplot(m_affair)

#1model building 
Model1=sm.logit('affairs~age+yearsmarried+religiousness+education+occupation+rating+gender+children',data=m_affair).fit()
Model1.summary()    #probablities of education & occupation are insignificant 
Model1.summary2()   #AIC value=627.51 

#2 model building by removing education and occupation as they are insignificant
Model2=sm.logit('affairs~age+yearsmarried+religiousness+rating+gender+children',data=m_affair).fit()
Model2.summary()
Model2.summary2()     #AIC value=624.15

#3 model building by removing children as it is insignificant
Model3=sm.logit('affairs~age+yearsmarried+religiousness+rating+gender',data=m_affair).fit()
Model3.summary()
Model3.summary2() #AIC value=623.85 

#as AIC value for model3 is low we are going to consider model 3

print(np.exp(Model3.params)) # as logistic regression is function of log(odds)
predict = Model3.predict(pd.DataFrame(m_affair[['age','yearsmarried','religiousness','rating','gender']]))

# Creating new column for storing predicted class of affairs
m_affair["predict"]=predict

from sklearn.metrics import confusion_matrix
C_Table=confusion_matrix(m_affair['affairs'],predict>0.5)
C_Table
accuracy = (437+26)/(437+14+124+26) 
accuracy  # 77 %

#roc curve
from sklearn import metrics
fpr,tpr,threshold=metrics.roc_curve(m_affair.affairs,predict)

plt.plot(fpr,tpr);plt.xlabel("False Positive");plt.ylabel("True Positive")
roc_auc=metrics.auc(fpr,tpr) # area under curve-0.706 which covers most higher the roc curve value better the model
roc_auc
