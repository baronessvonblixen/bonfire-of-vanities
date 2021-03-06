#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 09:48:32 2020

@author: meghan
"""

#start with importing data set as a dataframe in pandas

import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
import matplotlib

df_stalking_complete = pd.read_csv("/Users/meghan/Desktop/2006STALKINGSUPPLEMENT/DS0005/20080-0005-Data.tsv",sep='\t')
print(df_stalking_complete)

#count the number of occurances for attacks or attack attempts
# 1=yes, 2=no, 8=residue, 9=out of universe

attack_cols = ['S0156', 'S0157', 'S0158', 'S0159', 'S0160', 'S0161', 'S0162', 'S0163', 'S0164', 'S0165']
attack_df = df_stalking_complete[attack_cols]
attack_df.apply(pd.Series.value_counts)

#count the number of occurances for injuries sustained
# 1=yes, 0=no, 9=out of universe

inj_cols = ['S0178','S0179', 'S0180', 'S0181', 'S0182', 'S0183', 'S0184', 'S0185', 'S0186', 'S0187']
inj_df = df_stalking_complete[inj_cols]
inj_df.apply(pd.Series.value_counts)

#count the number of occurances for property damage
# 1=yes, 0=no, 9=out of universe

prop_cols = ['S0153','S0154', 'S0155']
prop_df = df_stalking_complete[prop_cols]
prop_df.apply(pd.Series.value_counts)

#create a column that indicates escalation or not

merge_attack_prop_df = pd.concat([attack_df, prop_df],axis=1)
merge_attack_prop_df
    
#create a column that indicates escalation or not
#sum indicates number of unique escalation cases

escalation_list = []

for row in merge_attack_prop_df.iterrows(): 
    if 1 in row[1].values:
        escalation_list.append(1)
    else: 
        escalation_list.append(0)
sum(escalation_list)
    
attack_list=[]
prop_list=[]

for row in attack_df.iterrows(): 
    if 1 in row[1].values:
        attack_list.append(1)
    else: 
        attack_list.append(0)

for row in prop_df.iterrows(): 
    if 1 in row[1].values:
        prop_list.append(1)
    else: 
        prop_list.append(0)   
        
print(sum(attack_list))
print(sum(prop_list))

#clean data frame so that there are binary "1"s for affirmative responses and nothing else

df_clean = df_stalking_complete.where(lambda x:x==1, other=0)
#df_clean

#sum(df_clean['S0156'] )

append the value in escalation_df to the end of the complete data set
#first make the list a dataframe

escalation_df = pd.DataFrame(escalation_list)

complete_w_escalation_df = pd.concat([df_clean, escalation_df],axis=1)
complete_w_escalation_df.rename(columns={0:'ESCAL'}, inplace=True)
complete_w_escalation_df

id_as_stalk = sum(complete_w_escalation_df['S0352'])
print(id_as_stalk)

complete_sort_by_incd = complete_w_escalation_df.sort_values(by=['S0352'], ascending=False)

pos_incd_only_df = complete_sort_by_incd[1:729]
no_incd_only_df = complete_sort_by_incd[730:78741]
print(sum(pos_incd_only_df['ESCAL']))
print(sum(no_incd_only_df['ESCAL']))

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

X = pos_incd_only_df.drop(['ESCAL'], axis=1)
y = pos_incd_only_df['ESCAL']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
clf_train = LogisticRegression(random_state=0).fit(X_train,y_train)

y_pred = clf_train.predict(X_test)

y_pred_proba = clf_train.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)

cfm = metrics.confusion_matrix(y_test, y_pred)
cfm

A = pos_incd_only_df.drop(['ESCAL'], axis=1)
b = pos_incd_only_df['ESCAL']
A_challenge = no_incd_only_df.drop(['ESCAL'], axis=1)
b_challenge = no_incd_only_df['ESCAL']

clf_train = LogisticRegression(random_state=0).fit(A,b)
b_pred = clf_train.predict(A_challenge)

b_pred_proba = clf_train.predict_proba(A_challenge)[::,1]
fpr, tpr, _ = metrics.roc_curve(b_challenge,  b_pred_proba)
auc = metrics.roc_auc_score(b_challenge, b_pred_proba)
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)

cfm = metrics.confusion_matrix(b_challenge, b_pred)
cfm

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X = complete_w_escalation_df.drop(['ESCAL'], axis=1)
y = complete_w_escalation_df['ESCAL']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
clf_train = LogisticRegression(random_state=0).fit(X_train,y_train)

y_pred = clf_train.predict(X_test)

y_pred_proba = clf.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)

cfm = metrics.confusion_matrix(y_test, y_pred)
cfm

#false negative rate very low!
#false positive rate high :-(

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler 
from sklearn.neighbors import KNeighborsClassifier
import plotly.express as px

# Training KNN Classifier
import 
@st.cache(suppress_st_warning=True)

def Knn_Classifier(X_train, X_test, y_train, y_test):
	clf = KNeighborsClassifier(n_neighbors=5)
	clf.fit(X_train, y_train)
	y_pred = clf.predict(X_test)
	score = metrics.accuracy_score(y_test, y_pred) * 100
	report = classification_report(y_test, y_pred)

	return score, report, clf