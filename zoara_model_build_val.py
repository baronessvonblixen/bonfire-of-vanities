#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 16:49:31 2020

@author: meghan
"""

start with importing data set as a dataframe in pandas

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

df_clean_w_indicators = df_stalking_complete.where(lambda x:x==1, other=0)
#df_clean

#sum(df_clean['S0156'] )

#remove indicators from predictors
df_clean = df_clean_w_indicators.drop(['S0166', 'S0167', 'S0177','S0156', 'S0157', 'S0158', 'S0159', 'S0160', 'S0161', 'S0162', 'S0163', 'S0164', 'S0165', 'S0153','S0154', 'S0155', 'S0176', 'S0175', 'S0178','S0179', 'S0180', 'S0181', 'S0182', 'S0183', 'S0184', 'S0185', 'S0186', 'S0187'], axis=1)
print(df_clean)

#append the value in escalation_df to the end of the complete data set
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

score = clf_train.score(X_test, y_test)
print(score)
cfm = metrics.confusion_matrix(y_test, y_pred)
cfm

A = pos_incd_only_df.drop(['ESCAL'], axis=1)
b = pos_incd_only_df['ESCAL']
A_challenge = no_incd_only_df.drop(['ESCAL'], axis=1)
b_challenge = no_incd_only_df['ESCAL']

clf_train = LogisticRegression(random_state=0).fit(A,b)
b_pred = clf_train.predict(A_challenge)

score = clf_train.score(A_challenge, b_challenge)
print(score)

b_pred_proba = clf_train.predict_proba(A_challenge)[::,1]
fpr, tpr, _ = metrics.roc_curve(b_challenge,  b_pred_proba)
auc = metrics.roc_auc_score(b_challenge, b_pred_proba)
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)

cfm = metrics.confusion_matrix(b_challenge, b_pred)
print(cfm)

coefficients = clf_train.coef_
print(coefficients)

print(coefficients.type)

features_id = list(zip(coefficients[0], A.columns))

from sklearn.feature_selection import RFE

rfe = RFE(clf_train, 25)
fit = rfe.fit(A, b)
print("Num Features: %d" % fit.n_features_)
print("Selected Features: %s" % fit.support_)
#print("Feature Ranking: %s" % fit.ranking_)

selected_features_boolean_df = pd.DataFrame(fit.support_)
features_id_df = pd.DataFrame(features_id)

features_ranking = pd.concat([features_id_df, selected_features_boolean_df], axis=1)
features_ranking.columns= ['coef', 'code', 'bool']
features_ranking_sort = features_ranking.sort_values(by= ['bool','coef'], ascending= [0,1])
print(features_ranking_sort.head(26))

#make data sets for 10,15, 20 questionaire queries
#run model and compare on these sets


key_features_twenty = ['S0097', 'S0196', 'S0266', 'S0237', 'S0250', 'S0284', 'S0006', 'S0126', 'S0190', 'S0206', 'S0195', 'S0088', 'S0340', 'V2041', 'S0333', 'S0300', 'S0026', 'V2091', 'S0018', 'S0079']
key_features_fifteen = ['S0097', 'S0196', 'S0266', 'S0237', 'S0250', 'S0284', 'S0006', 'S0190', 'S0206', 'S0195', 'S0194', 'S0088', 'S0340', 'V2041', 'S0333']
key_features_ten = ['S0097', 'S0196', 'S0266', 'S0237', 'S0250', 'S0284', 'S0006', 'S0190', 'S0206', 'S0195']

df_twenty_queries_data_incident = pos_incd_only_df[key_features_twenty]
df_twenty_queries_data_noincident = no_incd_only_df[key_features_twenty]
df_fifteen_queries_data_incident = pos_incd_only_df[key_features_fifteen]
df_fifteen_queries_data_noincident = no_incd_only_df[key_features_fifteen]
df_ten_queries_data_incident = pos_incd_only_df[key_features_ten]
df_ten_queries_data_noincident = no_incd_only_df[key_features_ten]

X = df_twenty_queries_data_incident
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

score = clf_train.score(X_test, y_test)
print(score)
cfm = metrics.confusion_matrix(y_test, y_pred)
print(cfm)

print(clf_train.intercept_)
print(clf_train.coef_)

A = df_twenty_queries_data_incident
b = pos_incd_only_df['ESCAL']
A_challenge = df_twenty_queries_data_noincident
b_challenge = no_incd_only_df['ESCAL']

clf_train = LogisticRegression(random_state=0).fit(A,b)
b_pred = clf_train.predict(A_challenge)

score = clf_train.score(A_challenge, b_challenge)
print(score)

b_pred_proba = clf_train.predict_proba(A_challenge)[::,1]
fpr, tpr, _ = metrics.roc_curve(b_challenge,  b_pred_proba)
auc = metrics.roc_auc_score(b_challenge, b_pred_proba)
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)

cfm = metrics.confusion_matrix(b_challenge, b_pred)
print(cfm)

print(clf_train.intercept_)

print(clf_train.coef_)

X = df_fifteen_queries_data_incident
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

score = clf_train.score(X_test, y_test)
print(score)
cfm = metrics.confusion_matrix(y_test, y_pred)
cfm

X = df_ten_queries_data_incident
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

score = clf_train.score(X_test, y_test)
print(score)
cfm = metrics.confusion_matrix(y_test, y_pred)
cfm

    