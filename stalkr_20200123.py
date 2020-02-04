#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 23:30:07 2020

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

clean_complete_df = pd.DataFrame(np.zeros((78741, 617)))
                             
for n_ind in range(1,78741):
    for m_ind in range(1,617):
        if df_stalking_complete.loc[n_ind][m_ind]==1:
            clean_complete_df.loc[n_ind][m_ind]=clean_complete_df.loc[n_ind][m_ind]+1
        else:
            clean_complete_df.loc[n_ind][m_ind]=clean_complete_df.loc[n_ind][m_ind]+0
print(clean_complete_df)

#append the value in escalation_df to the end of the complete data set
#first make the list a dataframe

escalation_df = pd.DataFrame(escalation_list)

complete_w_escalation_df = pd.concat([clean_complete_df, escalation_df],axis=1)
complete_w_escalation_df.rename(columns={0:'ESCAL'}, inplace=True)
complete_w_escalation_df
    