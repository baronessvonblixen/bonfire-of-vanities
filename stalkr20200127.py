#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 22:52:16 2020

@author: meghan
"""

import streamlit as st

st.title('#MeToo')
st.subheader('In the past year, 1/3 of Americans experienced harassment in some form.')
st.subheader('In the past year, 5.3 million Americans over the age of 18 were stalked or criminally harassed.')
st.subheader('Of the criminal harassment and stalking cases, 10% escalated to physical violence or property damage.')
st.title('The Bottom Line: You are not alone.')

st.header('Take this short questionaire to learn more about your situation and get advice for how to best protect yourself and your loved ones.')
#begin questionare questions

#gender victim
gender = st.radio( "1. Do you identify as...",('male', 'female', 'nonbinary'))
if gender == 'nonbinary':
    st.write('Sadly, there is not a lot of data on the stalking and harassment of nonbinary people. But its important to know your rights.  Find some helpful infomation here: https://transequality.org/know-your-rights/survivors-violence')
else:
    st.write('Contrary to popular belief, Men are equally as likely to experience harassment as women. Women are more likely to be stalked, however, and are more likely to have stalking escalate.')

#gender of perp
perpgender = st.radio("2. The person bothering me is...", ('male', 'female', 'nonbinary'))
if perpgender == 'nonbinary':
    st.write('Not enough data exists to accurately assess the risk of being stalked by a nonbinary person, but some studies suggest that it is very low.')
else:
    st.write('Men are more likely to criminally harass and stalk than women, both if victim is male and if the victim is female.  Behavior of male harassers and stalkers is also more likely to escalate.')

#relation to victim
relation = st.radio("3. The person bothering me is a...", ('stranger', 'acquaintance', 'intimate (friend, family, etc.'))
if relation == 'stranger':
    st.write('Only 30% of the time is the stalker or harasser a stranger.  However, the behavior is much more likely to escalate to violence if the perpetrator is a stranger.')
else: 
    st.write('70% of the time the victim is either an acquaintance or intimate of the victim.  This can make evaluating the behavior difficult for the victim, who is often being manipulated or gaslighted by their harasser/stalker.')

#prior intimate relationship 
intimate = st.radio("4. Are you...?", ('married', 'divorced, separated, recently broken up with an intimate partner', 'widowed', 'single'))

if intimate == 'divorced, separated, recently broken up with an intimate partner':
    st.write('Separated, divorced, and individuals who have recently had a breakup with a long term partner are 25% more likely to experience stalking or criminal harassment than single individuals and 3.5 times more likely than widowed or married individuals.')
else: 
    if intimate == 'married':
        st.write('Married and widowed individuals are 3.5 times less likely to be criminally harassed than single or separated/divorced individuals and 2.5 times less likely than single individuals. But that does not mean it does not happen.')
    else: 
        if intimate == 'single':
            st.write('Single individuals are 2.5 times more likely to be stalked or criminally harassed than married or widowed individuals, but have 25% less risk than individuals who are divorced, seperated, or have recently broken up with a long-term intimate partner.')
        else:
           st.write('Married and widowed individuals are 3.5 times less likely to be criminally harassed than single or separated/divorced individuals and 2.5 times less likely than single individuals. But that does not mean it does not happen.')
 
 #living conditions
home = st.radio("5. I live in a...", ('house or apartment by myself', 'house or apartment with family/roommates', 'university dormitory', 'shelter or other temporary housing'))
if home == 'house or apartment by myself':
    st.write('Individuals who live by themselves are at a higher risk than those who do not.  You can protect youself by telling your neighbors of the situation, having friends/coworkers escort you home, and having someone be a point of contact for ensuring you have arrived home safely.')
else: 
    if home == 'house or apartment with family/roommates':
        st.write('You are at low risk for escalation, but use your family and roommates at contact points for your safety. If you are worried about your family members or roommates becoming the target of the stalking/harassment, develop a system of accountability that includes them.')
    else:
        if home == 'university dormitory': 
            st.write('University students are at the highest risk for stalking and criminal harassment, and for these behaviors escalating. Dormitories are notorious for having high foot traffic, and therefore leave the victim particularly vulnerable. Protect yourself by speaking to campus police, your Resident Asssistant for your floor, and your immediate dormmates.')
        else: 
            st.write('Victims in domestic violence shelters are relatively safe, and victims in homeless shelters are particularly vulnerable. Alert your social worker or organizational point of contact. If the stalker/harasser knows where you are staying, pursue transferring to a new, undisclosed shelter.')

st.subheader('Below is a "map" that visualizes victims of stalking and criminal harassment and their relation to one another. Situations that escalated to physical violence or property damage are indicated in red, and non-escalating cases are in green.  Based on how you answered the questionaire, the X indicates where you are on the map.')
image = st.image('popup_3.png')

st.subheader('Less than 1% of victims who are being stalked or criminally harassed do not feel accompanying anxiety.  Trust your instincts! Below is scale bar to indicate your current level of anxiety about the situation. This relaxes or tightens the criteria for predicting escalation.')    
option = st.slider('On a scale of 1 to 10, where 1 is no anxiety and 10 is hoplessness/suicidal, indicate your leve of anxiety.', 1, 10)

     
