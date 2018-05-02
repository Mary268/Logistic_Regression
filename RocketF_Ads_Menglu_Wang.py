#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 17:05:41 2018

@author: Menglu Wang    ID 20707728
@Copy Right: Menglu(Mary) Wang  Jiani(Felicia) Zhang   Zheng Gu
"""

from matplotlib import pyplot
import pandas as pd
from pandas import read_csv
import seaborn as sns
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from pylab import scatter, show, legend, xlabel, ylabel
from sklearn.metrics import roc_curve
# read cvs file
filename = "rocketfuel.csv"
names = ['test', 'converted','tot_impr','mode_impr_day', 'mode_impr_hour'] 
rocketfuel = read_csv(filename, names=names,dtype='object')
# get pure data
data = rocketfuel.drop(rocketfuel.index[0]) # drpp row 0: names
#data = data.apply(pd.to_numeric, errors='ignore')
data[names] = data[names].apply(pd.to_numeric, errors='ignore')
print(data.shape)

##----------------------------Question 1 start--------------------------------------
#=================Bayes of test = 0/1  ============
test_group = data.groupby(['test']).get_group(1)
control_group = data.groupby(['test']).get_group(0)
size_test =len(test_group)
size_control =len(control_group)
buys_test = test_group['converted'].sum()
buys_control = control_group['converted'].sum()
print('size_test %d' % size_test)
print('size_control %d' % size_control)
print('buys_test %d' % buys_test)
print('buys_control %d' % buys_control)
print('-------bayes outcome of test = 0/1--------')
P_buys_test = float(buys_test/size_test)
P_buys_control = float(buys_control/size_control)
P_test_buys = float(buys_test/(buys_test+buys_control))
P_control_buys = float(buys_control/(buys_test+buys_control))
print('P_buys_test %f' % P_buys_test)
print('P_buys_control %f' % P_buys_control)
print('P_test_buys %f' % P_test_buys)
print('P_control_buys %f' % P_control_buys)
##----------------------------Question 1 end--------------------------------------

##----------------------------Question 3 start--------------------------------------
#=================Total Impressions============
group_after = 130
#---: test = 1 
group_t_1 = data.groupby(['test']).get_group(1) # choose test = 1
group_ads = group_t_1.groupby(['tot_impr'])
data_t_1 =  group_ads.sum().reset_index().drop(['mode_impr_day','mode_impr_hour'], axis =1)
#=== Group data start: sum rows when tot_impr is big===
r = group_after # shart to sum after tot_impr >= group_after
data_sum_impr1 = data_t_1.head(group_after) # i = 0 - (group_after-1)
len_1 = len(data_t_1)
step = range(0, len_1+1)
i = 1
while True:
    new_row = data_t_1[r:r + step[i]].sum()
    if r+step[i] <= len_1-1:
        new_row['tot_impr'] = data_t_1.ix[r+step[i],:]['tot_impr']
        data_sum_impr1 = data_sum_impr1.append(new_row, ignore_index=True)
        r = r + step[i]
        i = i + 1
#        print(r)
        continue
    else:
        new_row = data_t_1[r:len_1-r].sum()
        new_row['tot_impr'] = data_t_1.ix[len_1-1,:]['tot_impr']
        data_sum_impr1 = data_sum_impr1.append(new_row, ignore_index=True)
        break
        
impr_index = [0,14,29,49,69,89,109,129,145,166]
index_number = [1,2,3,4,5,6,7,8,9,10]
i = 0
impr_ix_label = impr_index
for i in range(0,len(impr_index)):
    impr_ix_label[i] = data_sum_impr1.ix[impr_index[i],:]['tot_impr']

pyplot.xticks([0,14,29,49,69,89,109,129,149,166], list(map(int, impr_ix_label)))
pyplot.bar(data_sum_impr1.index.tolist(),data_sum_impr1['converted']/data_sum_impr1['test'], color='r')
pyplot.xlabel('Ads Impression Number')
pyplot.ylabel('Conversion Rate')
pyplot.legend(['Ads group'])
pyplot.title('Conversion Rate of Ads Impression Number')
pyplot.show()
#---: test = 0
data['count'] = data['test']
data['count'] = 1  # Step1: set 1 to count number of people
group_t_0 = data.groupby(['test']).get_group(0)
group_PSA = group_t_0.groupby(['tot_impr'])
data_t_0 =  group_PSA.sum().reset_index().drop(['mode_impr_day','mode_impr_hour'], axis =1)
#=== Group data start: sum rows when tot_impr is big===
r = group_after # shart to sum after tot_impr >= group_after
data_sum_impr0 = data_t_0.head(group_after) # i = 0 - (group_after-1)
len_0 = len(data_t_0)
step = range(0, len_0+1)
i = 1
while True:
    new_row = data_t_0[r:r + step[i]].sum()
    if r+step[i] <= len_0-1:
        new_row['tot_impr'] = data_t_0.ix[r+step[i],:]['tot_impr']
        data_sum_impr0 = data_sum_impr0.append(new_row, ignore_index=True)
        r = r + step[i]
        i = i + 1
#        print(r)
        continue
    else:
        new_row = data_t_0[r:len_0-r].sum()
        new_row['tot_impr'] = data_t_0.ix[len_0-1,:]['tot_impr']
        data_sum_impr0 = data_sum_impr0.append(new_row, ignore_index=True)
        break
        
#=== Group data end: sum rows when tot_impr is big===
impr_index = [0,19,44,69,90,109,130,150]
index_number = [1,2,3,4,5,6,7,8]
i = 0
impr_ix_label = impr_index
for i in range(0,len(impr_index)):
    impr_ix_label[i] = data_sum_impr0.ix[impr_index[i],:]['tot_impr']
pyplot.xticks([0,19,44,69,90,109,130,150], list(map(int, impr_ix_label)))
#---: Plot Bar
pyplot.bar(data_sum_impr0.index.tolist(), data_sum_impr0['converted']/data_sum_impr0['count'], color = 'g')
#ax = pyplot.gca()
#ax.set_xscale('log')
pyplot.xlabel('PSA Impression Number')
pyplot.ylabel('Conversion Rate')
pyplot.legend(['PSA/control group'])
pyplot.title('Conversion Rate of PSA Impression Number')
pyplot.show()
#---: Plot Scatter
pyplot.scatter(group_ads.groups.keys(), data_t_1['converted']/data_t_1['test'], marker='o', c='r')
pyplot.scatter(group_PSA.groups.keys(), data_t_0['converted']/data_t_0['count'], marker='o', c='g')
xlabel('Ads/PSA')
ylabel('Conversion Rate')
legend(['Ads group','PSA/control group'])
pyplot.title('Conversion Rate of Ads vs PSA group')
pyplot.xlim(0, 250)
show()
##----------------------------Question 3 end--------------------------------------

##----------------------------Question 4 start--------------------------------------
#=================Days of week============
#---: test = 1 
group_t_1 = data.groupby(['test']).get_group(1) # choose test = 1
group_day_t_1 = group_t_1.groupby(['mode_impr_day'])
data_day_1 =  group_day_t_1.sum().reset_index().drop(['mode_impr_hour'], axis =1)
ads_day_1 = data_day_1['tot_impr']            # numbers of ads == group_day.count()['converted'] because all test = 1
buys_day_1  = data_day_1['converted'] # numbers of converted(buys_t_1) 
rate_day_1 = buys_day_1/ads_day_1
#---: test = 0
group_t_0 = data.groupby(['test']).get_group(0) # choose test = 0
group_day_t_0 = group_t_0.groupby(['mode_impr_day'])
data_day_0 =  group_day_t_0.sum().reset_index().drop(['mode_impr_hour'], axis =1)
ads_day_0 = data_day_0['tot_impr']            # numbers of ads == group_day.count()['converted'] because all test = 1
buys_day_0  = data_day_0['converted'] # numbers of converted(buys_t_1) 
rate_day_0 = buys_day_0/ads_day_0
#---: plot in one bar chart
fig, ax = pyplot.subplots()
index = data_day_1['mode_impr_day']
bar_width = 0.35
opacity = 0.8
rects1 = pyplot.bar(index, rate_day_1, bar_width,
                 alpha=opacity,
                 color='r',
                 label='Ads group')

rects0 = pyplot.bar(index + bar_width, rate_day_0, bar_width,
                 alpha=opacity,
                 color='g',
                 label='PSA/control group')
 
pyplot.xlabel('Days of Week')
pyplot.ylabel('Conversion Rate')
pyplot.title('Conversion Rate of different Days of Week')
pyplot.xticks(index + bar_width, ('Mon', 'Tu', 'Wen', 'Th', 'Fri', 'Sa', 'Sun'))
pyplot.legend()
pyplot.tight_layout()
pyplot.show()

#=================Hours of Day============
#---: test = 1 
group_t_1 = data.groupby(['test']).get_group(1) # choose test = 1
group_hour_t_1 = group_t_1.groupby(['mode_impr_hour'])
data_hour_1 =  group_hour_t_1.sum().reset_index().drop(['mode_impr_day'], axis =1)
ads_hour_1 = data_hour_1['tot_impr']            # numbers of ads == group_hour.count()['converted'] because all test = 1
buys_hour_1  = data_hour_1['converted'] # numbers of converted(buys_t_1) 
rate_hour_1 = buys_hour_1/ads_hour_1
#---: test = 0
group_t_0 = data.groupby(['test']).get_group(0) # choose test = 0
group_hour_t_0 = group_t_0.groupby(['mode_impr_hour'])
data_hour_0 =  group_hour_t_0.sum().reset_index().drop(['mode_impr_day'], axis =1)
ads_hour_0 = data_hour_0['tot_impr']            # numbers of ads == group_hour.count()['converted'] because all test = 1
buys_hour_0  = data_hour_0['converted'] # numbers of converted(buys_t_1) 
rate_hour_0 = buys_hour_0/ads_hour_0
#---: plot in one bar chart
fig, ax = pyplot.subplots()
index = data_hour_1['mode_impr_hour']
bar_width = 0.35
opacity = 0.8
rects1 = pyplot.bar(index, rate_hour_1, bar_width,
                 alpha=opacity,
                 color='r',
                 label='Ads group')

rects0 = pyplot.bar(index + bar_width, rate_hour_0, bar_width,
                 alpha=opacity,
                 color='g',
                 label='PSA/control group')
 
pyplot.xlabel('hours of Day')
pyplot.ylabel('Conversion Rate')
pyplot.title('Conversion Rate of different hours of Day')
pyplot.legend()
pyplot.tight_layout()
pyplot.show()

##======: CI plot of Days & Hours
sns.pointplot(x=data['mode_impr_day'], y=data['tot_impr'], data=data, capsize=.2)
pyplot.title('95% CI of mean Impression Number by Days of Week')
pyplot.show()

sns.pointplot(x=data['mode_impr_hour'], y=data['tot_impr'], data=data, capsize=.2)
pyplot.title('95% CI of mean Impression Number by Hours of Day')
pyplot.show()

##----------------------------Question 4 end--------------------------------------

#----------------------------Logistic Regression start-----------------------------
##======: Correlation plot of shows relationship of X & y
sns.heatmap(data.groupby(['test']).get_group(1).corr())
pyplot.title('Correlation Coefficient of all variables')
pyplot.show()

group_t_1 = data.groupby(['test']).get_group(1)
X = group_t_1[['tot_impr']]
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
y = group_t_1[['converted']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

#Discard_X_train, Discard_X_test, y_train, y_test = train_test_split(X_test,
#                                                            y_test,
#                                                            test_size=0.5)

LogisticR = LogisticRegression()
LR_fit1 = LogisticR.fit(X_train, y_train)
y_pred_rt = LogisticR.predict_proba(X_test)[:, 1]
fpr_rt_lm, tpr_rt_lm, _ = roc_curve(y_test, y_pred_rt)

print ('Test score of Logistic Regression: ')
#x_test_2D = np.reshape(x_test, (-1, 1))
print( LR_fit1.score(X_test,y_test))

#--------------------------visualize data----------------------------------------
#---: ROC Curve
pyplot.figure(1)
pyplot.plot([0, 1], [0, 1], 'k--')
pyplot.plot(fpr_rt_lm, tpr_rt_lm, label='Logistic Regression',c='b')
pyplot.xlabel('False positive rate')
pyplot.ylabel('True positive rate')
pyplot.title('ROC curve')
pyplot.legend(loc='best')
pyplot.show()

#---: Scatter Plot
scatter(min_max_scaler.fit_transform(X_test), y_pred_rt, marker='x', c='deepskyblue')
xlabel('Scaled Impression Number')
ylabel('Conversion Rate')
pyplot.title('Predicted Conversion Rate')
show()

scatter(min_max_scaler.fit_transform(X_test), y_test, marker='x', c='deepskyblue')
xlabel('Scaled Impression Number')
ylabel('Conversion Outcome')
pyplot.title('True Conversion Outcome')
show()

#----------------------------Logistic Regression end--------------------------------------
#Other plot
#data['count'] = data['test']
#data['count'] = 1  # Step1: set 1 to count number of people

pyplot.xlabel('Total Impression Numbers')
pyplot.ylabel('Frenquency')
pyplot.title('Distribution of Total Impression Numbers')
pyplot.hist(data['tot_impr'])
pyplot.show()

impr = data.groupby(['tot_impr'])
pyplot.plot(impr.groups.keys(), impr.sum()['converted']/impr.sum()['count'])
pyplot.xlabel('Total Impression Numbers')
pyplot.ylabel('Conversion Rate')
pyplot.title('Conversion Rate of Impression Numbers')
pyplot.show()

