# IPython log file


import sys; print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(['D:\\study\\pydata'])
years = range(1880, 2011)  #this will generate sequence from 1880 to 2010
pieces = []
columns = ['name', 'sex', 'births']
for year in years:
    path = './pydata-book/ch02/names/yob%d.txt' % year
    frame = pd.read_csv(path, names=columns)
    frame['year'] = year
    pieces.append(frame)
# concatenate everything into a single data frame
names = pd.concat(pieces, ignore_index=True)
# we have to pass ignore_index = True because we're not in interested
# in preserving the original row numbers returned from read_csv
names
total_births = names.pivot_table('births', rows='year', cols='sex', aggfunc=sum)
total_births.tail()
total_births.plot(title='Total Birth by Sex and Year')
#insert a column prop with the fraction of babies given each name relative to the total number
#of births.
#We first group the data by year and sex, then add the new column to each group
def add_prop(group):
    births = group.births.astype(float)  #we first need to cast integer to floating
    group['prop'] = births / births.sum() #births.sum() automatically sum by group
                                          #once the data frame has been grouped
    return group
names = names.groupby(['year', 'sex']).apply(add_prop)  #great use of apply!!
names
class(names.groupby(['year', 'sex']))   #pandas.core.groupby.DataFrameGroupBy
#!!! when performing a group operation like this, it's often valuable to do a sanity check
#like verifying that the prop column of each group sums to 1
#np.allclose
import numpy as np
np.allclose(names.groupby(['year', 'sex']).prop.sum(), 1)
#extract a subset to facilitate further analysis
#top 10000 names for each sex/year combination
def get_top1000(group):
    return group.sort_index(by='births', ascending=False)[:1000]
top1000 = names.groupby(['year', 'sex']).apply(get_top1000)
top1000
#if you prefer DIY approach, you could also do:
pieces = []
for year, group in names.groupby(['year', 'sex']):
    pieces.append(group.sort_index(by='births', ascending=False)[:1000])
top1000 = pd.concat(pieces, ignore_index=True)
top1000.tail()
#analyzing naming trends
boys = top1000[top1000.sex == 'M']
girls = top1000[top1000.sex == 'F']
total_births = top1000.pivot_table('births', rows='year', cols='name',aggfunc=sum)
subset = total_births[['John', 'Harry', 'Mary', 'Marilyn']]
subset.plot(subplots=True, figsize=(12, 10), grid=False, title='Number of Births per Year')
#Measuring the increase in naming diversity
prop_table = top1000.pivot_table('prop', rows='year', cols='sex', aggfunc=sum)
prop_table.plot(title='Sum of Table1000.Prop by Year and Sex', yticks=np.linspace(0, 1.2, 13), xticks=range(1880, 2020, 10))
#how many of the most popular names it takes to reach 50%.
df2010 = boys[boys.year == 2010]
df2010
prop_cumsum = df2010.sort_index(by='prop', ascending=False).prop.cumsum()
prop_cumsum[:10]
prop_cumsum.values.searchsorted(0.5) + 1  #searchsorted is a method of numpy.ndarray
        #plus 1 is because arrays are zero-indexed
type(prop_cumsum.values)
df1900 = boys[boys.year == 1900]
in1900 = df1900.sort_index(by='prop', ascending=False).prop.cumsum()
in1900.values.searchsorted(0.5) + 1  #so 2010's names are more diversified
def get_quantile_count(group, q=0.5):
    group = group.sort_index(by='prop', ascending=False)
    return group.prop.cumsum().values.searchsorted(q) + 1
diversity = top1000.groupby(['year', 'sex']).apply(get_quantile_count)
diversity = diversity.unstack('sex')  #now diversity has two time series, ne for each sex, indexed by year
diversity.head()
diversity.plot(title='Number of Popular Names in Top 50%')
#The 'Last Letter' Revolution
#exact last letter from name column
get_last_letter = lambda x: x[-1]
last_letters = names.name.map(get_last_letter)
    #Difference between map, applymap and apply methods in Pandas
    #map is a Series method whereas the rest are DataFrame methods.
    #apply is column-or-row-wise. applymap and map are element-wise.
last_letters.name = 'last_letter'
table = names.pivot_table('births', rows=last_letters, cols=['sex', 'year'], aggfunc=sum)
suitable = table.reindex(columns=[1910, 1960, 2010], level='year')
#DataFrame.reindex(index=None, columns=None, **kwargs)
type(suitable)
suitable.head()
suitable.sum()  #Return the sum of the values for the requested axis(axis : {row (0), columns (1)})
type(suitable.sum())   #pandas.core.series.Series
letter_prop = subtable/subtable.sum().astype(float)
import matplotlib.pyplot as plt
fig, axes = plt.subplots(2, 1, figsize=(10, 10))
letter_prop['M'].plot(kind='bar', rot=0, ax=axes[0], title='Male')
letter_prop['F'].plot(kind='bar', rot=0, ax=axes[1], title='Female', legend=False)
#boy names ending in 'n' have experienced significant growth since the 1960s
full_letter_prop = table/table.sum().astype(float)
dny_ts = full_letter_prop.ix[['d', 'n', 'y'], 'M'].T
dny_ts.head()
type(dny_ts)  #pandas.core.frame.DataFrame
dny_ts.plot()
#boy names that became girl names (and vice versa)
all_names = top1000.name.unique()
mask = np.array(['lesl' in x.lower() for x in all_names])
lesley_like = all_names[mask]
lesley_like
years = range(1880, 2011)  #this will generate sequence from 1880 to 2010
pieces = []
columns = ['name', 'sex', 'births']
for year in years:
    path = './pydata-book/ch02/names/yob%d.txt' % year
    frame = pd.read_csv(path, names=columns)
    frame['year'] = year
    pieces.append(frame)
# concatenate everything into a single data frame
names = pd.concat(pieces, ignore_index=True)
# we have to pass ignore_index = True because we're not in interested
# in preserving the original row numbers returned from read_csv
names
total_births = names.pivot_table('births', rows='year', cols='sex', aggfunc=sum)
total_births.tail()
total_births.plot(title='Total Birth by Sex and Year')
#insert a column prop with the fraction of babies given each name relative to the total number
#of births.
#We first group the data by year and sex, then add the new column to each group
def add_prop(group):
    births = group.births.astype(float)  #we first need to cast integer to floating
    group['prop'] = births / births.sum() #births.sum() automatically sum by group
                                          #once the data frame has been grouped
    return group
names = names.groupby(['year', 'sex']).apply(add_prop)  #great use of apply!!
names
class(names.groupby(['year', 'sex']))   #pandas.core.groupby.DataFrameGroupBy
#!!! when performing a group operation like this, it's often valuable to do a sanity check
#like verifying that the prop column of each group sums to 1
#np.allclose
import numpy as np
np.allclose(names.groupby(['year', 'sex']).prop.sum(), 1)
#extract a subset to facilitate further analysis
#top 10000 names for each sex/year combination
def get_top1000(group):
    return group.sort_index(by='births', ascending=False)[:1000]
top1000 = names.groupby(['year', 'sex']).apply(get_top1000)
top1000
#if you prefer DIY approach, you could also do:
pieces = []
for year, group in names.groupby(['year', 'sex']):
    pieces.append(group.sort_index(by='births', ascending=False)[:1000])
top1000 = pd.concat(pieces, ignore_index=True)
top1000.tail()
#analyzing naming trends
boys = top1000[top1000.sex == 'M']
girls = top1000[top1000.sex == 'F']
total_births = top1000.pivot_table('births', rows='year', cols='name',aggfunc=sum)
subset = total_births[['John', 'Harry', 'Mary', 'Marilyn']]
subset.plot(subplots=True, figsize=(12, 10), grid=False, title='Number of Births per Year')
#Measuring the increase in naming diversity
prop_table = top1000.pivot_table('prop', rows='year', cols='sex', aggfunc=sum)
prop_table.plot(title='Sum of Table1000.Prop by Year and Sex', yticks=np.linspace(0, 1.2, 13), xticks=range(1880, 2020, 10))
#how many of the most popular names it takes to reach 50%.
df2010 = boys[boys.year == 2010]
df2010
prop_cumsum = df2010.sort_index(by='prop', ascending=False).prop.cumsum()
prop_cumsum[:10]
prop_cumsum.values.searchsorted(0.5) + 1  #searchsorted is a method of numpy.ndarray
        #plus 1 is because arrays are zero-indexed
type(prop_cumsum.values)
df1900 = boys[boys.year == 1900]
in1900 = df1900.sort_index(by='prop', ascending=False).prop.cumsum()
in1900.values.searchsorted(0.5) + 1  #so 2010's names are more diversified
def get_quantile_count(group, q=0.5):
    group = group.sort_index(by='prop', ascending=False)
    return group.prop.cumsum().values.searchsorted(q) + 1
diversity = top1000.groupby(['year', 'sex']).apply(get_quantile_count)
diversity = diversity.unstack('sex')  #now diversity has two time series, ne for each sex, indexed by year
diversity.head()
diversity.plot(title='Number of Popular Names in Top 50%')
#The 'Last Letter' Revolution
#exact last letter from name column
get_last_letter = lambda x: x[-1]
last_letters = names.name.map(get_last_letter)
    #Difference between map, applymap and apply methods in Pandas
    #map is a Series method whereas the rest are DataFrame methods.
    #apply is column-or-row-wise. applymap and map are element-wise.
last_letters.name = 'last_letter'
table = names.pivot_table('births', rows=last_letters, cols=['sex', 'year'], aggfunc=sum)
suitable = table.reindex(columns=[1910, 1960, 2010], level='year')
#DataFrame.reindex(index=None, columns=None, **kwargs)
type(suitable)
suitable.head()
suitable.sum()  #Return the sum of the values for the requested axis(axis : {row (0), columns (1)})
type(suitable.sum())   #pandas.core.series.Series
letter_prop = subtable/subtable.sum().astype(float)
import matplotlib.pyplot as plt
fig, axes = plt.subplots(2, 1, figsize=(10, 10))
letter_prop['M'].plot(kind='bar', rot=0, ax=axes[0], title='Male')
letter_prop['F'].plot(kind='bar', rot=0, ax=axes[1], title='Female', legend=False)
#boy names ending in 'n' have experienced significant growth since the 1960s
full_letter_prop = table/table.sum().astype(float)
dny_ts = full_letter_prop.ix[['d', 'n', 'y'], 'M'].T
dny_ts.head()
type(dny_ts)  #pandas.core.frame.DataFrame
dny_ts.plot()
#boy names that became girl names (and vice versa)
all_names = top1000.name.unique()
mask = np.array(['lesl' in x.lower() for x in all_names])
lesley_like = all_names[mask]
lesley_like
columns = ['name', 'sex', 'births']
for year in years:
    path = './pydata-book/ch02/names/yob%d.txt' % year
    frame = pd.read_csv(path, names=columns)
    frame['year'] = year
    pieces.append(frame)
import pandas as pd
pieces = []
columns = ['name', 'sex', 'births']
for year in years:
    path = './pydata-book/ch02/names/yob%d.txt' % year
    frame = pd.read_csv(path, names=columns)
    frame['year'] = year
    pieces.append(frame)
names = pd.concat(pieces, ignore_index=True)
names
total_births = names.pivot_table('births', rows='year', cols='sex', aggfunc=sum)
total_births.tail()
total_births.plot(title='Total Birth by Sex and Year')
#of births.
#insert a column prop with the fraction of babies given each name relative to the total number
#We first group the data by year and sex, then add the new column to each group
births = group.births.astype(float)  #we first need to cast integer to floating
def add_prop(group):

    group['prop'] = births / births.sum() #births.sum() automatically sum by group
#once the data frame has been grouped
return group
names = names.groupby(['year', 'sex']).apply(add_prop)  #great use of apply!!
import numpy as np
np.allclose(names.groupby(['year', 'sex']).prop.sum(), 1)
class(names.groupby(['year', 'sex']))   #pandas.core.groupby.DataFrameGroupBy
#!!! when performing a group operation like this, it's often valuable to do a sanity check
#np.allclose
#like verifying that the prop column of each group sums to 1
from collections import defaultdict
from collections import Counter
from pandas import DataFrame, Series
names = pd.concat(pieces, ignore_index=True)
names
total_births = names.pivot_table('births', rows='year', cols='sex', aggfunc=sum)
total_births.tail()
total_births.plot(title='Total Birth by Sex and Year')
def add_prop(group):

    births = group.births.astype(float)  #we first need to cast integer to floating
group['prop'] = births / births.sum() #births.sum() automatically sum by group
#once the data frame has been grouped
return group
def add_prop(group):
    births = group.births.astype(float)  #we first need to cast integer to floating
    group['prop'] = births / births.sum() #births.sum() automatically sum by group
                                          #once the data frame has been grouped
    return group
names = names.groupby(['year', 'sex']).apply(add_prop)  #great use of apply!!
names
class(names.groupby(['year', 'sex']))   #pandas.core.groupby.DataFrameGroupBy
type(names.groupby(['year', 'sex']))   #pandas.core.groupby.DataFrameGroupBy
import numpy as np
np.allclose(names.groupby(['year', 'sex']).prop.sum(), 1)
def get_top1000(group):

    return group.sort_index(by='births', ascending=False)[:1000]
top1000 = names.groupby(['year', 'sex']).apply(get_top1000)
top1000
pieces = []
for year, group in names.groupby(['year', 'sex']):
    pieces.append(group.sort_index(by='births', ascending=False)[:1000])
top1000 = pd.concat(pieces, ignore_index=True)
top1000.tail()
boys = top1000[top1000.sex == 'M']
girls = top1000[top1000.sex == 'F']
total_births = top1000.pivot_table('births', rows='year', cols='name',aggfunc=sum)
subset = total_births[['John', 'Harry', 'Mary', 'Marilyn']]
subset.plot(subplots=True, figsize=(12, 10), grid=False, title='Number of Births per Year')
prop_table = top1000.pivot_table('prop', rows='year', cols='sex', aggfunc=sum)
prop_table.plot(title='Sum of Table1000.Prop by Year and Sex', yticks=np.linspace(0, 1.2, 13), xticks=range(1880, 2020, 10))
#how many of the most popular names it takes to reach 50%.
df2010 = boys[boys.year == 2010]
df2010
prop_cumsum = df2010.sort_index(by='prop', ascending=False).prop.cumsum()
prop_cumsum[:10]
prop_cumsum.values.searchsorted(0.5) + 1  #searchsorted is a method of numpy.ndarray
        #plus 1 is because arrays are zero-indexed
type(prop_cumsum.values)
df1900 = boys[boys.year == 1900]
in1900 = df1900.sort_index(by='prop', ascending=False).prop.cumsum()
in1900.values.searchsorted(0.5) + 1  #so 2010's names are more diversified
def get_quantile_count(group, q=0.5):
    group = group.sort_index(by='prop', ascending=False)
    return group.prop.cumsum().values.searchsorted(q) + 1
diversity = top1000.groupby(['year', 'sex']).apply(get_quantile_count)
diversity = diversity.unstack('sex')  #now diversity has two time series, ne for each sex, indexed by year
diversity.head()
diversity.plot(title='Number of Popular Names in Top 50%')
get_last_letter = lambda x: x[-1]
last_letters = names.name.map(get_last_letter)
    #Difference between map, applymap and apply methods in Pandas
    #map is a Series method whereas the rest are DataFrame methods.
    #apply is column-or-row-wise. applymap and map are element-wise.
last_letters.name = 'last_letter'
table = names.pivot_table('births', rows=last_letters, cols=['sex', 'year'], aggfunc=sum)
suitable = table.reindex(columns=[1910, 1960, 2010], level='year')
#DataFrame.reindex(index=None, columns=None, **kwargs)
type(suitable)
suitable.head()
suitable.sum()  #Return the sum of the values for the requested axis(axis : {row (0), columns (1)})
type(suitable.sum())   #pandas.core.series.Series
letter_prop = subtable/subtable.sum().astype(float)
letter_prop = suitable/subtable.sum().astype(float)
last_letters.name = 'last_letter'
table = names.pivot_table('births', rows=last_letters, cols=['sex', 'year'], aggfunc=sum)
suitable = table.reindex(columns=[1910, 1960, 2010], level='year')
#DataFrame.reindex(index=None, columns=None, **kwargs)
type(suitable)
suitable.head()
suitable.sum()  #Return the sum of the values for the requested axis(axis : {row (0), columns (1)})
type(suitable.sum())   #pandas.core.series.Series
letter_prop = suitable/subtable.sum().astype(float)
letter_prop = suitable/suitable.sum().astype(float)
subtable = table.reindex(columns=[1910, 1960, 2010], level='year')
#DataFrame.reindex(index=None, columns=None, **kwargs)
type(suitable)
subtable.head()
subtable.sum()  #Return the sum of the values for the requested axis(axis : {row (0), columns (1)})
type(suitable.sum())   #pandas.core.series.Series
letter_prop = subtable/subtable.sum().astype(float)
import matplotlib.pyplot as plt
fig, axes = plt.subplots(2, 1, figsize=(10, 10))
letter_prop['M'].plot(kind='bar', rot=0, ax=axes[0], title='Male')
letter_prop['F'].plot(kind='bar', rot=0, ax=axes[1], title='Female', legend=False)
#boy names ending in 'n' have experienced significant growth since the 1960s
full_letter_prop = table/table.sum().astype(float)
dny_ts = full_letter_prop.ix[['d', 'n', 'y'], 'M'].T
dny_ts.head()
type(dny_ts)  #pandas.core.frame.DataFrame
dny_ts.plot()
#boy names that became girl names (and vice versa)
all_names = top1000.name.unique()
mask = np.array(['lesl' in x.lower() for x in all_names])
lesley_like = all_names[mask]
lesley_like
filtered = top1000[top1000.name.isin(lesley_like)]
filtered.groupby('name').births.sum()
table = filtered.pivot_table('birth', rows='year', cols='sex', aggfunc='sum')
table = filtered.pivot_table('births', rows='year', cols='sex', aggfunc='sum')
table = table.div(table.sum(1), axis=0)
table
table.tail()
table.head()
table.plot(style={'M': 'k-', 'F': 'k--'})
table.plot(style={'M': 'k-', 'F': 'k--'}, title='Proportion of male/female Lesley-like names over time')
data = {i: randn() for i in xrange(7)}
data = {i : randn() for i in xrange(7)}
a
a = 5
a
from numpy import *
data = {i : randn() for i in xrange(7)}
data = {i : randn() for i in range(7)}
from scipy import *
data = {i : randn() for i in xrange(7)}
data2 = {i : np.random.randn() for i in xrange(7)}
data
data2
print data
import numpy.random as randn
data = {i : randn() for i in xrange(7)}
data
import numpy.random as randn
data = {i : randn() for i in xrange(7)}
data = {i:randn() for i in xrange(7)}
clear
get_ipython().magic(u'reset ')
import numpy.random as randn
data = {i:randn() for i in xrange(7)}
numpy.random
from numpy.random import randn
data = {i:randn() for i in xrange(7)}
data
print data  # if
import datatime
import datatime
import datetime
datatime.
datetime.timedelta
b = [1, 2, 3]
b = [1, 2, 3]
get_ipython().magic(u'pinfo b')
get_ipython().magic(u'pinfo b')
get_ipython().magic(u'pinfo b')
import numpy as np
get_ipython().magic(u'psearch np.*load*')
_
__
--
__
_
_27
_i27
_i100
_100
_i99
get_ipython().magic(u'run')
get_ipython().magic(u'logstart')
get_ipython().magic(u'logstate')
get_ipython().magic(u'logon')
get_ipython().magic(u'timeit x.startswith(y)')
x = 'foobar'
y = 'foo'
get_ipython().magic(u'timeit x.startswith(y)')
get_ipython().magic(u'timeit x[:3] == y')
get_ipython().magic(u'run')
get_ipython().magic(u'run prof_mod')
def add_and_sum(x, y):
    added = x + y
    summed = added.sum(axis=1)
    return summed
def call_function():
    x = randn(1000, 1000)
    y = randn(1000, 1000)
    return add_and_sum(x, y)
q
get_ipython().magic(u'prun add_and_sum(x,y)')
get_ipython().magic(u'prun add_and_sum(x,y)')
def add_and_sum(x, y):
    added = x + y
    summed = added.sum(axis=1)
    return summed
def call_function():
    x = randn(1000, 1000)
    y = randn(1000, 1000)
    return add_and_sum(x, y)
get_ipython().magic(u'prun add_and_sum(x,y)')
def add_and_sum(x, y):
    added = x + y
    summed = added.sum(axis=1)
    return summed
from numpy.random import randn
def call_function():
    x = randn(1000, 1000)
    y = randn(1000, 1000)
    return add_and_sum(x, y)
get_ipython().magic(u'prun add_and_sum(x,y)')
get_ipython().magic(u'prun call_function()')
c.TerminalIPythonApp.extensions = ['line_profiler']
ipython notebook  %pylab inline
ipython notebook %pylab inline
get_ipython().magic(u'pylab inline')
ipython notebook --pylab=inline
ipython notebook --pylab=inline
ipython notebook --pylab=inline
get_ipython().magic(u'pylab inline')
img = plt.imread('./pydata-book/ch03/stinkbug.png')
import matplotlib.pyplot as plt
img = plt.imread('./pydata-book/ch03/stinkbug.png')
imshow(img)
img = plt.imread('./pydata-book/ch03/stinkbug.png')
imshow(img)
import matplotlib.pyplot as plt
img = plt.imread('./pydata-book/ch03/stinkbug.png')
imshow(img)
img
from pylab import *
imshow(img)
img = plt.imread('./pydata-book/ch03/stinkbug.png')
img
imshow(img)
plot(randn(1000).cumsum())
plt.show()
imshow(img)
plt.show()
plt.show(img)
plt.show(imshow(img))
imshow(img)
plt.imshow(img)
img = plt.imread('./pydata-book/ch03/stinkbug.png')
plt.imshow(img)
from pylab import *
import matplotlib.pyplot as plt
img = plt.imread('./pydata-book/ch03/stinkbug.png')
get_ipython().magic(u'reset ')
import matplotlib.pyplot as plt
from pylab import *
plt.imshow()
plt.show()
img = plt.imread('./pydata-book/ch03/stinkbug.png')
imshow(img)
plot(randn(1000).cumsum())
