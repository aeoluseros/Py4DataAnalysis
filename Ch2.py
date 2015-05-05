__author__ = 'aeoluseros'
# --coding: utf-8
###JSON data

#1. read data
path = './pydata-book/ch02/usagov_bitly_data2012-03-16-1331923249.txt'
open(path).readline()

import json
records = [json.loads(line) for line in open(path)]   #list comprehension
records[0]     #records[0] is a dictionary

records[0]['tz'] #the 'u' infront of 'tz' stands for unicode.
print records[0]['tz']
type(records)         #list
type(records[0])      #dictionary
type(records[0]['tz'])   #unicode

time_zones = [rec['tz'] for rec in records] #error, because not all records have a tz
time_zones = [rec['tz'] for rec in records if 'tz' in rec]   #create a list
type(time_zones)        #list
print time_zones[0]
print time_zones[:10]  #some tz are missing, you can fill these out but I'll leave them in for now.

#To produce counts by time zeone I'll show three approaches: #the normal way
   #the harder way: using just the Python standard library
   #the easier way; using pandas

#(1) the normal way
def get_counts(sequence):
    counts = {}             #counts is a dictionary
    for x in sequence:
        #print "x is:", x
        if x in counts:
            counts[x] += 1      #x is the keyname. counts[x] is the value.
        else:
            counts[x] = 1
    return counts

#the above normal way could be realized by the standard library
from collections import defaultdict
def get_counts2(sequence):
    counts = defaultdict(int)   #values will initialize to 0
    for x in sequence:
        counts[x] += 1
    return counts

counts = get_counts(time_zones)  # or counts = get_counts(time_zones)

type(counts)    #dictionary
counts['America/New_York']
len(time_zones)
len(counts)
counts.items()

#top 10 time zones and their counts
def top_counts(count_dict, n=10):
    value_key_pairs = [(count, tz) for tz, count in count_dict.items()]
    value_key_pairs.sort()       #sort by key
    return value_key_pairs[-n:]

top_counts(counts)

#(2) Use python standard library - collections.Counter
from collections import Counter
counts = Counter(time_zones)
counts.most_common(10)

#(3) Counting Time zones with pandas
from pandas import DataFrame, Series
import pandas as pd
frame = DataFrame(records)
type(frame)

frame['tz'][:10]
type(frame['tz'])   #pandas.core.series.Series
#pandas.core.series.Series has a method value_counts
tz_counts = frame['tz'].value_counts()
tz_counts[:10]
type(tz_counts)  #pandas.core.series.Series

#2. plot the data
#(1) fill in MA data
#fillna()
clean_tz = frame['tz'].fillna('Missing')   #fillna is used to replace NA
   #filling method : {‘backfill’, ‘bfill’, ‘pad’, ‘ffill’, None}, default None
clean_tz[clean_tz == ''] = 'Unknown'   #empty strings can be replaced by boolean array indexing
tz_counts = clean_tz.value_counts()
tz_counts[:10]

#panda Series has a method plot
tz_counts[:10].plot(kind='barh', rot = 0) #no rotation --> horizontal
    # ‘bar’ or ‘barh’ for bar plots
    # ‘hist’ for histogram
    # ‘box’ for boxplot
    # ‘kde’ or 'density' for density plots
    # ‘area’ for area plots
    # ‘scatter’ for scatter plots
    # ‘hexbin’ for hexagonal bin plots
    # ‘pie’ for pie plots

frame['a'][1]
frame.a[1]        #two kinds of expression way.
frame['a'][50]
frame.a[50]
frame['a'][51]
frame.a[51]

#dropna()
results = Series([x.split()[0] for x in frame.a.dropna()])
results[:5]
results.value_counts()[:8]

#notnull()
cframe = frame[frame.a.notnull()]

import numpy as np
#numpy.where: Return elements, either from x or y, depending on condition.
operating_system = np.where(cframe['a'].str.contains('Windows'), 'Windows', 'NotWindows')
operating_system[:5]

#Group the day by its time zone columns and this new listing of operating systems:
by_tz_os = cframe.groupby(['tz', operating_system])
agg_counts = by_tz_os.size().unstack().fillna(0)   #if by_tz_os.size().unstack() is NaN, then fill with 0.
type(agg_counts)

#select the top overall time
# To do so, I construct an indirect index array from the row counts in agg_counts:
indexer = agg_counts.sum(1).argsort()     #ascending
#DataFrame.sum(axis=None, skipna=None, level=None, numeric_only=None, **kwargs)
# axis : {index (0), columns (1)}  . so here we add by column
type(agg_counts.sum(1))   #pandas.core.series.Series
#Series.argsort(axis=0, kind='quicksort', order=None)
      # Overrides ndarray.argsort. Argsorts the value, omitting NA/null values, and places the result
      # in the same locations as the non-NA values.
        # axis : int (can only be zero)
        # kind : {‘mergesort’, ‘quicksort’, ‘heapsort’}, default ‘quicksort’
        # Choice of sorting algorithm. See np.sort for more information. ‘mergesort’ is the only stable algorithm
        # order : ignored
      # Return: argsorted -- Series, with -1 indicated where nan values are present

#take the last(largest) ten
count_subset = agg_counts.take(indexer)[-10:]   #dataframe.take(): Analogous to ndarray.take, translate neg to pos indices (default)
count_subset.plot(kind='barh', stacked=True)
normed_subset = count_subset.div(count_subset.sum(1), axis=0) #count_subset / count_subset.sum(1)
normed_subset.plot(kind='barh', stacked=True)


###MovieLens 1M Data Set
#1. read data with pd.read_table()
import pandas as pd
unames = ['user_id', 'gender', 'age', 'occupation', 'zip']
users = pd.read_table(".\pydata-book\ch02\movielens\users.dat", sep='::',
                      header=None, names=unames)
users[:5]
rnames = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings = pd.read_table(".\pydata-book\ch02\movielens\\ratings.dat",
                        sep='::', header=None, names=rnames)
ratings[:5]
mnames = ['movie_id', 'title', 'genres']
movies = pd.read_table('.\pydata-book\ch02\movielens\movies.dat', sep='::',
                       header=None, names=mnames)
movies[:5]
ratings


#2. Analyze
# analyzing data across three tables is not a simple task
# this is much easier to do with all of the data merged together

data = pd.merge(pd.merge(ratings, users), movies)
type(data)        #data.frame
data[:10]
data.ix[0]   #because we can't use data[0]
data['user_id']
data

#DataFrame.xs(key, axis=0, level=None, copy=None, drop_level=True) --> MultiIndex Slicers
        # >>> df
        #                     A  B  C  D
        # first second third
        # bar   one    1      4  1  8  9
        #       two    1      7  5  5  0
        # baz   one    1      6  6  8  0
        #       three  2      5  3  5  3
        # >>> df.xs(('baz', 'three'))
        #        A  B  C  D
        # third
        # 2      5  3  5  3


#to get mean movie ratings for each film groued by gender, we can
#use the pivot_table method.

mean_ratings = data.pivot_table('rating', rows='title', cols='gender', aggfunc='mean')
mean_ratings[:5]   #another data frame

ratings_by_title = data.groupby('title').size()
ratings_by_title[:10]
active_titles_series = ratings_by_title.ix[ratings_by_title >= 250]   #series
active_titles = ratings_by_title.index[ratings_by_title >= 250]
type(active_titles)  #pandas.core.index.Index
active_titles

mean_ratings = mean_ratings.ix[active_titles]
mean_ratings   #1216

top_female_ratings = mean_ratings.sort_index(by='F', ascending=False)
top_female_ratingstop_female_ratings['F'][:10]


# measuring rating disagreement
mean_ratings['diff'] = mean_ratings['M'] - mean_ratings['F']
sorted_by_diff = mean_ratings.sort_index(by='diff')
sorted_by_diff[::-1][:15]
# https://docs.python.org/release/2.3.5/whatsnew/section-slices.html
#the slicing syntax has supported an optional third ``step'' or ``stride'' argument: L[1:10:2]
# L = range(10)
# L[::2]
# [0, 2, 4, 6, 8]
#Negative values also work to make a copy of the same list in reverse order:
#>>> L[::-1]
# [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
# L[::-2]
# [9, 7, 5, 3, 1]
# s='abcd'
# s[::2]
# 'ac'
# s[::-1]
# 'dcba'

#Std of rating grouped by title
rating_std_by_title = data.groupby('title')['rating'].std()
rating_std_by_title
rating_std_by_active_title = rating_std_by_title.ix[active_titles]
rating_std_by_active_title.order(ascending=False)[:10]



###US Baby Names (1880 - 2010)
names1880 = pd.read_csv('./pydata-book/ch02/names/yob1880.txt', names=['name','sex','births'])
names1880.head(10)  #same as names1880[:10]
names1880.head()  #default to show 5 records

names1880.groupby('sex').births.sum()   #how many children were born
#same as:
names1880.groupby('sex')['births'].sum()

names1880.groupby('sex').births.size()  #how many different names

#1. read in files
# data set is split into files by year, we need to assemble all into a single DF
# pandas.concat
# 2010 is the last available year right now
years = range(1880, 2011)  #this will generate sequence from 1880 to 2010
import pandas as pd
from pandas import DataFrame, Series

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
type(names.groupby(['year', 'sex']))   #pandas.core.groupby.DataFrameGroupBy

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
    #Difference between map, applymap and apply methods in Pandas:
    #map is a Series method whereas the rest are DataFrame methods.
    #apply is column-or-row-wise. applymap and map are element-wise.
last_letters.name = 'last_letter'
table = names.pivot_table('births', rows=last_letters, cols=['sex', 'year'], aggfunc=sum)
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

filtered = top1000[top1000.name.isin(lesley_like)]   #isin()
filtered.groupby('name').births.sum()
#aggregate by sex and year and normalize within year
table = filtered.pivot_table('births', rows='year', cols='sex', aggfunc='sum')
table = table.div(table.sum(1), axis=0)
table.tail()
table.head()
table.plot(style={'M': 'k-', 'F': 'k--'}, title='Proportion of male/female Lesley-like names over time')


####Addition######
#Python: List to Dictionary
# Let's say I have a list a in Python whose entries conveniently map to a dictionary. Each even element represents
# the key to the dictionary, and the following odd element is the value.
a = ['hello','world','1','2']
#and I'd like to convert it to a dictionary b, where
#b['hello'] = 'world'
#b['1'] = '2'
b = dict(zip(a[0::2], a[1::2]))  #dict([('hello', 'world'), ('1', '2')]
b
#If a is large, you will probably want to do something like the following, which doesn't make any temporary lists a[0::2], a[1::2] like the above.
from itertools import izip
i = iter(a)         #iter is so easy to use. every time I call the iter object, it will move to the next value.
b = dict(izip(i, i))
b
#or you could use:
b = {a[i]: a[i+1] for i in range(0, len(a), 2)}
#or:
i = iter(a)
b = dict(zip(i, i))




