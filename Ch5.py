__author__ = 'aeoluseros'

#Getting Started with pandas
#Throughout the rest of the book, I use the following import conventions for pandas:
from pandas import Series, DataFrame
import pandas as pd
import numpy as np
#Thus, whenever you see pd. in code, it’s referring to pandas. Series and DataFrame are
#used so much that I find it easier to import them into the local namespace.

#1. Introduction to pandas Data Structures
#(1) Series
#A Series is a one-dimensional array-like object containing an array of data (of any
#NumPy data type) and an associated array of data labels, called its index.
obj = Series([4, 7, -5, 3])
obj
obj.values
obj.index

# A Series’s index can be altered in place by assignment:
obj.index = ['Bob', 'Steve', 'Jeff', 'Ryan']
obj

#Often it will be desirable to create a Series with an index identifying each data point:
obj2 = Series([4, 7, -5, 3], index=['d', 'b', 'a', 'c'])
obj2
obj2['a']
obj2['d'] = 6
obj2[['c', 'a', 'd']]   #use [] if I want to subset more than one elements.
obj2
obj2[obj2 > 0]
obj2 * 2
np.exp(obj2)
'b' in obj2
'e' in obj2
6 in obj2  #False

#Another way to think about a Series is as a fixed-length, ordered dict, as it is a mapping
#of index values to data values.
#Should you have data contained in a Python dict, you can create a Series from it by
#passing the dict. When only passing a dict, the index in the resulting Series will have
#the dict’s keys in sorted order.
sdata = {'Ohio': 35000, 'Texas': 71000, 'Oregon': 16000, 'Utah': 5000}
obj3 = Series(sdata)
obj3

# In the following case, 3 values found in sdata were placed in the appropriate locations, but since no value for
# 'California' was found, it appears as NaN (not a number) which is considered in pandas to mark missing or NA values.
states = ['California', 'Ohio', 'Oregon', 'Texas']
obj4 = Series(sdata, index=states)
obj4

#The isnull and notnull functions in pandas should be used to detect missing data
pd.isnull(obj4)
pd.notnull(obj4)
#Series also has these as instance methods:
obj4.isnull()

#Series automatically aligns differently indexed data in arithmetic operations
obj3
obj4
obj3 + obj4

#Both the Series object itself and its index have a name attribute, which integrates with
#other key areas of pandas functionality:
obj4.name = 'population
obj4.index.name = 'state'
obj4


#(2) DataFrame
# The DataFrame has both a row and column index; it can be thought of as a dict of Series. Compared with other
# such DataFrame-like structures you may have used before (like R’s data.frame), roworiented and column-oriented
# operations in DataFrame are treated roughly symmetrically.

# There are numerous ways to construct a DataFrame, though one of the most common is from a dict of equal-length
# lists or NumPy arrays.
data = {'state': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada'],
        'year': [2000, 2001, 2002, 2001, 2002],
        'pop': [1.5, 1.7, 3.6, 2.4, 2.9]}
type(data)    #dict
frame = DataFrame(data)
frame # The resulting DataFrame will have its index assigned automatically as with Series, and the columns are
      # placed in sorted order:

#If you specify a sequence of columns, the DataFrame’s columns will be exactly what you pass:
DataFrame(data, columns=['year', 'state', 'pop'])
#As with Series, if you pass a column that isn’t contained in data, it will appear with NA
#values in the result:
frame2 = DataFrame(data, columns=['year', 'state', 'pop', 'debt'], index=['one', 'two', 'three', 'four', 'five'])
frame2
frame2.columns  #column names

#A column in a DataFrame can be retrieved as a Series either by dict-like notation or by attribute:
frame2['state']
frame2.state
frame2.year
#Rows can also be retrieved by position or name by a couple of methods, such as the ix indexing field
frame2.ix['three']

#Columns can be modified by assignment.
frame2['debt'] = 16.5
frame2
frame2['debt'] = np.arange(5.)
frame2

#When assigning lists or arrays to a column, the value’s length must match the length
#of the DataFrame. If you assign a Series, it will be instead conformed exactly to the
#DataFrame’s index, inserting missing values in any holes:
val = Series([-1.2, -1.5, -1.7], index=['two', 'four', 'five'])
frame2['debt'] = val
frame2

#Assigning a column that doesn’t exist will create a new column.
frame2['eastern'] = frame2.state == 'Ohio'
#The del keyword will delete columns as with a dict:
del frame2['eastern']
frame2.columns
frame2
###!!! The column returned when indexing a DataFrame is a view on the underlying data, not a copy.
#The column can be explicitly copied using the Series’s copy method.

#Another common form of data is a nested dict of dicts format:
pop = {'Nevada': {2001: 2.4, 2002: 2.9}, 'Ohio': {2000: 1.5, 2001: 1.7, 2002: 3.6}}
frame3 = DataFrame(pop)
frame3
#Of course you can always transpose the result
frame3.T
#The keys in the inner dicts are unioned (full join) and sorted to form the index in the result. This
#isn’t true if an explicit index is specified:
DataFrame(pop, index=[2001, 2002, 2003])

#Dicts of Series are treated much in the same way:
pdata = {'Ohio': frame3['Ohio'][:-1],'Nevada': frame3['Nevada'][:2]}
type(pdata)       #dict
DataFrame(pdata)   #columns will be automatically arranged in sorted order. so Nevada will be in the first column

#DataFrame’s index and columns could have their name attributes set
frame3.index.name = 'year'; frame3.columns.name = 'state'
frame3

#Like Series, the values attribute returns the data contained in the DataFrame as a 2D ndarray:
frame3.values
frame3.index
frame3.columns

###Possible data inputs to DataFrame constructor
#2D ndarray: A matrix of data, passing optional row and column labels
#dict of arrays, lists, or tuples: Each sequence becomes a column in the DataFrame. All sequences must be the same length.
#NumPy structured/record array: Treated as the “dict of arrays” case dict of Series: Each value becomes a column.
     # Indexes from each Series are unioned together to form the result’s row index if no explicit index is passed.
#dict of dicts: Each inner dict becomes a column. Keys are unioned to form the row index as in the “dict of Series” case.
#list of dicts or Series: Each item becomes a row in the DataFrame. Union of dict keys or Series indexes become the
    # DataFrame’s column labels
#List of lists or tuples: Treated as the “2D ndarray” case
#Another DataFrame: The DataFrame’s indexes are used unless different ones are passed
#NumPy MaskedArray: Like the “2D ndarray” case except masked values become NA/missing in the DataFrame result


#(3)Index Objects
obj = Series(range(3), index=['a', 'b', 'c'])
index = obj.index
index
index[1:]

#Index objects are immutable and thus can’t be modified by the user:
index[1] = 'd'   #TypeError
#Immutability is important so that Index objects can be safely shared among data structures:
index = pd.Index(np.arange(3))
type(index)        #pandas.core.index.Int64Index
obj2 = Series([1.5, -2.5, 0], index=index)
obj2.index is index

#Main Index objects in pandas
# Index: The most general Index object, representing axis labels in a NumPy array of Python objects.
# Int64Index: Specialized Index for integer values.
# MultiIndex: “Hierarchical” index object representing multiple levels of indexing on a single axis. Can be thought of
            # as similar to an array of tuples.
# DatetimeIndex: Stores nanosecond timestamps (represented using NumPy’s datetime64 dtype).
# PeriodIndex: Specialized Index for Period data (timespans).

#Each Index has a number of methods and properties for set logic
frame3
'Ohio' in frame3.columns
2003 in frame3.index
# append: Concatenate with additional Index objects, producing a new Index
# diff: Compute set difference as an Index
# intersection: Compute set intersection
# union: Compute set union
# isin: Compute boolean array indicating whether each value is contained in the passed collection
# delete: Compute new Index with element at index i deleted
# drop: Compute new index by deleting passed values
# insert: Compute new Index by inserting element at index i
# is_monotonic: Returns True if each element is greater than or equal to the previous element
# is_unique: Returns True if the Index has no duplicate values
# unique: Compute the array of unique values in the Index


#2. Essential Functionality
#(1) Reindexing
#Series
obj = Series([4.5, 7.2, -5.3, 3.6], index=['d', 'b', 'a', 'c'])
obj
obj2 = obj.reindex(['a', 'b', 'c', 'd', 'e'])
obj.reindex(['a', 'b', 'c', 'd', 'e'], fill_value=0)

# For ordered data like time series, it may be desirable to do some interpolation or filling
# of values when reindexing. The method option allows us to do this, using a method such
# as ffill which forward fills the values:
obj3 = Series(['blue', 'purple', 'yellow'], index=[0, 2, 4])
obj3.reindex(range(6), method='ffill')
    #ffill or pad Fill (or carry) values forward
    #bfill or backfill Fill (or carry) values backward

#Data Frame.
#With DataFrame, reindex can alter either the (row) index, columns, or both.
frame = DataFrame(np.arange(9).reshape((3, 3)), index=['a','c','d'], columns=['Ohio', 'Texas', 'California'])
frame
frame2 = frame.reindex(index=['a', 'b', 'c', 'd'])
frame2
states = ['Texas', 'Utah', 'California']
frame.reindex(columns=states)

frame2 = frame.reindex(index=['a', 'b', 'c', 'd'], method='ffill', columns=['Texas', 'Utah', 'California'])
frame2
#reindexing can be done more succinctly by label-indexing with ix:
frame.ix[['a', 'b', 'c', 'd'], states]

##Relatedly, when reindexing a Series or DataFrame, you can also specify a different fill value:
frame.reindex(columns=states, fill_value=0)

#reindex function arguments
# index: New sequence to use as index. Can be Index instance or any other sequence-like Python data structure. An
        # Index will be used exactly as is without any copying
# method: Interpolation (fill) method, see Table 5-4 for options.
# fill_value: Substitute value to use when introducing missing data by reindexing
# limit: When forward- or backfilling, maximum size gap to fill
# level: Match simple Index on level of MultiIndex, otherwise select subset of
# copy: Do not copy underlying data if new index is equivalent to old index. True by default (i.e. always copy data).


#(2) Dropping entries from an axis  -- create a copy
obj = Series(np.arange(5.), index=['a', 'b', 'c', 'd', 'e'])
new_obj = obj.drop('c')
new_obj
obj.drop(['d', 'c'])

# drop(self, labels, axis=0, level=None, inplace=False, **kwargs) unbound pandas.core.frame.DataFrame method
#     Return new object with labels in requested axis removed
#
#     Parameters
#     ----------
#     labels : single label or list-like
#     axis : int or axis name
#     level : int or name, default None
#         For MultiIndex
#     inplace : bool, default False
#         If True, do operation inplace and return None.

#With DataFrame, index values can be deleted from either axis:
data = DataFrame(np.arange(16).reshape((4, 4)), index=['Ohio', 'Colorado', 'Utah', 'New York'], columns=['one', 'two', 'three', 'four'])
data.drop(['Colorado', 'Ohio'])   #it's a copy
data.drop('two', axis=1)     #to delete rows, we need to specify axis=1 !!!!
data.drop(['two', 'four'], axis=1)

#(3) Indexing, selection, and filtering
#subsetting of Series
obj = Series(np.arange(4.), index=['a', 'b', 'c', 'd'])
obj['b']
obj[1] == obj['b']
obj[2:4]
obj[['b', 'a', 'd']]
obj[[1, 3]]
obj[obj < 2]
obj['b':'c']
obj['b':'c'] = 5
obj

#column subsetting of DataFrame
data = DataFrame(np.arange(16).reshape((4, 4)), index=['Ohio', 'Colorado', 'Utah', 'New York'], columns=['one', 'two', 'three', 'four'])
data
data['two']
data[['three', 'one']]
data[:2]

#indexing with a boolean DataFrame -- This is intended to make DataFrame syntactically more like an ndarray in this case.
data[data['three'] > 5]
data[data < 5] = 0
data

#use ix to subset both column and row
data.ix['Colorado', ['two', 'three']]
data.ix[['Colorado', 'Utah'], [3, 0, 1]]
data.ix[2]
type(data.ix[2])   #pandas.core.series.Series
data.xs('Colorado')
data.ix[:'Utah', 'two']
data.ix[data.three > 5, :3]
#also ix could be used to do reindexing
#When designing pandas, I felt that having to type frame[:, col] to select
#a column was too verbose (and error-prone), since column selection is
#one of the most common operations. Thus I made the design trade-off
#to push all of the rich label-indexing into ix.

#Indexing options with DataFrame
# obj[val]: Select single column or sequence of columns from the DataFrame. Special case conveniences: boolean
          # array (filter rows), slice (slice rows), or boolean DataFrame (set values based on some criterion).
# obj.ix[val]: Selects single row of subset of rows from the DataFrame.
# obj.ix[:, val]: Selects single column of subset of columns.
# obj.ix[val1, val2]: Select both rows and columns.
# reindex method: Conform one or more axes to new indexes.
# xs method: Select single row or column as a Series by label.
# icol, irow methods: Select single column or row, respectively, as a Series by integer location.
data.icol(2)
data.irow(2)
# get_value, set_value methods: Select single value by row and column label.
data
data.get_value('Ohio', 'two')


#(4) Arithmetic and data alignment
# When adding together objects, if any index pairs are not the same, the respective index in the result
# will be the union of the index pairs.
s1 = Series([7.3, -2.5, 3.4, 1.5], index=['a', 'c', 'd', 'e'])
s2 = Series([-2.1, 3.6, -1.5, 4, 3.1], index=['a', 'c', 'e', 'f', 'g'])
s1
s2
s1 + s2

#In the case of DataFrame, alignment is performed on both the rows and the columns:
df1 = DataFrame(np.arange(9.).reshape((3, 3)), columns=list('bcd'), index=['Ohio', 'Texas', 'Colorado'])
df2 = DataFrame(np.arange(12.).reshape((4, 3)), columns=list('bde'), index=['Utah', 'Ohio', 'Texas', 'Oregon'])
df1
df2
df1 + df2

#Arithmetic methods with fill values
df1 = DataFrame(np.arange(12.).reshape((3, 4)), columns=list('abcd'))
df2 = DataFrame(np.arange(20.).reshape((4, 5)), columns=list('abcde'))
df1
df2
df1 + df2
#Using the add method on df1, I pass df2 and an argument to fill_value:
df1.add(df2, fill_value=0)

#Flexible arithmetic methods
    # add Method for addition (+)
    # sub Method for subtraction (-)
df1.sub(df2, fill_value=0)
    # div Method for division (/)
df1.div(df2, fill_value=10)
    # mul Method for multiplication (*)
df1.mul(df2, fill_value=10)


##Operations between DataFrame and Series
# Operatons between different-dimension ndarrays
arr = np.arange(12.).reshape((3, 4))
arr   #numpy.ndarray
arr[0]         #array([ 0.,  1.,  2.,  3.]), numpy.ndarray
arr - arr[0]  #broadcasting
# array([[ 0.,  0.,  0.,  0.],
#       [ 4.,  4.,  4.,  4.],
#       [ 8.,  8.,  8.,  8.]])

#Operations between a DataFrame and a Series are similar:
frame = DataFrame(np.arange(12.).reshape((4, 3)), columns=list('bde'),index=['Utah', 'Ohio', 'Texas', 'Oregon'])
series = frame.ix[0]
frame
series
frame - series #By default, arithmetic between DataFrame and Series matches the index of the Series
               #on the DataFrame's columns, broadcasting down the rows.
#If an index value is not found in either the DataFrame’s columns or the Series’s index,
#the objects will be reindexed to form the union:
series2 = Series(range(3), index=['b', 'e', 'f'])
frame + series2

###If you want to instead broadcast over the columns, matching on the rows, you have to
#use one of the arithmetic methods.
series3 = frame['d']
frame
series3
frame.sub(series3, axis=0)   #use axis = 0
frame.sub(series3, axis=1)   #will get a lot of NaN


#(5) Function application and mapping
#NumPy ufuncs (element-wise array methods) work fine with pandas objects:
frame = DataFrame(np.random.randn(4, 3), columns=list('bde'), index=['Utah', 'Ohio', 'Texas', 'Oregon'])
frame
np.abs(frame)

###Another frequent operation is applying a function on 1D arrays to each column or row.
#DataFrame’s apply method does exactly this:
f = lambda x: x.max() - x.min()
frame.apply(f)           #apply on each column (row-wise)
frame.apply(f, axis=1)   #apply on each row (column-wise)      #same as apply in R!

#The function passed to apply need not return a scalar value, it can also return a Series with multiple values:
def f(x):
    return Series([x.min(), x.max()], index=['min', 'max'])
frame.apply(f)    #each column will generate a series.

# Element-wise functions:
# Series has a map method for applying an element-wise function:
frame['e'].map(format)
# Data Frame has applymap:
format = lambda x: '%.2f' % x
frame.applymap(format)


#(6) Sorting and ranking
#(a) Sort by Index:
# To sort lexicographically by row or column index, use the sort_index method, which returns
# a new, sorted object:
obj = Series(range(4), index=['d', 'a', 'b', 'c'])
obj.sort_index()

#With a DataFrame, you can sort by index on either axis:
frame = DataFrame(np.arange(8).reshape((2, 4)), index=['three', 'one'], columns=['d', 'a', 'b', 'c'])
frame
frame.sort_index()
frame.sort_index(axis=1)
frame.sort_index(axis=1, ascending=False)

#(b) Sort by Value:
#Series:
#To sort a Series by its values, use its order method:
obj = Series([4, 7, -3, 2])
obj.order()
#Any missing values are sorted to the end of the Series by default:
obj = Series([4, np.nan, 7, np.nan, -3, 2])
obj.order()

# On DataFrame, you may want to sort by the values in one or more columns. To do so,
# pass one or more column names to the 'by' option:
frame = DataFrame({'b': [4, 7, -3, 2], 'a': [0, 1, 0, 1]})
frame
frame.sort_index(by='b')
#To sort by multiple columns, pass a list of names:
frame.sort_index(by=['a', 'b'])


# Ranking is closely related to sorting, assigning ranks from one through the number of
# valid data points in an array. It is similar to the indirect sort indices produced by
# numpy.argsort.
#numpy.argsort:
#One dimensional array:
x = np.array([3, 1, 2])
x
np.argsort(x)   #array([1, 2, 0]) --> an sort index produced

#Two-dimensional array:
x = np.array([[0, 3], [2, 2]])
x      #array([[0, 3], [2, 2]])
np.argsort(x, axis=0)   #array([[0, 1], [1, 0]])
np.argsort(x, axis=1)   #array([[0, 1], [0, 1]])

#Sorting with keys:
x = np.array([(1, 0), (0, 1)], dtype=[('x', '<i4'), ('y', '<i4')])
x
np.argsort(x, order=('x','y'))   #array([1, 0])
np.argsort(x, order=('y','x'))   #array([0, 1])

# The rank methods for Series and DataFrame are the place to look; by default
obj = Series([7, -5, 7, 4, 2, 0, 4, 4])
obj.rank()  #by default, rank breaks ties by assigning each group the mean rank.

#Ranks can also be assigned according to the order they’re observed in the data:
obj.rank(method='first')
#Naturally, you can rank in descending order, too:
obj.rank(ascending=False, method='max')

# a list of tie-breaking methods available:
# 'average' Default: assign the average rank to each entry in the equal group.
# 'min' Use the minimum rank for the whole group.
# 'max' Use the maximum rank for the whole group.
# 'first' Assign ranks in the order the values appear in the data.


#DataFrame can compute ranks over the rows or the columns:
frame = DataFrame({'b': [4.3, 7, -3, 2], 'a': [0, 1, 0, 1], 'c': [-2, 5, 8, -2.5]})
frame
frame.rank(axis=1)


#(7)Axis indexes with duplicate values
# Many pandas functions (like reindex) require that the labels be unique
obj = Series(range(5), index=['a', 'a', 'b', 'b', 'c'])
obj
#The index’s is_unique property can tell you whether its values are unique or not:
obj.index.is_unique

# Data selection is one of the main things that behaves differently with duplicates.
# Indexing a value with multiple entries returns a Series while single entries return a scalar value:
obj['a']
type(obj['a'])   #pandas.core.series.Series
obj['c']
type(obj['c'])   #numpy.int64
#The same logic extends to indexing rows in a DataFrame:
df = DataFrame(np.random.randn(4, 3), index=['a', 'a', 'b', 'b'])
df.ix['b']


#3. Summarizing and Computing Descriptive Statistics
#(1) reductions or summary statistics:
df = DataFrame([[1.4, np.nan], [7.1, -4.5], [np.nan, np.nan], [0.75, -1.3]], index=['a', 'b', 'c', 'd'], columns=['one', 'two'])
df  ###NaN should use np.nan

#Calling DataFrame’s sum method returns a Series containing column sums:
df.sum()
#Passing axis=1 sums over the rows instead:
df.sum(axis=1)   #skipna: Exclude missing values, True by default.

#Options for reduction methods
# axis Axis to reduce over. 0 for DataFrame’s rows and 1 for columns.
# skipna Exclude missing values, True by default.
# level int, default None. Reduce grouped by level if the axis is hierarchically-indexed (MultiIndex).

#(2) Other methods are accumulations:
df.cumsum()

#(3) Another type of method is neither a reduction nor an accumulation. describe is one
# such example, producing multiple summary statistics in one shot:
df.describe()    #summary in R
#On non-numeric data, describe produces alternate summary statistics:
obj = Series(['a', 'a', 'b', 'c'] * 4)
obj.describe()

df.quantile(q=0.25)    # default 0.5 (50% quantile)


###Descriptive and summary statistics
# count: Number of non-NA values
# describe: Compute set of summary statistics for Series or each DataFrame column
# min, max: Compute minimum and maximum values
# argmin, argmax: Compute index locations (integers) at which minimum or maximum value obtained, respectively
# idxmin, idxmax: Compute index values at which minimum or maximum value obtained, respectively
# quantile: Compute sample quantile ranging from 0 to 1
# sum: Sum of values
# mean: Mean of values
# median: Arithmetic median (50% quantile) of values
# mad: Mean absolute deviation from mean value
# var: Sample variance of values
# std: Sample standard deviation of values
# skew: Sample skewness (3rd moment) of values
# kurt: Sample kurtosis (4th moment) of values
# cumsum: Cumulative sum of values
# cummin, cummax: Cumulative minimum or maximum of values, respectively
# cumprod: Cumulative product of values
# diff: Compute 1st arithmetic difference (useful for time series)
# pct_change: Compute percent changes

frame = DataFrame(np.random.randn(20, 6))
frame
frame.std(1)
frame.std(0)  #defaul



#(4) Correlation and Covariance
#Let’s consider some DataFrames of stock prices and volumes obtained from Yahoo! Finance:
import pandas.io.data as web
all_data = {}
for ticker in ['AAPL', 'IBM', 'MSFT', 'GOOGL']:
    all_data[ticker] = web.get_data_yahoo(ticker, '1/1/2003', '5/2/2015')


all_data['GOOGL'].head()

price = DataFrame({tic: data['Adj Close'] for tic, data in all_data.iteritems()})
price                        #D.iteritems() -> an iterator over the (key, value) items of D
volume = DataFrame({tic: data['Volume'] for tic, data in all_data.iteritems()})

returns = price.pct_change()
type(returns)  #pandas.core.frame.DataFrame
returns.tail()

returns.MSFT.corr(returns.IBM)
returns.MSFT.cov(returns.IBM)

#DataFrame’s corr and cov methods, on the other hand, return a full correlation or covariance matrix as a DataFrame
returns.cov()
returns.corr()

# Using DataFrame’s corrwith method, you can compute pairwise correlations between a DataFrame’s columns or rows with
# another Series or DataFrame. Passing a Series returns a Series with the correlation value computed for each column:
returns.corrwith(returns.IBM)

# Passing a DataFrame computes the correlations of matching column names. Here I compute correlations of percent
# changes with volume:
returns.corrwith(volume)

# Passing axis=1 does things row-wise instead. In all cases, the data points are aligned by
# label before computing the correlation.


#4. Unique Values, Value Counts, and Membership
obj = Series(['c', 'a', 'd', 'a', 'a', 'b', 'b', 'c', 'c'])
uniques = obj.unique()
uniques
type(uniques)   #numpy.ndarray
#The unique values are not necessarily returned in sorted order, but could be sorted after
#the fact if needed (uniques.sort()).
uniques.sort()   #in-place sort
uniques
#sort(self, axis=0, kind='quicksort', order=None, ascending=True) unbound pandas.core.series.Series method
    # Sort values and index labels by value, in place. For compatibility with
    # ndarray API. No return value
    #
    # Parameters
    # ----------
    # axis : int (can only be zero)
    # kind : {'mergesort', 'quicksort', 'heapsort'}, default 'quicksort'
    #     Choice of sorting algorithm. See np.sort for more
    #     information. 'mergesort' is the only stable algorithm
    # order : ignored
    # ascending : boolean, default True
    #     Sort ascending. Passing False sorts descending
obj.sort()   #in-place
uniques = obj.unique()
uniques

#value_counts is also available as a top-level pandas method that can be used with any array or sequence:
obj = Series(['c', 'a', 'd', 'a', 'a', 'b', 'b', 'c', 'c'])
obj
obj.value_counts()
type(obj.value_counts())        #pandas.core.series.Series
pd.value_counts(obj.values, sort=False)

# value_counts(values, sort=True, ascending=False, normalize=False, bins=None)
#     Compute a histogram of the counts of non-null values
#     values : ndarray (1-d)
#     sort : boolean, default True, Sort by values
#     ascending : boolean, default False, Sort in ascending order
#     normalize: boolean, default False, if True then compute a relative histogram
#     bins : integer, optional, Rather than count values, group them into half-open bins,
#         convenience for pd.cut, only works with numeric data

#isin is equivalent to %in% in R
mask = obj.isin(['b', 'c'])
mask
obj[mask]


#In some cases, you may want to compute a histogram on multiple related columns in a DataFrame.
data = DataFrame({'Qu1': [1, 3, 4, 3, 4],
                  'Qu2': [2, 3, 1, 2, 3],
                  'Qu3': [1, 5, 2, 4, 4]})
data
result = data.apply(pd.value_counts).fillna(0)


#4. Handling Missing Data
# pandas uses the floating point value NaN (Not a Number) to represent missing data in
# both floating as well as in non-floating point arrays. It is just used as a sentinel that can
# be easily detected:
string_data = Series(['aardvark', 'artichoke', np.nan, 'avocado'])
string_data
string_data.isnull()
string_data.notnull()
#The built-in Python None value is also treated as NA in object arrays:
string_data[0] = None
string_data.isnull()

# dropna: Filter axis labels based on whether values for each label have missing data, with varying thresholds for how much
        # missing data to tolerate.
# fillna: Fill in missing data with some value or using an interpolation method such as 'ffill' or 'bfill'.
# isnull: Return like-type object containing boolean values indicating which values are missing / NA.
# notnull: Negation of isnull.

#(1) Filtering Out Missing Data
from numpy import nan as NA
#Series
data = Series([1, NA, 3.5, NA, 7])
data.dropna()
data = Series([1, NA, 3.5, NA, 7])
data[data.notnull()]

# With DataFrame objects, these are a bit more complex.
# dropna is a copy action
data = DataFrame([[1., 6.5, 3.], [1., NA, NA], [NA, NA, NA], [NA, 6.5, 3.]])
data
#dropna by default drops any row containing a missing value:
cleaned = data.dropna()
cleaned

#Passing how='all' will only drop rows that are all NA:
data.dropna(how='all')
#Dropping columns in the same way is only a matter of passing axis=1:
data[4] = NA
data
data.dropna(axis=1, how='all')


#A related way to filter out DataFrame rows tends to concern time series data.
#Suppose you want to keep only rows containing a certain number of observations.
df = DataFrame(np.random.randn(7, 3))
df
df.ix[:4, 1] = NA; df.ix[:2, 2] = NA
df
df.dropna(thresh=3)
df.dropna(thresh=2)
df.dropna(thresh=1)

#Filling in Missing Data
df.fillna(0)
df.fillna({1: 0.5, 2: -1})

#fillna returns a new object, but you can modify the existing object in place:
_ = df.fillna(0, inplace=True)  #always returns a reference to the filled object
df
_
df = DataFrame(np.random.randn(6, 3))
df.ix[2:, 1] = NA; df.ix[4:, 2] = NA
df
df.fillna(method='ffill')
df.fillna(method='ffill', limit=2)  #only fill in 2 values at a time.

#With fillna you can do lots of other things with a little creativity.
data = Series([1., NA, 3.5, NA, 7])
data.fillna(data.mean())

#fillna function arguments
    # value: Scalar value or dict-like object to use to fill missing values
    # method: Interpolation, by default 'ffill' if function called with no other arguments
    # axis: Axis to fill on, default axis=0
    # inplace: Modify the calling object without producing a copy
    # limit: For forward and backward filling, maximum number of consecutive periods to fill


#5. Hierarchical Indexing
#(1) Series
data = Series(np.random.randn(10), index=[['a', 'a', 'a', 'b', 'b', 'b', 'c', 'c', 'd', 'd'], [1, 2, 3, 1, 2, 3, 1, 2, 2, 3]])
data
data.index
data.values

#With a hierarchically-indexed object, so-called partial indexing is possible, enabling
#you to concisely select subsets of the data:
data['b']
data['b':'c']
data.ix[['b', 'd']]
#Selection is even possible in some cases from an “inner” level:
data[:, 2]


#Hierarchical indexing plays a critical role in reshaping data and group-based operations
#like forming a pivot table. For example, this data could be rearranged into a DataFrame
#using its unstack method:
data.unstack()
# The inverse operation of unstack is stack
data.unstack().stack()

#(2) With a DataFrame, either axis can have a hierarchical index:
frame = DataFrame(np.arange(12).reshape((4, 3)), index=[['a', 'a', 'b', 'b'], [1, 2, 1, 2]],
                 columns=[['Ohio', 'Ohio', 'Colorado'], ['Green', 'Red', 'Green']])
frame
frame.stack()
frame.index.names = ['key1', 'key2']
frame.columns.names = ['state', 'color']
frame
frame['Ohio']

#A MultiIndex can be created by itself and then reused:
idx = pd.MultiIndex.from_arrays([['a', 'a', 'b', 'b'], [1, 2, 1, 2]], names=['key1', 'key2'])
colname = pd.MultiIndex.from_arrays([['Ohio', 'Ohio', 'Colorado'], ['Green', 'Red', 'Green']], names=['state', 'color'])
frame = DataFrame(np.arange(12).reshape((4, 3)), index=idx, columns=colname)
frame


#(3) Reordering and Sorting Levels
#At times you will need to rearrange the order of the levels on an axis or sort the data
#by the values in one specific level.

# The swaplevel takes two level numbers or names and returns a new object with the levels
# interchanged (but the data is otherwise unaltered):
frame.swaplevel('key1', 'key2')

#sortlevel, on the other hand, sorts the data (stably) using only the values in a single level.
#When swapping levels, it’s not uncommon to also use sortlevel so that the result is lexicographically sorted
frame
frame.sortlevel(1)
frame.swaplevel(0, 1).sortlevel(0)   #to make sure lexicographically sorted starting with the outermost level
frame.swaplevel(0, 1).sort_index()

# Data selection performance is much better on hierarchically indexed objects if the index is lexicographically
# sorted starting with the outermost level, that is, the result of calling sortlevel(0) or sort_index().

#(4) Summary Statistics by Level
frame.sum()
frame.sum(level='key2')
frame.sum(level='color', axis=1)   #add columns
#Under the hood, this utilizes pandas’s groupby machinery which will be discussed in more detail later in the book.


#(5) Using a DataFrame’s Columns
frame = DataFrame({'a': range(7), 'b': range(7, 0, -1), 'c': ['one', 'one', 'one', 'two', 'two', 'two', 'two'], 'd': [0, 1, 2, 0, 1, 2, 3]})
frame
#DataFrame’s set_index function will create a new DataFrame using "one or more of its columns as the index":
frame2 = frame.set_index(['c', 'd'])   #making 'c' and 'd' as the index!
frame2       #By default the columns are removed from the DataFrame, though you can leave them in

#reset_index, on the other hand, does the opposite of set_index; the hierarchical index levels are are moved into the columns:
frame2.reset_index()


#6.Other pandas Topics
#(1) Integer Indexing
# Working with pandas objects indexed by integers is something that often trips up new
# users due to some differences with indexing semantics on built-in Python data structures
# like lists and tuples. For example, you would not expect the following code to generate an error:
ser = Series(np.arange(3.))
ser[-1]  #error, beccause there may be an index called -1. To avoid ambiguity, we shouldn't use -1 unless the index is non-integer.

# In this case, pandas could “fall back” on integer indexing, but there’s not a safe and
# general way (that I know of) to do this without introducing subtle bugs.
ser
#On the other hand, with a non-integer index, there is no potential for ambiguity:
ser2 = Series(np.arange(3.), index=['a', 'b', 'c'])
ser2[-1]
#To keep things consistent, if you have an axis index containing indexers, data selection
#with integers will always be label-oriented. This includes slicing with ix, too.
idx = np.arange(6)
np.random.seed(123)
np.random.shuffle(idx)
ser = Series(np.arange(3.0*2)[::-1], index =idx)
ser.ix[:1]    #read until the index labeled 1
ser[:2] == ser.ix[:1]


#In cases where you need reliable position-based indexing regardless of the index type,
#you can use the iget_value method from Series
ser3 = Series(range(3), index=[-5, 1, 3])
ser3.iget_value(2)

# irow and icol methods from DataFrame
frame = DataFrame(np.arange(6).reshape(3, 2), index=[2, 0, 1])
frame
frame.irow(0)
frame.icol(0)


#7. Panel Data - you can think of as a three-dimensional analogue of DataFrame
import pandas.io.data as web
pdata = pd.Panel(dict((stk, web.get_data_yahoo(stk, '1/1/2009', '6/1/2014')) for stk in ['AAPL', 'GOOGL', 'MSFT', 'DELL']))
pdata
pdata = pd.Panel({stk: web.get_data_yahoo(stk, '1/1/2009', '6/1/2014') for stk in ['AAPL', 'GOOGL', 'MSFT', 'DELL']})
pdata   #the same as above
#<class 'pandas.core.panel.Panel'>
    # Dimensions: 4 (items) x 1381 (major_axis) x 6 (minor_axis)
    # Items axis: AAPL to MSFT
    # Major_axis axis: 2009-01-02 00:00:00 to 2014-05-30 00:00:00
    # Minor_axis axis: Open to Adj Close
pdata = pdata.swapaxes('items', 'minor')    #items is the first-level catogery!!!!!!!!!!!!!!!!!!!!
#swapaxes(self, axis1, axis2, copy=True) unbound pandas.core.panel.Panel method
    # Interchange axes and swap values axes appropriately
    # Returns:  y : same as input
pdata
    # <class 'pandas.core.panel.Panel'>
    # Dimensions: 6 (items) x 1381 (major_axis) x 4 (minor_axis)
    # Items axis: Open to Adj Close
    # Major_axis axis: 2009-01-02 00:00:00 to 2014-05-30 00:00:00
    # Minor_axis axis: AAPL to MSFT
pdata['Adj Close']

#ix-based label indexing generalizes to three dimensions, so we can select all data at a
#particular date or a range of dates like so:
pdata.ix[:, '6/1/2012', :]
pdata.ix['Adj Close', '5/22/2012':, :]
#see this interesting guy:
pdata = pdata.swapaxes('items', 'minor')
pdata.ix[:, '6/1/2012', :]
# swapback:
pdata = pdata.swapaxes('items', 'minor')

type(pdata.ix[:, '6/1/2012', :])  #pandas.core.frame.DataFrame
#An alternate way to represent panel data, especially for fitting statistical models, is in "stacked" DataFrame form:
stacked = pdata.ix[:, '5/30/2012':, :].to_frame()
stacked
#DataFrame has a related to_panel method, the inverse of to_frame:
stacked.to_panel()