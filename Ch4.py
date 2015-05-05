__author__ = 'aeoluseros'

import numpy as np

#whenever you see 'array', 'NumPy array', or 'ndarray' in the text with
#few exceptions, they all refer to the same thing: the ndarray object

#1. Create ndarrays
#(1) use np.array; np.asarray
data1 = [6, 7.5, 8, 0, 1]
arr1 = np.array(data1)
arr1
type(arr1)   #numpy.ndarray

data2 = [[1, 2, 3, 4], [5, 6, 7, 8]]
arr2 = np.array(data2)
arr2

data3 = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
arr3 = np.asarray(data3)
arr3

#array copies the input data by default
#asarray do not copy if the input is already an ndarry

arr2 * 10
arr2 + arr2
arr2.shape  #dimension
arr2.ndim   #2-dimension array

arr1.dtype   #dtype('float64')
arr2.dtype   #dtype('int32')

#(2) other functions like zeros, ones, empty...
#pass a tuple
np.zeros(10)
np.zeros((3, 6))
np.zeros((3, 6, 2))

np.ones((3, 6, 2))

np.empty((2, 3, 2))   #create an array without initializing its values to any particular value.

np.arange(15)     #like the built-in range, but returns an ndarray instead of a list.

np.zeros_like(arr2)   #takes another array/tuple and produces a one array of the same shape and dtyple
np.ones_like(arr2)    #takes another array/tuple and produces a zero array of the same shape and dtyple
np.empty_like(arr2)

np.eye(5)        #identity matrix
np.identity(5)   #same as above


#(3) set the data type of ndarrays
arr1 = np.array([1, 2, 3], dtype=np.float64)   #64 bits or 8 bytes
arr1
arr1 = np.array([1, 2, 3], dtype=np.int32)
arr1

#NumPy data types:
    #int8/uint8 -- int64/uint64:  shorthand code: i1/u1 -- i4/u4
    #float16(f2) -- float 128:   shorthand code: f16 is f4 or f; float64 is f8 or d; float128 is f16 or g;
    #complex64, complex128, complex256:   shorthand code: c8, c16, c32
    #bool:  shorthand code: ?
    #object
    #string:   shorthand code: S
    #unicode:   shorthand code: U_

empty_uint32 = np.empty(8, dtype='u4')
empty_uint32

arr = np.array([1, 2, 3, 4, 5])
arr.dtype
float_arr = arr.astype(np.float64)
float_arr.dtype
arr.astype(np.int32)

#convert an array of strings representing numbers to numeric form
numeric_strings = np.array(['1.25', '-9.6', '42'])
numeric_strings.astype(float)
###!!! calling astype is always creating a new array, even if the new
#dtype is the same as the old dtype.


#2. Operations between Arrays and Scalars --  vectorization
arr = np.array([[1., 2., 3.], [4., 5., 6.]])
arr

#(1) element-wise operation
arr * arr
arr - arr
1/arr
arr ** 0.5
#Operations between different sized arrays is called broadcasting. (not necessary for most of this book)


#(2) Basic indexing and slicing
arr = np.arange(10)
arr
arr[5]
arr[5:8]

###array slices are views on the original array. This means that the data is not copied,
#and any modifications to the view will be reflected in the source array.
arr_slice = arr[5:8]  #arr_slice is just a view or the arr[5:8]
                      #threfore, modifying arr_slice will also modify the original data
arr_slice[1] = 12345
arr

arr_slice[:] = 64
arr
# As NumPy has been designed with large data use cases in mind, you could imagine performance
# and memory problems if NumPy insisted on copying data left and right.

#!!!if you want to a copy of a slice of an ndarray instead of a view you will need to explicitly
# copy the array; for example, arr[5:8].copy

data1 = [6, 7.5, 8, 0, 1]
data1_slice = data1[2:4]   #this is a copy instead of a reference.
data1_slice = 64
data1     #for list, it will not be changed


#with higher dimensions:
arr2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])  #2D array
arr2d
arr2d[2]
arr2d[0][2]
arr2d[0, 2]   #same as above
arr2d[2, 2]   #9

arr3d = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
print arr3d
arr3d.shape
arr3d[0].shape
arr3d[0]

old_values = arr3d[0].copy()
type(old_values)   #numpy.ndarray
arr3d[0] = 42
arr3d
arr3d[0]
arr3d[0] = old_values
arr3d

arr3d[0, 1, 2]

#indexing with slices
arr[1:6]
arr2d[:2]   #arr[0] and arr[1]

arr2d[:2, 1:]


#(3) Boolean Indexing  -- always create a copy of the data, enven if the returned array is unchanged.
names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
from numpy.random import randn
data = randn(7, 4)
type(data)   #numpy.ndarray
names
data

names == 'Bob'  #Out[15]: array([ True, False, False,  True, False, False, False], dtype=bool)
data[names == 'Bob']   #the first row and fourth row

data[names == 'Bob', 3]
data[-(names == 'Bob')] == data[names != 'Bob']

mask = (names == 'Bob') | (names == 'Will')
data[mask]

#The python keyword 'and' and 'or' do not work with boolean arrays.
mask = (names == 'Bob') and (names == 'Will')
mask = (names == 'Bob') or (names == 'Will')
mask = (names == 'Bob') & (names == 'Will')

data[data < 0] = 0
data

data[names != 'Joe'] = 7

#(4) Fancy indexing --indexing using integer arrays
# to select out a subset of the rows in a particular order,
         #you can simply pass a list or ndarry of integers specifying the desired order.
arr = np.empty((8, 4))
for i in range(8):
    arr[i] = i
arr[[4, 3, 0, 6]]      #arr[[4, 3, 0, 6],] or arr[[4, 3, 0, 6], :] are also okay
arr[:, [3, 1, 0, 2]]   #select columns,  there must be in front of comma
arr
#using negative indices select rows from the end:
arr[[-3, -5, -7]]

#passing multiple index arrays select a 1D array of elemens:
arr = np.arange(32).reshape((8, 4))
arr
arr[[1, 5, 7, 2], [0, 3, 1, 2]]   #(1, 0), (5, 3), (7, 1), (2, 2)

arr[[1, 5, 7, 2]][:, [0, 3, 1, 2]]

#use the np.ix_ function --> converts two 1D integer arrays to an indexer that
#selects the square region.
arr[np.ix_([1, 5, 7, 2], [0, 3, 1, 2])]  #compare with arr[[1, 5, 7, 2], [0, 3, 1, 2]]


#(5) transposing arrays and swapping axes
arr = np.arange(15).reshape((3, 5))
arr
arr.T
arr = np.random.randn(6, 3)
np.dot(arr.T, arr)  #dot multiplication (matrix product) should us np.dot

#for higher dimensional arrays, transpose will accept a tuple of axis numbers to "permute" the axes
arr = np.arange(16).reshape((2, 2, 4))
arr
        # array([[[ 0,  1,  2,  3],
        #         [ 4,  5,  6,  7]],
        #        [[ 8,  9, 10, 11],
        #         [12, 13, 14, 15]]])
arr[0]
arr[0, 1]
arr[0][1]
arr[0, 1, 2]     #6
arr[1, 0, 2]     #10

#first dimension
arr[0, :, :]
arr[1, :, :]

#second dimension
arr[:, 0, :]
arr[:, 1, :]

#third dimension
arr[:, :, 0]
arr[:, :, 1]
arr[:, :, 2]
arr[:, :, 3]

arr.transpose((1, 0, 2))  #the 2nd dimension turns into the 1st dimension. the 1st becomes the 2nd.
arr.transpose((0, 1, 2))  #no change
arr.transpose((2, 1, 0))  #swap the third dimension and the 1st dimension

#simple transposing with .T is just a special case of swapping axes.
#ndarray has the method swapaxes which takes a pair of axis numbers.
arr.swapaxes(1, 2)   #swap the first and second dimensions. doesn't change the 3rd dimension


#3. Universal functions(ufunc): Fast element-wise array functions
arr = np.arange(10)

#Comparison between np.arange and built-in range
size = int(1E6)
%timeit for x in range(size): x ** 2
# out: 10 loops, best of 3: 136 ms per loop
%timeit for x in xrange(size): x ** 2
# out: 10 loops, best of 3: 88.9 ms per loop
# avoid this
%timeit for x in np.arange(size): x ** 2
#out: 1 loops, best of 3: 1.16 s per loop
# use this
%timeit np.arange(size) ** 2
#out: 100 loops, best of 3: 19.5 ms per loop

#unary functions:
np.sqrt(arr)
np.exp(arr)
#binary functions:
x = randn(8)
y = randn(8)
x
y
np.maximum(x, y)
np.add(x, y)

#while not common, a ufunc can return multiple arrays.
arr = randn(7) * 5
np.modf(arr)   #returns the fractional and integral parts of a floating point array
###modf is the vertorized version of the built-in Python "divmod" (dividend + mod)

#check the help document: example: np.info(np.copysign)

#abs, fabs: Compute the absolute value element-wise for integer, floating point, or complex values. Use fabs as a
          # faster alternative for non-complex-valued data
#sqrt: Compute the square root of each element. Equivalent to arr ** 0.5
#square: Compute the square of each element. Equivalent to arr ** 2
#exp: Compute the exponent ex of each element
#log, log10, log2, log1p : Natural logarithm (base e), log base 10, log base 2, and log(1 + x), respectively
#sign: Compute the sign of each element: 1 (positive), 0 (zero), or -1 (negative)
#ceil: Compute the ceiling of each element, i.e. the smallest integer greater than or equal to each element
#floor: Compute the floor of each element, i.e. the largest integer less than or equal to each element
#rint: Round elements to the nearest integer, preserving the dtype
#modf: Return fractional and integral parts of array as separate array
#isnan: Return boolean array indicating whether each value is NaN (Not a Number)
#isfinite, isinf: Return boolean array indicating whether each element is finite (non-inf, non-NaN) or infinite, respectively
#cos, cosh, sin, sinh, tan, tanh: Regular and hyperbolic trigonometric functions
#arccos, arccosh, arcsin, arcsinh, arctan, arctanh: Inverse trigonometric functions
#logical_not: Compute truth value of not x element-wise. Equivalent to -arr.

# add: Add corresponding elements in arrays
# subtract: Subtract elements in second array from first array
# multiply: Multiply array elements
# divide, floor_divide: Divide or floor divide (truncating the remainder)
# power: Raise elements in first array to powers indicated in second array
# maximum, fmax: Element-wise maximum. fmax ignores NaN!!!
# minimum, fmin: Element-wise minimum. fmin ignores NaN!!!
# mod: Element-wise modulus (remainder of division)
# copysign: Copy sign of values in second argument to values in first argument.  np.copysign(1.3, -1) = -1.3
# greater, greater_equal,less, less_equal, equal,not_equal: Perform element-wise comparison, yielding boolean array.
                # Equivalent to infix operators >, >=, <, <=, ==, !=
# logical_and,logical_or, logical_xor(exclusive or): Compute element-wise truth value of logical operation.
                # Equivalent to infix operators &, |, ^


#4. Data Processing Using Arrays
#suppose we wishedto evaluate the function sqrt(x^2 + y^2) across a regular grid of values.
#np.meshgrid(x, y) takes two 1D arrays and produces two 2D matrices corresponding to all pairs of (x,y) in the two arrays
points = np.arange(-5, 5, 0.01)
points
xs, ys = np.meshgrid(points, points)
xs
ys   #xs is the transpose of ys


#evaluating the function is a simple matter of writing the same expression you would write with two points
import matplotlib.pyplot as plt
z = np.sqrt(xs**2 + ys**2)
z
plt.imshow(z, cmap=plt.cm.gray); plt.colorbar()
plt.title("Image plt of $\sqrt{x^2+y^2}$ for a grid of values")


#5. Expressing conditional logic as Array Operations
#numpy.where is a verctorized version of "x if c else y"
xarr = np.array([1.1, 1.2, 1.3, 1.4, 1.5])
yarr = np.array([2.1, 2.2, 2.3, 2.4, 2.5])
cond = np.array([True, False, True, True, False])

#first method
result = [(x if c else y) for x, y, c in zip(xarr, yarr, cond)]
  #zip: Return a list of tuples, where each tuple contains the i-th element from each of the argument sequences.
result
#this method will not be very fast for large arrays and will not work with multidimensionalarrays

#second method: with np.where you can write this very closely:
result = np.where(cond, xarr, yarr)
result
#the 2nd and 3rd arguments to np.where don't need to be arrays; one or both of them can be scalars.

#A typical use of np.where in data analysis is to produce a new array of values based on another array.
arr = randn(4, 4)
arr
np.where(arr > 0, 2, -2)
np.where(arr > 0, 2, arr)  #set only positive values to 2

#more complex logic:
#consider this example where I have two boolean arrays, cond1 and cond2, and wish to assign a
#different value for each of the 4 possible pairs of boolean values:
cond1 = np.array([True, False, True, True, False])
cond2 = np.array([False, True, False, True, False])
result = []
for i in range(5):
    if cond1[i] and cond2[i]:       #both true
        result.append(0)
    elif cond1[i]:
        result.append(1)
    elif cond2[i]:
        result.append(2)
    else:                  #both false
        result.append(3)
result
#other two methods:
result = np.where(cond1 & cond2, 0, np.where(cond1, 1, np.where(cond2, 2, 3)))
result
result = 1 * cond1 + 2 * cond2 + 3 * -(cond1 | cond2) - 3 * (cond1 & cond2)
result


#6. Methematical and Statistical Methods
arr = np.random.randn(5, 4)
arr.mean()
np.mean(arr)  #same as above
arr.sum()

arr.mean(axis=1)  #by column
arr.sum(0)   #by rows

arr = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
arr
arr.cumsum(0)
arr.cumprod(1)

# sum: Sum of all the elements in the array or along an axis. Zero-length arrays have sum 0.
# mean: Arithmetic mean. Zero-length arrays have NaN mean.
# std, var: Standard deviation and variance, respectively, with optional degrees of freedom adjustment (default denominator n).
# min, max: Minimum and maximum.
# argmin, argmax: Indices of minimum and maximum elements, respectively.
# cumsum: Cumulative sum of elements starting from 0
# cumprod: Cumulative product of elements starting from 1


#7. Methods for Boolean Arrays
#Boolean values are coerced to 1 (True) and 0 (False) in the above methods. Thus, sum
#is often used as a means of counting True values in a boolean array:
arr = randn(100)
(arr > 0).sum() # Number of positive values

#There are two additional methods, any and all
bools = np.array([False, False, True, False])
bools.any()
bools.all()
#These methods also work with non-boolean arrays, where non-zero elements evaluate to True.


#8. Sorting
arr = randn(8)
arr
arr.sort()   # this is an in-place sort!!!!!!!!!!
arr

#Multidimensional arrays can have each 1D section of values sorted in-place along an
#axis by passing the axis number to sort:
arr = randn(5, 3)
arr
arr.sort(1)
arr
arr.sort(0)
arr
arr[::-1]

np.random.seed(123)
large_arr = randn(1000)
large_arr.sort()
large_arr[int(0.05 * len(large_arr))] # 5% quantile


#Unique and Other Set Logic
names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
np.unique(names)
ints = np.array([3, 3, 3, 2, 2, 1, 1, 4, 4])
np.unique(ints)

#Contrast np.unique with the pure Python alternative:
sorted(set(names))      #set(): make set

#Another function, np.in1d, tests membership of the values in one array in another,
#returning a boolean array:
values = np.array([6, 0, 0, 3, 2, 5, 6])
np.in1d(values, [2, 3, 6])    #in1d: in one-dimension array

# unique(x) Compute the sorted, unique elements in x
# intersect1d(x, y) Compute the sorted, common elements in x and y
np.intersect1d([1, 3, 4, 3], [3, 1, 2, 1])   #array([1, 3])
# union1d(x, y) Compute the sorted union of elements
# in1d(x, y) Compute a boolean array indicating whether each element of x is contained in y
test = np.array([0, 1, 2, 5, 0])
states = [0, 2]
mask = np.in1d(test, states)
mask    #array([ True, False,  True, False,  True], dtype=bool)
test[mask]  #array([0, 2, 0])
mask = np.in1d(test, states, invert=True)   #If True, the values in the returned array are inverted (that is,
                                            #False where an element of `ar1` is in `ar2` and True otherwise).
mask    #array([False,  True, False,  True, False], dtype=bool)
test[mask]    #array([1, 5])
# setdiff1d(x, y) Set difference, elements in x that are not in y
a = np.array([1, 2, 3, 2, 4, 1])
b = np.array([3, 4, 5, 6])
np.setdiff1d(a, b)   #array([1, 2])
# setxor1d(x, y) Set symmetric differences; elements that are in either of the arrays, but not both
a = np.array([1, 2, 3, 2, 4])
b = np.array([2, 3, 5, 7, 5])
np.setxor1d(a,b)    #array([1, 4, 5, 7])


#7. File Input and Output with Arrays -- in text or binary format.
#(1) np.save and np.load are the two workhorse functions for efficiently saving and loading
#array data on disk. Arrays are saved by default in an uncompressed raw binary format
#with file extension .npy.
arr = np.arange(10)
np.save('some_array', arr)
np.load('some_array.npy')
#You save multiple arrays in a zip archive using np.savez
np.savez('array_archive.npz', a=arr, b=arr)
#When loading an .npz file, you get back a dict-like object which loads the individual arrays lazily:
arch = np.load('array_archive.npz')
arch
type(arch)   #numpy.lib.npyio.NpzFile
arch['a']
arch['b']

#(2) Saving and Loading Text Files
# The landscape of file reading and writing functions in Python can be a bit confusing for a newcomer,
# so I will focus mainly on the read_csv and read_table functions in pandas. It will at times be useful
# to load data into vanilla NumPy arrays using np.loadtxt or the more specialized np.genfromtxt.
arr = np.loadtxt('array_ex.txt', delimiter=',')
arr
#np.savetxt performs the inverse operation

#genfromtxt is similar to loadtxt but is geared for structured arrays and missing data handling


#8. Linear Algebra
x = np.array([[1., 2., 3.], [4., 5., 6.]])
y = np.array([[6., 23.], [-1, 7], [8, 9]])
x
y
x.dot(y)   # equivalently np.dot(x, y)

np.dot(x, np.ones(3))   #results in a 1D array

# numpy.linalg has a standard set of matrix decompositions and things like inverse and
# determinant. These are implemented under the hood using the same industry-standard
# Fortran libraries used in other languages like MATLAB and R, such as like BLAS, LAPACK,
# or possibly (depending on your NumPy build) the Intel MKL.

from numpy.linalg import inv, qr
X = randn(5, 5)
mat = X.T.dot(X)
mat
inv(mat)
mat.dot(inv(mat))
q, r = qr(mat)
r

# Commonly-used numpy.linalg functions
# diag: Return the diagonal (or off-diagonal) elements of a square matrix as a 1D array, or convert a 1D array into a square
# matrix with zeros on the off-diagonal
# dot: Matrix multiplication
# trace: Compute the sum of the diagonal elements
# det: Compute the matrix determinant
# eig: Compute the eigenvalues and eigenvectors of a square matrix
# inv: Compute the inverse of a square matrix
# pinv: Compute the Moore-Penrose pseudo-inverse inverse of a square matrix
# qr: Compute the QR decomposition
# svd: Compute the singular value decomposition (SVD)
# solve: Solve the linear system Ax = b for x, where A is a square matrix
#!!! lstsq: Compute the least-squares solution to y = Xb
   #numpy.linalg.lstsq(a, b, rcond=-1), (2-norm || b - a x ||^2) --> b is the y, a is the data X, x is the coefficient vector
        # x : {(N,), (N, K)} ndarray: Least-squares solution. If b is two-dimensional, the solutions are in the K columns of x.
        # residuals : {(), (1,), (K,)} ndarray: Sums of residuals; squared Euclidean 2-norm for each column in b - a*x.
                                  # If the rank of a is < N or M <= N, this is an empty array. If b is 1-dimensional,
                                  # this is a (1,) shape array. Otherwise the shape is (K,).
        # rank : int. Rank of matrix a.
        # s : (min(M, N),) ndarray. Singular values of a.

#Example: Fit a line, y = mx + c, through some noisy data-points:
x = np.array([0, 1, 2, 3])
y = np.array([-1, 0.2, 0.9, 2.1])
#We can rewrite the line equation as y = Ap, where A = [[x 1]] and p = [[m], [c]].
A = np.vstack([x, np.ones(len(x))]).T
A
####
#numpy.vstack(tup)[source] -- Take a sequence of arrays and stack them vertically to make a single array. Rebuild arrays divided by vsplit.
a = np.array([1, 2, 3])
b = np.array([2, 3, 4])
np.vstack((a,b))
a = np.array([[1], [2], [3]])
b = np.array([[2], [3], [4]])
np.vstack((a,b))
#####
m, c = np.linalg.lstsq(A, y)[0]
print m, c                #c is the intercept.
import matplotlib.pyplot as plt
plt.plot(x, y, 'o', label='Original data', markersize=10)
plt.plot(x, m*x + c, 'r', label='Fitted line')
plt.legend()
plt.show()

#9. Random Number Generation
#you can get a 4 by 4 array of samples from the standard normal distribution using normal
np.random.seed(123)
samples = np.random.normal(size=(4, 4))

#Python’s built-in random module, by contrast, only samples one value at a time.
#numpy.random is well over an order of magnitude faster for generating very large samples:
from random import normalvariate
N = 1000000
%timeit samples = [normalvariate(0, 1) for _ in xrange(N)]  #The general idiom for assigning to a value that isn't used is to name it _.
type(samples)  #list
samples
len(samples)

%timeit np.random.normal(size=N)

##_ is assigned the last result that returned in an interactive python session.
for _ in xrange(10): pass
_
1+2
_

#Partial list of numpy.random functions
# seed: Seed the random number generator
# permutation: Return a random permutation of a sequence, or return a permuted range
# shuffle: Randomly permute a sequence in place
# rand: Draw samples from a uniform distribution
# randint: Draw random integers from a given low-to-high range
# randn: Draw samples from a normal distribution with mean 0 and standard deviation 1 (MATLAB-like interface)
# binomial: Draw samples a binomial distribution
# normal: Draw samples from a normal (Gaussian) distribution
# beta: Draw samples from a beta distribution
# chisquare: Draw samples from a chi-square distribution
# gamma: Draw samples from a gamma distribution
# uniform: Draw samples from a uniform [0, 1) distribution

np.random.seed(123)
np.random.rand()
np.random.seed(123)
np.random.uniform()  #so this is the same with up.random.rand()

#Example: Random Walks
#(1) A pure Python way to implement a single random walk with 1,000 steps using the built-in random module:
import random
position = 0
walk = [position]
steps = 1000
for i in xrange(steps):
    step = 1 if random.randint(0, 1) else -1   #!!! x if c else y
    position += step
    walk.append(position)
plt.plot(walk[:100])
plt.title("Random Walk with +1/-1 steps")
#(2) np.random
nsteps = 1000
draws = np.random.randint(0, 2, size=nsteps)
steps = np.where(draws > 0, 1, -1)
walk = steps.cumsum()
plt.plot(walk[:100])
plt.title("Random Walk with +1/-1 steps")

walk.min()
walk.max()

#A more complicated statistic is the first crossing time, the step at which the random
#walk reaches a particular value.
(np.abs(walk) >= 10).argmax()  #argmax returns the first index of the maximum value in the boolean array
    # Note that using argmax here is not always efficient because it always makes a full scan of the array.
    # In this special case once a True is observed we know it to be the maximum value.

#Simulating Many Random Walks at Once
nwalks = 5000
nsteps = 1000
draws = np.random.randint(0, 2, size=(nwalks, nsteps)) # 0 or 1
steps = np.where(draws > 0, 1, -1)
walks = steps.cumsum(1)
walks.max()
walks.min()
#Out of these walks, let’s compute the minimum crossing time to 30 or -30.
#This is slightly tricky because not all 5,000 of them reach 30.
hits30 = (np.abs(walks) >= 30).any(1)   #any functon is great!
hits30
hits30.sum() # Number that hit 30 or -30
crossing_times = (np.abs(walks[hits30]) >= 30).argmax(1)
crossing_times.mean()
crossing_times.shape


#other distributions for the steps other than equal sized coin flips.
steps = np.random.normal(loc=0, scale=1,size=(nwalks, nsteps))
walks = steps.cumsum(1)
walks.max()
walks.min()
crossing_times = (np.abs(walks[(np.abs(walks) >= 20).any(1)]) >= 20).argmax(1)
crossing_times
crossing_times.mean()
















