__author__ = 'aeoluseros'

#Data Loading, Storage, and File Formats
# Input and output typically falls into a few main categories: (1) reading text files and other
# more efficient on-disk formats, (2) loading data from databases, and (3) interacting with network sources like web APIs.

#1. Reading and Writing Data in Text Format
#Parsing functions in pandas
    # read_csv: Load delimited data from a file, URL, or file-like object. Use comma as default delimiter
    # read_table: Load delimited data from a file, URL, or file-like object. Use tab ('\t') as default delimiter
    # read_fwf: Read data in fixed-width column format (that is, no delimiters)
    # read_clipboard: Version ofread_table that reads data from the clipboard. Useful for converting tables from web pages

from pandas import Series, DataFrame
import pandas.io.data as web   #web.get_data_yahoo
import pandas as pd
import numpy as np
from pandas.io.parsers import TextParser

#（1) Type inference is one of the more important features of these functions; that means you
#don’t have to specify which columns are numeric, integer, boolean, or string.
!cat ./pydata-book/ch06/ex1.csv
df = pd.read_csv('./pydata-book/ch06/ex1.csv')
pd.read_table('./pydata-book/ch06/ex1.csv', sep=',')
!cat ./pydata-book/ch06/ex2.csv
pd.read_csv('./pydata-book/ch06/ex2.csv', header=None)
pd.read_csv('./pydata-book/ch06/ex2.csv', names=['a', 'b', 'c', 'd', 'message'])

#(2) Suppose you wanted the message column to be the index of the returned DataFrame:
names = ['a', 'b', 'c', 'd', 'message']
pd.read_csv('./pydata-book/ch06/ex2.csv', names=names, index_col='message')

#In the event that you want to form a hierarchical index from multiple columns, just
#pass a list of column numbers or names:
!cat ./pydata-book/ch06/csv_mindex.csv
parsed = pd.read_csv('./pydata-book/ch06/csv_mindex.csv', index_col=['key1', 'key2'])
parsed

#(3) In some cases, a table might not have a fixed delimiter, using whitespace or some other pattern to
# separate fields. In these cases, you can pass a regular expression as a delimiter for read_table.
list(open('./pydata-book/ch06/ex3.txt'))  #in this case fields are separated by a variable amount of whitespace
   #list(), make the object a list
!cat ./pydata-book/ch06/ex3.txt
#This can be expressed by the regular expression \s+
result = pd.read_table('./pydata-book/ch06/ex3.txt', sep='\s+')  #\s means space
result
#Because there was one fewer column name than the number of data rows, read_table
#infers that the first column should be the DataFrame’s index in this special case.

#(4) The parser functions have many additional arguments to help you handle the wide variety of exception file formats that occur.
#For example, you can skip the first, third, and fourth rows of a file with skiprows:
!cat ./pydata-book/ch06/ex4.csv
pd.read_csv('./pydata-book/ch06/ex4.csv', skiprows=[0, 2, 3])

#(5)Handling missing values
#By default, pandas uses a set of commonly occurring sentinels, such as NA, -1.#IND, and NULL:
!cat ./pydata-book/ch06/ex5.csv
result = pd.read_csv('./pydata-book/ch06/ex5.csv')  #space and NA automatically turned into NaN
result
pd.isnull(result)

#The na_values option can take either a list or set of strings to consider missing values:
result = pd.read_csv('./pydata-book/ch06/ex5.csv', na_values=[9, 12, 2, 'world'])
result

#Different NA sentinels can be specified for each column in a dict:
sentinels = {'message': ['foo', 'NA'], 'something': ['two']}
pd.read_csv('./pydata-book/ch06/ex5.csv', na_values=sentinels)

###read_csv /read_table function arguments
#path: String indicating filesystem location, URL, or file-like object
#sep or delimiter: Character sequence or regular expression to use to split fields in each row
#header: Row number to use as column names. Defaults to 0 (first row), but should be None if there is no header row
#index_col: Column numbers or names to use as the row index in the result. Can be a single name/number or a list of them for a hierarchical index
#names: List of column names for result, combine with header=None
#skiprows: Number of rows at beginning of file to ignore or list of row numbers (starting from 0) to skip
#na_values: Sequence of values to replace with NA
#comment: Character or characters to split comments off the end of lines
#parse_dates Attempt to parse data to datetime; False by default. If True, will attempt to parse all columns. Otherwise
    #can specify a list of column numbers or name to parse. If element of list is tuple or list, will combine
    #multiple columns together and parse to date (for example if date/time split across two columns)
#keep_date_col If joining columns to parse date, drop the joined columns. Default True
#!!!converters: Dict containing column number of name mapping to functions. For example{'foo': f} would apply
    #the function f to all values in the 'foo' column
#dayfirst: When parsing potentially ambiguous dates, treat as international format (e.g. 7/6/2012 -> June 7, 2012). Default False
#date_parser: Function to use to parse dates
#nrows: Number of rows to read from beginning of file
#iterator: Return a TextParser object for reading file piecemeal
#chunksize For iteration, size of file chunks
#!!!skip_footer Number of lines to ignore at end of file
#verbose Print various parser output information, like the number of missing values placed in non-numeric columns
#encoding: Text encoding for unicode. For example 'utf-8' for UTF-8 encoded text
#!!!squeeze: If the parsed data only contains one column return a Series
#!!!thousands: Separator for thousands, e.g. ',' or '.'


#(6)Reading Text Files in Pieces
#When processing very large files or figuring out the right set of arguments to correctly
#process a large file, you may only want to read in a small piece of a file or iterate through
#smaller chunks of the file.
result = pd.read_csv('./pydata-book/ch06/ex6.csv')
result
#If you want to only read out a small number of rows (avoiding reading the entire file), specify that with nrows
pd.read_csv('./pydata-book/ch06/ex6.csv', nrows=5)

#To read out a file in pieces, specify a chunksize as a number of rows:
chunker = pd.read_csv('./pydata-book/ch06/ex6.csv', iterator=True)
#chunker there is an iterator.
chunker = pd.read_csv('./pydata-book/ch06/ex6.csv', chunksize=1000)  #iterator will be implicitly set to True once we specify chunksize
                                                #that is, chukker is born as an iterator.
chunker   #<pandas.io.parsers.TextFileReader at 0xd0fc7f0>  --> a kind of iterator.
#The TextParser object returned by read_csv allows you to iterate over the parts of the file according to the chunksize
tot = Series([])
for piece in chunker:
    tot = tot.add(piece['key'].value_counts(), fill_value=0)   #piece is dict. every element in chunker is dict. chunker is a dict generator
        #add is the method of expand tot.
tot = tot.order(ascending=False)
tot[:10]
#TextParser is also equipped with a get_chunk method which enables you to read pieces of an arbitrary size.
chunker.get_chunk()   #After 10 times, it will become an error. Every chunker generated are different.


#(7) Writing Data Out to Text Format
#Data can also be exported to delimited format
data = pd.read_csv('pydata-book/ch06/ex5.csv')
data
data.to_csv('ch06-out.csv')
!cat ch06-out.csv

###Other delimiters can be used:
data.to_csv(sys.stdout, sep='|') (writing to sys.stdout so it just prints the text result)
# Missing values appear as empty strings in the output.
# You might want to denote them by some other sentinel value:
data.to_csv(sys.stdout, na_rep='May')

#With no other options specified, both the row and column labels are written.Both of these can be disabled:
data.to_csv(sys.stdout, index=False, header=False)

#You can also write only a subset of the columns, and in an order of your choosing:
data.to_csv(sys.stdout, index=False, cols=['a', 'b', 'c'])


#Series also has a to_csv method:
dates = pd.date_range('1/1/2000', periods=7)
    # date_range(start=None, end=None, periods=None, freq='D', tz=None, normalize=False, name=None, closed=None)
    # Return a fixed frequency datetime index, with day (calendar) as the default frequency
dates   #[2000-01-01, ..., 2000-01-07]   <class 'pandas.tseries.index.DatetimeIndex'>
ts = Series(np.arange(7), index=dates)
ts
ts.to_csv('ch06-tseries.csv')
!cat ch06-tseries.csv

#With a bit of wrangling (no header, first column as index), you can read a CSV version
#of a Series with read_csv, but there is also a from_csv convenience method that makes
#it a bit simpler:
tseries = Series.from_csv('ch06-tseries.csv', parse_dates=True)
type(tseries)    #pandas.core.series.Series

#(8) Manually Working with Delimited Formats
#For any file with a single-character delimiter, you can use Python’s built-in csv module.
!cat pydata-book/ch06/ex7.csv
import csv
f = open('pydata-book/ch06/ex7.csv')
reader = csv.reader(f)
reader   #reader is a iterator, <_csv.reader at 0xd3fc048>
for line in reader:
    print line

#From there, it’s up to you to do the wrangling necessary to put the data in the form that you need it. For example:
lines = list(csv.reader(open('pydata-book/ch06/ex7.csv')))
lines
headers, values = lines[0], lines[1:]   #multiple return
data_dict = {h: v for h, v in zip(headers, zip(*values))}  #turn list into a dict!!!
data_dict

# CSV files come in many different flavors. Defining a new format with a different delimiter, string quoting
# convention, or line terminator is done by defining a simple subclass of csv.Dialect:
class my_dialect(csv.Dialect):
    lineterminator = '\n'
    delimiter = ';'
    quotechar = '"'
    quoting = csv.QUOTE_ALL   #whether to add quotes to elements

# Class Dialect
    #  |  Data and other attributes defined here:
    #  |  delimiter = None
    #  |  doublequote = None
    #  |  escapechar = None
    #  |  lineterminator = None
    #  |  quotechar = None
    #  |  quoting = None
    #  |  skipinitialspace = None
#CSV dialect options
# delimiter: One-character string to separate fields. Defaults to ','.
# lineterminator: Line terminator for writing, defaults to '\r\n'. Reader ignores this and recognizes cross-platform line terminators.
# quotechar: Quote character for fields with special characters (like a delimiter). Default is '"'.
# quoting: Quoting convention. Options include csv.QUOTE_ALL (quote all fields), csv.QUOTE_MINIMAL
            # (only fields with special characters like the delimiter), csv.QUOTE_NONNUMERIC, and
            # csv.QUOTE_NONE (no quoting). Defaults to QUOTE_MINIMAL.
#skipinitialspace: Ignore whitespace after each delimiter. Default False.
#doublequote: How to handle quoting character inside a field. If True, it is doubled. See online documentation for full detail and behavior.
#escapechar: String to escape the delimiter if quoting is set to csv.QUOTE_NONE. Disabled by default.

!cat pydata-book/ch06/ex7.csv
f = open('pydata-book/ch06/ex7.csv')
reader = csv.reader(f, dialect=my_dialect)
for line in reader:
    print line
list(reader)


#To write delimited files manually, you can use csv.writer. It accepts an open, writable
#file object and the same dialect and format options as csv.reader
with open('mydata.csv', 'w') as f:                 #create a connection
    writer = csv.writer(f, dialect=my_dialect)
    writer.writerow(('one', 'two', 'three'))
    writer.writerow(('1', '2', '3'))
    writer.writerow(('4', '5', '6'))
    writer.writerow(('7', '8', '9'))

f=open('mydata.csv', 'w')
writer = csv.writer(f, dialect=my_dialect)
type(writer)   #_csv.writer


#2. JSON Data  --> a much more flexible data format than a tabular text form like CSV
obj = """
{"name": "Wes",
"places_lived": ["United States", "Spain", "Germany"],
"pet": null,
"siblings": [{"name": "Scott", "age": 25, "pet": "Zuko"},
{"name": "Katie", "age": 33, "pet": "Cisco"}]
}
"""
type(obj)   #str
# JSON is very nearly valid Python code with the exception of its null value null and some other
# nuances (such as disallowing trailing commas at the end of lists). The basic types of JSON are
# objects (dicts), arrays (lists), strings, numbers, booleans, and nulls. All of the keys in an
# object must be strings.

# There are several Python libraries for reading and writing JSON data. I’ll use json here as it
# is built into the Python standard library. To convert a JSON string to Python form, use json.loads:
import json
result = json.loads(obj)
#records = [json.loads(line) for line in open(path)]   #list comprehension
result
type(result)   #dict

#json.dumps on the other hand converts a Python object back to JSON
asjson = json.dumps(result)

#How you convert a JSON object or list of objects to a DataFrame or some other data structure for analysis will be up to you.
#Conveniently, you can pass a list of JSON objects to the DataFrame constructor and select a subset of the data fields:
siblings = DataFrame(result['siblings'], columns=['name', 'age'])   #pick the value to the key 'siblings'
siblings

###An effort is underway to add fast native JSON export (to_json) and decoding (from_json) to pandas.


#3. XML and HTML: Web Scraping
#Many websites make data available in HTML tables for viewing in a browser, but not
#downloadable as an easily machine-readable format like JSON, HTML, or XML.
#(1) I noticed that this was the case with Yahoo! Finance’s stock options data.
from lxml.html import parse
from urllib2 import urlopen
parsed = parse(urlopen('http://finance.yahoo.com/q/op?s=AAPL+Options'))
doc = parsed.getroot()   #XML's root.
#Using the document root’s findall method along with an XPath
links = doc.findall('.//a')
type(links)    #list
links[15:20]         #these are objects representing HTML elements
#to get the URL and link text you have to use each element’s get method (for the URL) and text_content method (for the display text):
lnk = links[38]
lnk
type(lnk)    # lxml.html.HtmlElement
lnk.get('href')   #href is the hyperlink.
lnk.text_content()
#Thus, getting a list of all URLs in the document is a matter of writing this list comprehension:
urls = [lnk.get('href') for lnk in doc.findall('.//a')]
contents = [lnk.text_content() for lnk in doc.findall('.//a')]
urls[-10:]
contents

#Now, finding the right tables in the document can be a matter of trial and error;
#!!! some websites make it easier by giving a table of interest an id attribute.
tables = doc.findall('.//table')
len(tables)
calls = tables[1]    #HtmlElement
puts = tables[2]

callrows = calls.findall('.//tr')
putrows = puts.findall('.//tr')
#For the header as well as the data rows, we want to extract the text from each cell; in
#the case of the header these are th cells and td cells for the data:
def _unpack(row, kind='td'):
    elts = row.findall('.//%s' % kind)
    return [str(val.text_content().strip().splitlines()[0]) for val in elts]

_unpack(callrows[0], kind='th')
_unpack(callrows[2], kind='td')
_unpack(callrows[3], kind='td')
_unpack(callrows[15], kind='td')

_unpack(putrows[0], kind='th')
_unpack(putrows[2], kind='td')

# it’s a matter of combining all of these steps together to convert this data into a DataFrame.
from pandas.io.parsers import TextParser
def parse_options_data(table):
    rows = table.findall('.//tr')
    header = _unpack(rows[0], kind='th')
    data = [_unpack(r) for r in rows[2:]]
    return TextParser(data, names=header).get_chunk()
call_data = parse_options_data(calls)
put_data = parse_options_data(puts)
call_data[:10]
put_data[:10]


#(2) Parsing XML with lxml.objectify




#4. Binary Data Formats
# One of the easiest ways to store data efficiently in binary format is using Python’s builtin pickle serialization.
# Conveniently, pandas objects all have a save method which writes the data to disk as a pickle:
frame = pd.read_csv('pydata-book/ch06/ex1.csv')
frame.save('ch06-frame_pickle')
# pickle is only recommended as a short-term storage format. The problem is that it is hard to guarantee that
# the format will be stable over time; an object pickled today may not unpickle with a later version of a library.


#5. Using HDF5 Format





















