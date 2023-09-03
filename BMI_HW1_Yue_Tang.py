## author: Yue Tang (Goizueta Business School)
## email: yue.tang@emory.edu
## date: Aug 29, 2023

##################################################################################
'''
Fresher
a) Run the script from session 1 and review the different tasks we covered.
b) Modify the code for each task run the code.
c) Look up the documentation for lists, dictionaries and strings.
d) We discussed a little about dictionaries during the session. We only looked at the basic dictionary. Are there other types of dictionaries? What is a defaultdict?
'''
##################################################################################

"""
#2 HELLO WORLD PROGRAM IN PYCHARM/PYTHON
"""
print('----------------------')
print('Task 2 ...............')
print('----------------------')
print('hello world!')

print( 'hello ' + ' world')
print( 'hello','world')
print( 'Yue ' + 'Tang')

##################################################################################

"""
#3 SYNTAX IN PYTHON
"""
##C++-LIKE CODE:
a = 5
if(a>4):
    print(a)
    a = a+1
    print(a)


print('----------------------')
print('Task 3 ...............')
print('----------------------')

##################################################################################

"""
#4 TYPES:
"""
#STRINGS
print ('----------------------')
print ('Task 4 ...............')
print ('----------------------')

s1 = "This is line 1."
s2 = 'This is line 2.'
s3 = 'This line has a \' in the middle'
s4 = '''These are lines 4 and 5.
      It is a multi-line string'''
s5 = """This is another multi-line
    string."""
print(s1)
print(s2)
print(s3)
print(s4)
print(s5)



#Integers and floats
i1 = 1245
i2 = int('1234')
f1 = 42.
f2 = 42.123123

print (i1,i2,f1,f2)

#Null value
null_value = None
print(null_value)

#boolean
b1 = True
b2 = False
print(b2)
#These are all equivalent to False: False, 0, "", {}, [], None

#Type conversions

num = input("Enter a number: ")
print ("The number is is:", str(num))
num=1
num=1.1
num = '1'
num = []
num = {}
num = '1'
intnum = int(num)
print(type(intnum))
##################################################################################

"""
#5 DATA STRUCTURES
"""
print('----------------------')
print('Task 5 ...............')
print('----------------------')

#LISTS
list1 = []
list2 = list()
list3 = [1,4,2,5,2,6]
list4 = [5,1,5,2]
list5 = []
list6 = ['this', 'is', 'a', 'list', 'of', 'strings']
print(list1, list2, list3, list4, list5, list6)
print(list3[0])

#appending
list5.append(10)
list5.append("what?!")
list5.append("BMI! NLP!")
print(list5)

#extending
list5.extend(['another','heterogeneous','list',53,None])
print (list5)

list5.extend(list3)
print (list5)

#concatenation
list5 = list3 + list4
print (list3)
print (list4)
print (list5)

#lengths
print (len(list5))

#access + operations
print (list5[0])
print (list5.count(2))
print (list5.index(6))
print (sorted(list5))
print (list5)
print (list5.pop())
print (list5)
print (list5[:3])
print (list5[3:])

#DICTIONARIES
patient = {}
patient = {'age': 28, 'name': 'Yue Tang'}
patient['gender'] = 'male'
patient['visa type'] = 'F1'
patient[123] = 'any immutable object can be a key'
patient['list of items']  = ['a', 'value', 'can', 'be', 'anything']
print(patient['age'])
print (patient[123])
print (patient['list of items'])
print (patient.keys())
print (patient.values())
print (patient.items())
key = 'age'
print(patient[key])

#SETS
list1 = [1,2,2,3,3,3,3,4,5,6,7]
list2 = [3,4,6,7,9]
print (list1)
print (set(list1))
print (set(list1).intersection(set(list2)))
print (set(list1).union(list2))

#DEFAULTDICT
#Note: we did not cover this during the session, but we will come across it in week 2
from collections import defaultdict
listdict = defaultdict(list)
listdict['firstlist'].append(2)
listdict['firstlist'].append(3)
listdict['otherlist'].append('this')
listdict['otherlist'].append('is a string..')
print (listdict['firstlist'])
print (listdict['otherlist'])
listdict['anotherentry'].append(patient)
print (listdict['anotherentry'])
s = [('red', 1), ('blue', 2), ('red', 3), ('blue', 4), ('red', 1), ('blue', 4)]
d = defaultdict(set)
for i, j in s:
    d[i].add(j)

print (d.items())

#TUPLE
t1 = (2,3,4)
print (t1[0])
print (t1[1])

#################################################################################
"""
#6 OPERATORS
"""
print ('----------------------')
print ('Task 6 ...............')
print ('----------------------')
#ARITHMETIC OPERATORS
a = 2
a += 11
a -= 1
b = a * 2
c = a / 2
d = a % 3
e = a** 2
print (e)

#STRING OPERATORS
#Concatenating lists
two_phrases = 'first one and ' + 'second one'
print (two_phrases)
another_one = two_phrases + '... some more...'
print (another_one)

#Join
print (list6)
list6string = '$$$'.join(list6)
print (list6string)

#printing and dictionary
address = '%d %s' % (101, 'Randall Rollins')
print (address)
address2 = '%(number)d %(name)s' % {'number': 101, 'name':'Randall Rollins'}
print (address2)

#################################################################################
"""
#7 LOGICAL OPERATORS AND CONDITIONS
"""
print ('----------------------')
print ('Task 7 ...............')
print ('----------------------')

#LOGICAL OPERATORS (and, or, not, ==, is)
a = True
b = True
print (a or b)
print (not a == b)
a = [1,2,3]
b = [1,2,3]
print (a is b)
print (a is not b)
print (a == b)
# The == operator compares the value or equality of two objects, whereas the Python is operator checks whether two variables point to the same object in memory. In the vast majority of cases, this means you should use the equality operators == and !=
#CONDITIONS IF/ELSE
bp = 145
temp = 15
if bp > 140 and temp >40:
    print ('both are above the limit')
elif bp > 140 and temp < 40 and temp >30:
    print ('bp > 140, temp between 30 and 40')
elif bp < 140 and temp > 40:
    print ('bp < 140, temp is above 40')
elif bp < 140 and temp < 40:
    print ('both are lower?')
else:
    print ('when will we reach this....?')
    if temp < 30:
        print ('temp is too low?')

#Other common uses
if len(list1) == 0: #to check if a data structure is empty
    print("list1 is EMPTY!")
else:
    print("not something you are looking for?")
#################################################################################
"""
#8 LOOPS
"""
print ('----------------------')
print ('Task 8 ...............')
print ('----------------------')

#Basic for loop
for x in range(10):#(1,10,2)
    print (x)

for x in range(1,10,2):
    print (x)


#Loop over lists
list7 = ['This','is','another','list']
for item in list7:
    print (item)

#Looping through 2 dimensional array
list8 = [[1,2,3],[4,5,6],[7,8,9]]
for i in list8:
    for j in i:
        print (j)

#Looping through dictionaries
for k in patient.keys():
    print (k)
    print (k,'\t',patient[k])

for k,v in patient.items():
    print (k,'\t',v)

#List comprehensions
#Note: Not covered in session 1. Will be covered soon.
x = []
for i in range(1,20):
    if i % 3 == 0:
        x.append(i)
        print (x)

x = [i for i in range(1,20) if i%3 ==0]
print (x)

y = [i +j for i in range(1,20) for j in range (1,30) if (i*i)%j==0]
print (y)

#What will these do?

S = [x**2 for x in range(10)]
V = [2**i for i in range(13)]
M = [x for x in S if x % 2 == 0]
N = [2*3.14*r for r in range(2,20,3)]
print (S)
print (V)
print (M)
print(N)
z = [(i,i*2,i**2,i+i) for i in range(1,20)]
for item in z:
    print (item)
#################################################################################

#################################################################################
# Homework1: Q2-Q5
#################################################################################

#################################################################################
"""
#HW1 Q2
a) Write some code to initialize a list and fill it up with the first n values of the fibonacci sequence. n will be input by the user when the program runs.
"""
fibonacci = []
n = int(input("Enter a number: "))
for i in range(0,n):
    if(i==0 or i==1):
        fibonacci.append(1)
    else:
        fibonacci.append(fibonacci[i-1]+fibonacci[i-2])

print(fibonacci)

"""
#HW1 Q3
3. A palindrome is a string that reads the same backwards and forwards. 
a) Write and run some code that checks if a given string is palindrome and prints the result on the screen
"""
listx = [1,2,3,5,6]
listy = []
for i in range(len(listx)-1,-1,-1):
    listy.append(listx[i])

if(listx==listy):
    print("Palindrome as it is!")
else:
    print("No way!")

"""
#HW1 Q4
4. Python functions.
a) Look up how to write basic functions in python. 
b) Convert the previous code to a function. The string must be passed as a function parameter and the function will return True if the string is a palindrome.
"""

def fibonacci(n):
    list_ = []
    for i in range(0, n):
        if (i == 0 or i == 1):
            list_.append(1)
        else:
            list_.append(list_[i - 1] + list_[i - 2])
    print("The first "+str(n)+"numbers in Fibonacci sequence is: ")
    print(list_)

n = 10
print(fibonacci(n))

def Palindrome(listx):
    listy = []
    for i in range(len(listx) - 1, -1, -1):
        listy.append(listx[i])
    if (listx == listy):
        print("Palindrome as it is!")
    else:
        print("No way!")

listkk = [1,6,7,8,9,5,1]
print(Palindrome(listkk))

"""
#HW1 Q5
5. Consider the following lists.
names = ['Real Madrid','AC Milan','Manchester United’,’FC Barcelona','Bayern Munich', 'Liverpool FC’, 'Internazionale']
number_of_intl_titles = [14, 7, 3, 5, 6, 6, 3]
country = ['Spain', 'Italy', 'England', 'Spain','Germany', 'England', 'Italy']

The lists contain the names of soccer clubs from Europe, the number of European championships they have won and their countries of origin (in the same sequence).
a) Store the club names and number of titles in a dictionary with the club name as key and the number as the value.
b) Store the total number of titles per country in a dictionary- country as key and the sum of the titles as value.
c) Based on the data, which country has the lowest number of european titles? Which country has the highest? What is the average number of wins per country?
"""

name_title = {}
country_title = {}
names = ['Real Madrid','AC Milan','Manchester United','FC Barcelona','Bayern Munich','Liverpool FC','Internazionale']
number_of_intl_titles = [14, 7, 3, 5, 6, 6, 3]
country = ['Spain', 'Italy', 'England', 'Spain','Germany', 'England', 'Italy']

## for a)
for i in range(0, len(names)):
    name_title[names[i]] = number_of_intl_titles[i]

print(name_title)
## for b)
for i in range(len(country)):
    if country[i] in country_title:
        country_title[country[i]] += number_of_intl_titles[i]
    else:
        country_title[country[i]] = number_of_intl_titles[i]
print()

# country with lowest number of titles
lowest_country = min(country_title, key=country_title.get)
# country with highest number of titles
highest_country = max(country_title, key=country_title.get)
# average number of titles
mean_titles = sum(number_of_intl_titles)/len(number_of_intl_titles)

print(lowest_country, highest_country, mean_titles)