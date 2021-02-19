import math
# Part 1 Practice
my_list = []
for i in range(10):
    i = i+1
    i = math.pow(i, 2)
    my_list.append(i)

print(my_list)
my_list.sort(reverse=True)
print(my_list)
my_list.pop(-3)
my_list.pop(-3)
print(my_list)

# Part 2
empty_tuple = ()
print(empty_tuple)
tup = 'python', 'geeks'
print(tup)
tup1 = (0, 1, 2, 3)
tup2 = ('python', 'geek')
print(tup1+tup2)
tup3 = (tup1, tup2)
print(tup3)
tup3 = ('python',)*3
print(tup3)

# Code for converting a list and a string into a tuple

list1 = [0, 1, 2]
print(tuple(list1))
print(tuple('python')) # string 'python'

# python code for creating tuples in a loop

tup = ('ENPM661',)
n = 5  # Number of time loop runs
for i in range(int(n)):
    tup = (tup,)
    print(tup)

    # Creating an empty Dictionary
    Dict = {}
    print("Empty Dictionary: ")
    print(Dict)

    # Creating a Dictionary
    # with Integer Keys
    Dict = {1: 'Geeks', 2: 'For', 3: 'Geeks'}
    print("\nDictionary with the use of Integer Keys: ")
    print(Dict)

    # Creating a Dictionary
    # with Mixed keys
    Dict = {'Name': 'Geeks', 1: [1, 2, 3, 4]}
    print("\nDictionary with the use of Mixed Keys: ")
    print(Dict)

    # Creating a Dictionary
    # with dict() method
    Dict = dict({1: 'Geeks', 2: 'For', 3: 'Geeks'})
    print("\nDictionary with the use of dict(): ")
    print(Dict)

    # Creating a Dictionary
    # with each item as a Pair
    Dict = dict([(1, 'Geeks'), (2, 'For')])
    print("\nDictionary with each item as a pair: ")
    print(Dict)
# Creating a Nested Dictionary
# as shown in the below image
Dict = {1: 'Geeks', 2: 'For',
        3: {'A': 'Welcome', 'B': 'To', 'C': 'Geeks'}}

print(Dict)
# Creating an empty Dictionary
Dict = {}
print("Empty Dictionary: ")
print(Dict)

# Adding elements one at a time
Dict[0] = 'Geeks'
Dict[2] = 'For'
Dict[3] = 1
print("\nDictionary after adding 3 elements: ")
print(Dict)

# Adding set of values
# to a single Key
Dict['Value_set'] = 2, 3, 4
print("\nDictionary after adding 3 elements: ")
print(Dict)

# Updating existing Key's Value
Dict[2] = 'Welcome'
print("\nUpdated key value: ")
print(Dict)

# Adding Nested Key value to Dictionary
Dict[5] = {'Nested' :{'1' : 'Life', '2' : 'Geeks'}}
print("\nAdding a Nested Key: ")
print(Dict)
