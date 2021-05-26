import pandas as pd
import numpy as np

data = pd.read_fwf("magic04.data", header = None)

array = np.zeros((data.shape[0], 12))
print(data.shape)
i = 0

for i in range(data.shape[0]):
    temp = data.loc[i].to_string()
    temp = temp.split(',')
    print(temp)
    print(len(temp))
    i += 1

#data = np.array(data, dtype = np.float32)

# def maximum(filename, attribute_number):
#     #Input: Filename a string and the attribute number 1 to 10
#     #Output: Returns maximum value
#     #Do not print anything to stdout

# def minimum(filename, attribute_number):
#     #Input: Filename a string and the attribute number 1 to 10
#     #Output: Returns minimum value
#     #Do not print anything to stdout

# def mean(filename, attribute_number):
#     #Input: Filename a string and the attribute number 1 to 10
#     #Output: Returns mean value
#     #Do not print anything to stdout

# def std(filename, attribute_number):
#     #Input: Filename a string and the attribute number 1 to 10
#     #Output: Returns standard deviation value
#     #Do not print anything to stdout

# def q1(filename, attribute_number):
#     #Input: Filename a string and the attribute number 1 to 10
#     #Output: Returns Q1 value
#     #Do not print anything to stdout

# def q3(filename, attribute_number):
#     #Input: Filename a string and the attribute number 1 to 10
#     #Output: Returns Q3 value
#     #Do not print anything to stdout

# def median(filename, attribute_number):
#     #Input: Filename a string and the attribute number 1 to 10
#     #Output: Returns median value
#     #Do not print anything to stdout

# def iqr(filename, attribute_number):
#     #Input: Filename a string and the attribute number 1 to 10
#     #Output: Returns IQR value
#     #Do not print anything to stdout

# def count(filename, attribute_number):
#     #Input: Filename a string and the attribute number 1 to 10
#     #Output: Returns count value
#     #Do not print anything to stdout
