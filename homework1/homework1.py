import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt

filename = sys.argv[-2]
attribute_number = int(sys.argv[-1])

def maximum(filename, attribute_number):
    #Input: Filename a string and the attribute number 1 to 10
    #Output: Returns maximum value
    #Do not print anything to stdout
    data = pd.read_csv(filename, header = None)
    data = data.drop(columns = [10])
    data = data.to_numpy()
    data = np.array(data, dtype = np.float32)
    return np.amax(data[:, attribute_number])

def minimum(filename, attribute_number):
    #Input: Filename a string and the attribute number 1 to 10
    #Output: Returns minimum value
    #Do not print anything to stdout
    data = pd.read_csv(filename, header = None)
    data = data.drop(columns = [10])
    data = data.to_numpy()
    data = np.array(data, dtype = np.float32)
    return np.amin(data[:, attribute_number])

def mean(filename, attribute_number):
    #Input: Filename a string and the attribute number 1 to 10
    #Output: Returns mean value
    #Do not print anything to stdout
    data = pd.read_csv(filename, header = None)
    data = data.drop(columns = [10])
    data = data.to_numpy()
    data = np.array(data, dtype = np.float32)
    return np.mean(data[:, attribute_number])

def std(filename, attribute_number):
    #Input: Filename a string and the attribute number 1 to 10
    #Output: Returns standard deviation value
    #Do not print anything to stdout
    data = pd.read_csv(filename, header = None)
    data = data.drop(columns = [10])
    data = data.to_numpy()
    data = np.array(data, dtype = np.float32)
    return np.std(data[:, attribute_number])

def q1(filename, attribute_number):
    #Input: Filename a string and the attribute number 1 to 10
    #Output: Returns Q1 value
    #Do not print anything to stdout
    data = pd.read_csv(filename, header = None)
    data = data.drop(columns = [10])
    data = data.to_numpy()
    data = np.array(data, dtype = np.float32)
    return np.quantile(data[:, attribute_number], 1 / 4)

def q3(filename, attribute_number):
    #Input: Filename a string and the attribute number 1 to 10
    #Output: Returns Q3 value
    #Do not print anything to stdout
    data = pd.read_csv(filename, header = None)
    data = data.drop(columns = [10])
    data = data.to_numpy()
    data = np.array(data, dtype = np.float32)
    return np.quantile(data[:, attribute_number], 3 / 4)

def median(filename, attribute_number):
    #Input: Filename a string and the attribute number 1 to 10
    #Output: Returns median value
    #Do not print anything to stdout
    data = pd.read_csv(filename, header = None)
    data = data.drop(columns = [10])
    data = data.to_numpy()
    data = np.array(data, dtype = np.float32)
    return np.quantile(data[:, attribute_number], 1 / 2)

def iqr(filename, attribute_number):
    #Input: Filename a string and the attribute number 1 to 10
    #Output: Returns IQR value
    #Do not print anything to stdout
    data = pd.read_csv(filename, header = None)
    data = data.drop(columns = [10])
    data = data.to_numpy()
    data = np.array(data, dtype = np.float32)
    return q3(filename, attribute_number) - q1(filename, attribute_number)

def count(filename, attribute_number):
    #Input: Filename a string and the attribute number 1 to 10
    #Output: Returns count value
    #Do not print anything to stdout
    data = pd.read_csv(filename, header = None)
    data = data.drop(columns = [10])
    data = data.to_numpy()
    data = np.array(data, dtype = np.float32)
    return len(data[:, attribute_number])

print("maximum", maximum(filename, attribute_number))
print("minimum", minimum(filename, attribute_number))
print("mean", mean(filename, attribute_number))
print("std", std(filename, attribute_number))
print("q1", q1(filename, attribute_number))
print("q3", q3(filename, attribute_number))
print("median", median(filename, attribute_number))
print("iqr", iqr(filename, attribute_number))
print("count", count(filename, attribute_number))


data = pd.read_csv(filename, header = None)
data = data.drop(columns = [10])
data = data.to_numpy()
data = np.array(data, dtype = np.float32)
print(data[:, 4])
print(data[:, 5])
plt.scatter(data[:, 4], data[:, 5])
plt.show()

