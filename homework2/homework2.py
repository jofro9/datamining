import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

def correlation ( attribute1 , fileName1 , attribute2, fileName2 ):
    '''
    Input Parameters:
        attribute1: The attribute you want to consider from file1
        attribute2: The attribute you want to consider from file2
        fileName1: The comma seperated file1 with the different format first line removed
        fileName2: The comma seperated file2 with the different format first line removed

    Output:
        Return the correlation coefficient
        Do not return a string
        Do not write anything to stdout
        Points will be deducted if these instructions are not followed strictly

    '''

    # antiquated I know but so I can use numpy for speed
    if attribute1 == "close":
        attribute1 = 0

    elif attribute1 == "volume":
        attribute1 = 1

    elif attribute1 == "open":
        attribute1 = 2

    elif attribute1 == "high":
        attribute1 = 3

    elif attribute1 == "low":
        attribute1 = 4

    if attribute2 == "close":
            attribute2 = 0

    elif attribute2 == "volume":
        attribute2 = 1

    elif attribute2 == "open":
        attribute2 = 2

    elif attribute2 == "high":
        attribute2 = 3

    elif attribute2 == "low":
        attribute2 = 4


    #TODO: Write code given the Input / Output Paramters.
    data1 = pd.read_csv(fileName1, header = 0)
    data2 = pd.read_csv(fileName2, header = 0)
    data1 = data1.drop(columns = "Date")
    data2 = data2.drop(columns = "Date")

    # clean data values to be floats
    data1['Close/Last'] = data1['Close/Last'].str.replace('$', '', regex = True)
    data1['Open'] = data1['Open'].str.replace('$', '', regex = True)
    data1['High'] = data1['High'].str.replace('$', '', regex = True)
    data1['Low'] = data1['Low'].str.replace('$', '', regex = True)
    data2['Close/Last'] = data2['Close/Last'].str.replace('$', '', regex = True)
    data2['Open'] = data2['Open'].str.replace('$', '', regex = True)
    data2['High'] = data2['High'].str.replace('$', '', regex = True)
    data2['Low'] = data2['Low'].str.replace('$', '', regex = True)

    # convert to numpy arrays for easy operation
    data1 = data1.to_numpy()
    data2 = data2.to_numpy()
    data1 = np.array(data1, dtype = np.float32)
    data2 = np.array(data2, dtype = np.float32)

    # calculate correlation based on book formula
    mean1 = np.mean(data1[:, attribute1])
    mean2 = np.mean(data2[:, attribute2])

    mean1 = np.full(data1[:, attribute1].shape, mean1, dtype = np.float32)
    mean2 = np.full(data2[:, attribute2].shape, mean2, dtype = np.float32)

    temp = (data1[:, attribute1] - mean1[:]) * (data2[:, attribute2] - mean2[:])
    tempsum = sum(temp)

    return tempsum / (temp.shape[0] * np.std(data1[:, attribute1]) * np.std(data2[:, attribute2]))

def min_max_normalization(fileName, attribute):
    '''
    Input Parameters:
        fileName: The comma seperated file that must be considered for the normalization with the different format first line removed
        attribute: The attribute for which you are performing the normalization

    Output:
        Return an object of list type of normalized values
        Please do not return a list of strings
        Please do not print anything to stdout
        Use only Python3
        Points will be deducted if you do not follow exact instructions
    '''
    data1 = pd.read_csv(fileName, header = 0)
    data1 = data1.drop(columns = "Date")

    # clean data values to be floats
    data1['Close/Last'] = data1['Close/Last'].str.replace('$', '', regex = True)
    data1['Open'] = data1['Open'].str.replace('$', '', regex = True)
    data1['High'] = data1['High'].str.replace('$', '', regex = True)
    data1['Low'] = data1['Low'].str.replace('$', '', regex = True)

    # convert to numpy
    data1 = data1.to_numpy()
    data1 = np.array(data1, dtype = np.float32)

    # TODO: Write code given the Input / Output Paramters.

    # antiquated I know but so I can use numpy for speed
    if attribute == "close":
        attribute = 0

    elif attribute == "volume":
        attribute = 1

    elif attribute == "open":
        attribute = 2

    elif attribute == "high":
        attribute = 3

    elif attribute == "low":
        attribute = 4

    amin = np.full(data1[:, attribute].shape, np.amin(data1[:, attribute]), dtype = np.float32)
    amax = np.full(data1[:, attribute].shape, np.amax(data1[:, attribute]), dtype = np.float32)
    ones = np.ones(data1[:, attribute].shape)
    zeroes = np.zeros(data1[:, attribute].shape)

    return ((data1[:, attribute] - amin[:]) / (amax[:] - amin[:])) * (ones[:] - zeroes[:])

def zscore_normalization(fileName, attribute):
    '''
    Input Parameters:
        fileName: The comma seperated file that must be considered for the normalization with the different format first line removed
        attribute: The attribute for which you are performing the normalization

    Output:
        Return an object of list type of normalized values
        Please do not return a list of strings
        Please do not print anything to stdout
        Use only Python3
        Points will be deducted if you do not follow exact instructions
    '''

    data1 = pd.read_csv(fileName, header = 0)
    data1 = data1.drop(columns = "Date")

    # clean data values to be floats
    data1['Close/Last'] = data1['Close/Last'].str.replace('$', '', regex = True)
    data1['Open'] = data1['Open'].str.replace('$', '', regex = True)
    data1['High'] = data1['High'].str.replace('$', '', regex = True)
    data1['Low'] = data1['Low'].str.replace('$', '', regex = True)

    # convert to numpy
    data1 = data1.to_numpy()
    data1 = np.array(data1, dtype = np.float32)

    # TODO: Write code given the Input / Output Paramters.

    # antiquated I know but so I can use numpy for speed
    if attribute == "close":
        attribute = 0

    elif attribute == "volume":
        attribute = 1

    elif attribute == "open":
        attribute = 2

    elif attribute == "high":
        attribute = 3

    elif attribute == "low":
        attribute = 4

    mean = np.full(data1[:, attribute].shape, np.mean(data1[:, attribute]), dtype = np.float32)
    std = np.full(data1[:, attribute].shape, np.std(data1[:, attribute]), dtype = np.float32)

    return (data1[:, attribute] - mean[:]) / std[:]



### Testing ###
print(correlation("high", "adsk.csv", "volume", "adsk.csv"))
print(min_max_normalization("adsk.csv", "close"))
print(zscore_normalization("adsk.csv", "open"))

### Plots ###
filename = sys.argv[-1]
data1 = pd.read_csv(filename, header = 0)

data1 = data1.drop(columns = "Date")

# clean data values to be floats
data1['Close/Last'] = data1['Close/Last'].str.replace('$', '', regex = True)
data1['Open'] = data1['Open'].str.replace('$', '', regex = True)
data1['High'] = data1['High'].str.replace('$', '', regex = True)
data1['Low'] = data1['Low'].str.replace('$', '', regex = True)

data = data1.to_numpy()
data = np.array(data, dtype = np.float32)
n = len(data[:, 3])
end = 2022 * 365 - 162
start = end - n

time = np.arange(start, end, 1)
time = time[:] / np.full(n, 365)

plt.plot(time, data[:, 3])
plt.plot(time, data[:, 4])
plt.xlabel("time (years)")
plt.ylabel("price ($)")
plt.xticks([2018, 2019, 2020, 2021, 2022])
plt.title("Home Depot High and Low Stock Prices for 5 years")
plt.legend(["High", "Low"])

plt.show()

my_dict = {'Open': data[:, 2], 'Close/Last': data[:, 0]}

fig, ax = plt.subplots()
ax.boxplot(my_dict.values())
ax.set_xticklabels(my_dict.keys())
plt.title("Home Depot Open and Close Price box plots")

plt.show()

plt.hist(data[:, 1], bins = 10)
plt.xlabel("stock volume (10 millions)")
plt.title("Histogram of Volume of Home Depot Held")
plt.show()

plt.plot(time[:time.shape[0]], zscore_normalization("hd.csv", "high"))
plt.plot(time[:time.shape[0]], zscore_normalization("hd.csv", "volume"))
plt.title("Z-score Standardized Volume and High over time")
plt.xticks([2018, 2019, 2020, 2021, 2022])
plt.xlabel("time (years)")
plt.ylabel("z-score")
plt.legend(["High", "Volume"])
plt.show()
