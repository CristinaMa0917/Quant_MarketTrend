import pandas as pd
import numpy as np
import os

def preprocess():
    pwd = os.getcwd()
    data = pd.read_csv(os.path.join(pwd, "csg_000300_15m.csv"), encoding="gb2312")
    data = data.drop(np.where((data["volume"] == 0) | (data["close"] == 0))[0])
    data = data.iloc[-4000:,:]
    return data

def addlabel(data):
    label = np.sign(np.diff(data["close"].values))
    data = data.iloc[:-1,:]
    data["label"] = label
    return data

def identifyShock(): # 返回震荡和趋势点对应的下标
    data = preprocess()
    data = addlabel(data)
    data["direction"] = 0
    i = 0
    while i < len(data):
        if data["label"].values[i] == 0:
            k = i + 1
            while k < len(data):
                if data["label"].values[k] != 0:
                    direction = data["label"].values[k]
                    break
                k += 1
        else:
            direction = data["label"].values[i]

        j = i + 1
        while j < len(data):
            if data["label"].values[j] == direction or data["label"].values[j] == 0:
                j += 1
            elif direction == 1 and j < len(data)-9 and np.max(data["close"].values[j+1:j+9]) > data["close"].values[j]:
                j += 1
            elif direction == -1 and j < len(data)-9 and np.min(data["close"].values[j+1:j+9]) < data["close"].values[j]:
                j += 1
            else:
                break

        ##效率E：位移/路程
        if j == len(data):
            E = np.abs((data["close"].values[j-1] - data["close"].values[i])) / np.sum(np.abs(np.diff(data["close"].values[i:j])))
        else:
            E = np.abs((data["close"].values[j] - data["close"].values[i])) / np.sum(np.abs(np.diff(data["close"].values[i:j+1])))

        ##波段波动rtn
        if j == len(data):
            rtn = direction * (data["close"].values[j-1] - data["close"].values[i]) / data["close"].values[i]
        else:
            rtn = direction * (data["close"].values[j] - data["close"].values[i]) / data["close"].values[i]
        #if rtn > 0.012 and E > 0.36 and j - i > 3:
        if rtn > 0.012 and E > 0.4 and j - i > 3:
            data["direction"].values[range(i,j)] = 1
    #添加路径有效率来更好的划分趋势与震荡
        i = j

    data = data.reset_index().reset_index()
    return data[data["direction"]==0]['level_0'],data[data["direction"]==1]['level_0']

def accuracyCal(trend):
    shockReal,trendReal = identifyShock()
    trendReal = set(tuple(trendReal))
    trend = set(tuple(trend))
    print("实际可交易点总数："+str(len(trendReal)))
    print("预测可交易点总数：" + str(len(trend)))
    accuracy = len(trendReal&trend)/len(trend)*100
    return accuracy

shockReal,t = identifyShock()
print(len(shockReal))
print(len(t))