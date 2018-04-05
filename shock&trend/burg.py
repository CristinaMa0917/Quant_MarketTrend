from spectrum import arma2psd, arburg # pip install spectrum
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
#---parameters----
N = 40 # 回测的数据量
threshold = 60 #60m 40d
order = 8 # burg的参数
interval = 10 # 阈值浮动的范围

#---input--------
data = pd.read_csv('close300m.csv')
data1 = pd.read_csv('close500d.csv')
data2 = pd.read_csv('close500m.csv')# 股市大跌停牌的一段时间会报错，显示rho为负值，除此之外一切正常
data3 = pd.read_csv('close50030m.csv')

close300m = data.iloc[:,1].ravel()
close500d = data1.iloc[:,1].ravel()
close500m = data2.iloc[200:,1].ravel()
close50030m = data3.iloc[:,1].ravel()

pwd = os.getcwd()
data_raw = pd.read_csv(os.path.join(pwd, "csg_000300_15m.csv"), encoding="gb2312")
data_raw = data_raw.drop(np.where((data_raw["volume"] == 0) | (data_raw["close"] == 0))[0])
data = data_raw["close"].values.copy()[-4000:]
input = data
input1 = (input-min(input))/(max(input)-min(input))

#---iteration----
l = len(input)
fre = np.zeros(N-1).tolist()
print(input.shape)

#input = input1 #是否正则化
for i in range(l-N+1):
    x = input[i:(i+N)]
    AR, rho, ref = arburg(x, order)
    PSD = arma2psd(AR, rho=rho, NFFT=200)
    PSD = PSD[len(PSD):len(PSD) // 2:-1]
    PSD = (10 * np.log10(abs(PSD) * 2. / (2. * np.pi))).tolist()
    PSD = PSD[threshold:]
    index = PSD.index(max(PSD)) + threshold
    fre.append(index)  #认为这个频率就是对应该时点最大的频率

#freCopy = [x for x in fre if x>0]
#minFre = min(freCopy) #排除初始39个零值的fre最小值
selected = [i for i,x in enumerate(fre) if x <= threshold+interval ] #选出fre在一定最小区间的fre下标
tradeP = [input[i] for i in selected] #根据下标选出交易时点，这个时点就是满足条件的趋势点，可交易点

plt.subplot(2,1,1)
plt.scatter(selected,tradeP,s = 10,c='#DC143C',marker='o')
plt.plot(input,c='#0000CD')
plt.grid(True)
plt.title('close price - time(ZZ500 30m)')
plt.subplot(2,1,2)
plt.plot(fre,c='#0000CD')
plt.title('frequency - time')
plt.grid(True, linestyle = "-.")
plt.show()

