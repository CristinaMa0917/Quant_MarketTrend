from accuracyCheck import accuracyCal,identifyShock
from myFunc import burgSimple,lstmPre,burgComplex,bigDiff,burgPeak
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

pwd = os.getcwd()
data_raw = pd.read_csv(os.path.join(pwd, "csg_000300_15m.csv"), encoding="gb2312")
data_raw = data_raw.drop(np.where((data_raw["volume"] == 0) | (data_raw["close"] == 0))[0])
data = data_raw["close"].values.copy()[-4000:]

#for i in [30,40,50,60,70,80]:
    #trend = burgComplex(data,threshold=i)[0]
    #accuracy = accuracyCal(trend)
    #print('threshold(complex way) :%d,then accuracy = %d' %(i,accuracy))

#pred= lstmPre(lb=20)

#for i in np.arange(1,8):#看lstm股票差价阈值对预测结果的影响，其实没什么影响
shockReal = identifyShock()[0]
#shock1 = bigDiff(pred,threshold=i)
# for i in np.arange(1,3,0.1): # 预测下来准确度对参数不敏感，考虑分母差异，根据风险值来选择shock点
shock2 = np.array(burgPeak(data,threshold=2.5)[0])#np.array(burgSimple(data,threshold=60)[0])
shock = list(shock2) # list(shock2) #list(set(shock1)|set(shock2))
#shockReal = shockReal.tolist()
# plot the distribution of prediction and true
index = []
rightP = []
for x in shock :
    if x in shockReal:
        index.append(shock.index(x))
        rightP.append(x)

#---展示正确分类的shock在总预测shock的比重---------------
    #plt.scatter(range(len(shock)),shock,marker='o')
    #plt.scatter(index,rightP,marker='*')
    #plt.title('shockPredict - rightShockPoint')
    #plt.show()

print("实际shock点总数：" + str(len(shockReal)))
print("预测schock点总数：" + str(len(shock)))
accuracy = len(rightP) / len(shock) * 100
print(accuracy)

