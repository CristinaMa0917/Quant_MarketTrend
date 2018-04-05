from spectrum import arma2psd, arburg # pip install spectrum
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
data = pd.read_csv('close500m.csv')
data = data.drop(np.where(data.iloc[:,1] == 0)[0])
print(data.head())
#--------------------------
