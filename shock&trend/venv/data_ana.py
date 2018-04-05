from spectrum import arma2psd, arburg # pip install spectrum
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
data2 = pd.read_csv('close500m.csv')
close500m = data2.iloc[:,1].ravel()
plt.plot(close500m)
plt.show()