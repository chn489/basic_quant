import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

df = pd.read_csv('D:/desktop/FDATA/dataset600000.csv')
pe = np.array(df['pe_ratio'].dropna()).reshape(-1, 1)
pb = np.array(df['pb_ratio'].dropna()).reshape(-1, 1)
pe_pb = np.array(df.iloc[:, [1, 4]])
# print(pe_pb)
price = np.array(df['price'].dropna()).reshape(-1, 1)
ret = np.empty(26)
for i in range(len(price) - 1):
    ret[i] = (price[i + 1] / price[i] - 1) * 100
# print(ret)
lrg = LinearRegression()
lrg.fit(pe_pb, ret)
score = lrg.score(pe_pb, ret)
print('coefficients=', lrg.coef_, ' ', 'intercept=', lrg.intercept_, '\n')
print('score=', score, '\n')
plt.plot(pe, label='pe')
plt.plot(pb, label='pb')
plt.plot(ret, label='ret')
plt.legend()
plt.show()