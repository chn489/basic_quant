import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

df = pd.read_csv('D:/desktop/FDATA/dataset600000.csv')
pe = np.array(df['pe_ratio'].dropna()).reshape(-1, 1)
price = np.array(df['price'].dropna())
ret = np.empty(26)
for i in range(len(price) - 1):
    ret[i] = 100 * (price[i + 1] / price[i] - 1)
# print(ret)
lrg = LinearRegression()
lrg.fit(pe, ret)
print('coefficients=', lrg.coef_, ' ', 'intercept=', lrg.intercept_, '\n')
score = lrg.score(pe, ret)
print(score)
plt.plot(pe, label='pe')
plt.plot(ret, label='ret')
plt.legend()
plt.show()
