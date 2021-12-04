import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, LassoCV
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

df = pd.read_excel('D:\\desktop\\平安银行.xlsx')
# 读取列数据并使用array储存
# 需要预测的量
price = np.array(df['price'])


def str2flo(array, lis):  # 将一个由字符串组成的一维列表变成一个float类型的二维列表，即将字符串里面的数字拆出来放到列表里并且转化数据类型
    c = []
    for i in range(len(lis)):
        b = []
        for j in range(5):
            b.append(float(np.array(array[i][j])))
        c.append(b)
    return c


def clean_list(lis):
    arr = []  # 创建一个中间列表
    for num in range(len(lis)):
        lis[num] = lis[num].replace(" ", '')
        lis[num] = lis[num].replace('[', '')
        lis[num] = lis[num].replace(']', '')
        lis[num] = lis[num].split(',')
        arr.append(np.array(lis[num]))
    ar = np.array(arr)
    ar = str2flo(ar, lis)  # 将一个字符串型且数值间带空格的list转化成float类型的二维列表
    ar2 = []
    for i in range(len(lis)):
        ar2.append(np.mean(ar[i]))
    return ar2  # 将二维数组中每行的平均值求出来放到一维列表中


def rol_mean(vec, size):
    vec2 = []
    for i in range(len(vec)):
        if i - 1 < size:
            vec2.append(vec[i])
        else:
            s = 0
            for j in range(i - size - 1, i - 1, 1):
                s += vec[j]
            s = float(s / size)
            vec2.append(s)
    return vec2


# 准备最终数据，构建训练集和测试集
price3 = rol_mean(price, 3)
price20 = rol_mean(price, 20)
price100 = rol_mean(price, 100)
price1200 = rol_mean(price, 1200)
price3600 = rol_mean(price, 3600)
feature = np.vstack(
    (price3, price20, price100, price1200, price3600)).transpose()
feature_train, feature_test, price_train, price_test = train_test_split(feature, price, test_size=0.2, random_state=1)
# print(feature.shape, price.shape)

# 建立LASSO回归模型
Lamdas = np.logspace(-5, 2, 200)
lasso_cv = LassoCV(alphas=Lamdas, normalize=True, max_iter=2000)
lasso_cv.fit(feature_train, price_train)
# print(lasso_cv.alpha_)
lasso = Lasso(alpha=lasso_cv.alpha_, normalize=True, max_iter=2000)
lasso.fit(feature_train, price_train)
print(lasso.intercept_, lasso.coef_)
# 评价模型
pred = lasso.predict(feature_test)
MSE = mean_squared_error(price_test, pred)
print('mse=', MSE)
r2 = r2_score(price_test, pred)
print('r2=', r2)

# 画图分析
plt.plot(pred, label='pred')
plt.plot(price_test, label='price_test')
plt.legend()
plt.show()
