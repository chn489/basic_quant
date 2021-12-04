import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, LassoCV
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# df=pd.read_excel('D:\\desktop\\10支票_tick_data.xlsx',sheet_name='平安银行')
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


def power(lis, n):
    lis_pow = []
    for i in range(len(lis)):
        lis_pow.append(lis[i] ** n)
    return lis_pow


# 准备最终数据，构建训练集和测试集
price3 = rol_mean(price, 3)
price3_sq = power(price3, 2)
price3_th = power(price3, 3)
price3_fo = power(price3, 4)
price1200 = rol_mean(price, 1200)
price1200_sq = power(price1200, 2)
feature = np.vstack(
    (price3, price3_sq, price3_th, price3_fo, price1200, price1200_sq)).transpose()
feature_train, feature_test, price_train, price_test = train_test_split(feature, price, test_size=0.2, random_state=1)
# print(feature.shape, price.shape)
"""
file = open('feature.csv', 'w', encoding='utf-8-sig')
for i in range(len(price)):
    for j in range(6):
        file.write(str(feature[i][j])+', ')
    file.write('\n')
file.close()
"""
# 构建LASSO回归模型
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
plt.plot(pred, label='pred')
plt.plot(price_test, label='price_test')
plt.legend()
plt.show()

# 预测未来30个tick的数据
next_target = np.array(feature[-1199:-1])
tar_price = []
res = []
for i in range(-1199, 0, 1):
    tar_price.append(price[i])
file = open('predict_price.csv', 'w', encoding='utf-8-sig')
for i in range(30):
    result = lasso.predict([next_target[-1]])
    res.append(result)
    file.write(str(result) + '\n')
    tar_price = np.append(tar_price, result)
    tar_price3 = rol_mean(tar_price, 3)
    tar_price3_sq = power(tar_price3, 2)
    tar_price3_th = power(tar_price3, 3)
    tar_price3_fo = power(tar_price3, 4)
    tar_price1200 = rol_mean(tar_price, 1200)
    tar_price1200_sq = power(tar_price1200, 2)
    next_target = np.vstack((next_target, [tar_price3[-1], tar_price3_sq[-1], tar_price3_th[-1],
                                           tar_price3_fo[-1], tar_price1200[-1], tar_price1200_sq[-1]]))
file.close()
plt.plot(res, label='pred_price_30')
plt.xlabel('ticks')
plt.ylabel('price')
plt.legend()
plt.show()
