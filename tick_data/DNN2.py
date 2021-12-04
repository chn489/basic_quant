import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import models, layers
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
df = pd.read_excel('D:\\desktop\\平安银行.xlsx')
# 读取列数据并使用array储存
# 需要预测的量
price = np.array(df['price'])

# 可能作为因子的量
bid_price = df['bid_price'].tolist()  # bid_price[num]里面装载着第num个字符串向量，但我们的目标是提取向量中的浮点数值并求平均值
ask_price = df['ask_price'].tolist()


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
bid_price = clean_list(bid_price)
ask_price = clean_list(ask_price)
price3 = rol_mean(price, 3)
price3_sq = power(price3, 2)
price3_th = power(price3, 3)
price3_fo = power(price3, 4)
price1200 = rol_mean(price, 1200)
price1200_sq = power(price1200, 2)
feature = np.vstack(
    (price3, price1200, price1200_sq, bid_price, ask_price)).transpose()
feature_train, feature_test, price_train, price_test = train_test_split(feature, price, test_size=0.2, random_state=1)

model = models.Sequential()
model.add(layers.Dense(128, activation='relu'))
#model.add(layers.Dropout(0.1)) #如果添加dropout，模型预测结果会出现异常结果，这表明少部分神经元对模型效果起决定性作用
model.add(layers.Dense(128, activation='relu'))
#model.add(layers.Dropout(0.1))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(1))
model.compile(optimizer='adam', loss='mse')
history = model.fit(feature_train, price_train, epochs=10, verbose=1, validation_data=(feature_test, price_test))
model.summary()

pred = model.predict(feature_test)
plt.Figure()
plt.plot(pred, label='pred')
plt.plot(price_test, label='price_test')
plt.xlabel('test_num')
plt.ylabel('price')
plt.legend()
plt.show()
