import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

df = pd.read_csv('D:/desktop/FDATA/dataset600000.csv')
pe = np.array(df['pe_ratio'].dropna()).reshape(-1, 1)
pb = np.array(df['pb_ratio'].dropna()).reshape(-1, 1)
pe_pb = np.array(df.iloc[:, [1, 4]])
# print(pe_pb)
price = np.array(df['price'].dropna()).reshape(-1, 1)
ret=np.empty(26)
clas = np.empty(26,dtype=np.str)

for i in range(len(price) - 1):
    ret[i] = (price[i + 1] / price[i] - 1) * 100
    if ret[i] >= 1:
        clas[i] = 'good'
    elif ret[i] >= 0:
        clas[i] = 'mid'
    else:
        clas[i] = 'bad'
# print(ret)
feature_train, feature_test, target_train, target_test = train_test_split(pe_pb, clas, test_size=0.1,
                                                                          random_state=0)

dt_model = tree.DecisionTreeClassifier(criterion='gini', min_samples_leaf=3)
dt_model.fit(pe_pb, clas)
fig = plt.figure(figsize=(8, 8))
tree.plot_tree(dt_model, filled='True', feature_names=['pe', 'pb'],
               class_names=['good', 'mid', 'bad'])
plt.show()
