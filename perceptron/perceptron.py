from matplotlib import pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
import pandas as pd

# 导入数据
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['label'] = iris.target
df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']

plt.scatter(df[:50]['sepal length'], df[:50]['sepal width'], c='red', label='0')
plt.scatter(df[50:100]['sepal length'], df[50:100]['sepal width'], c='green', label='1')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend()
plt.show()

# 取两列数据并将标签转化为1/-1
data = np.array(df.iloc[:100, [0, 1, -1]])
X, y = data[:,:-1], data[:,-1]
y = np.array([1 if i == 1 else -1 for i in y])


# 初始化权重和偏移
def initialize(dim):
    w = np.zeros(dim)
    b = 0.0
    return w, b


# 定义符号函数
def sign(x, w, b):
    return np.dot(x, w) + b


# 训练函数
def train(X_train, y_train, lr):
    w, b = initialize(X_train.shape[1])
    is_wrong = False

    while not is_wrong:
        wrong_count = 0
        for i in range(len(X_train)):
            X = X_train[i]
            y = y_train[i]
            # 有误分类点更新参数，直到没有
            if y * sign(X, w, b) <= 0:
                w = w + lr*np.dot(y, X)
                b = b + lr*y
                wrong_count += 1
        if wrong_count == 0:
            is_wrong = True

        params = {'w': w, 'b': b}
    return params


params = train(X, y, 0.01)
print(params)


# 结果可视化
x_points = np.linspace(4, 7, 10)
y_hat = -(params['w'][0]*x_points + params['b'])/params['w'][1]
plt.plot(x_points, y_hat)

plt.plot(data[:50, 0], data[:50, 1], color='red', label='0')
plt.plot(data[50:100, 0], data[50:100, 1], color='green', label='1')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend()
plt.show()
