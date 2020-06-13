# パーセプトロンAPI
import numpy as np


class Perceptron(object):
    """Perceptron classifier.

    Parameters
    ------------
    eta : float
      Learning rate (between 0.0 and 1.0)
    n_iter : int
      Passes over the training dataset.(トレーニングデータのトレーニング回数)
    random_state : int
      Random number generator seed for random weight
      initialization.（重みを初期化するための乱数シード）

    Attributes（属性）
    -----------
    w_ : 1d-array
      Weights after fitting.
    errors_ : list
      Number of misclassifications (updates) in each epoch.

    """
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """Fit training data.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
          Training vectors, where n_samples is the number of samples and
          n_features is the number of features.
        y : array-like, shape = [n_samples]
          Target values.

        Returns
        -------
        self : object

        """
        # 正規分布の従う乱数
        # numpy.random.normal(平均, 標準偏差, 出力する件数) 
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.errors_ = []

        for _ in range(self.n_iter):   # トレーニング回数分トレーニングデータを反復
            errors = 0
            for xi, target in zip(X, y):  # 各サンプルで重みを更新
                update = self.eta * (target - self.predict(xi))  # 式2.1.5
                self.w_[1:] += update * xi
                # 重みw0の更新
                self.w_[0] += update
                # 更新が0でない場合は誤分類としてカウント
                errors += int(update != 0.0)
            # 反復回数毎の誤差を格納
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        """Calculate net input"""
        # 内積
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, -1)