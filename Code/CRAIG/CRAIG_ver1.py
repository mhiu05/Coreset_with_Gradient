import numpy as np
from scipy.spatial import ConvexHull
import time

from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

import pandas as pd
df = pd.read_csv(r"D:\ICPC_TOEIC_CORESET\Coreset_with_Gradient\Code\CRAIG\data\teen_phone_addiction_dataset.csv")
df.head()

import matplotlib.pyplot as plt
plt.hist(df['Phone_Checks_Per_Day'], bins=20, edgecolor='black', color='skyblue')
plt.title('Distribution of Phone Checks Per Day')
plt.xlabel('Số lần kiểm tra điện thoại mỗi ngày')
plt.ylabel('Frequency (Số lượng)')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

plt.hist(df['Age'], bins=20, edgecolor='black', color='skyblue')
plt.title('Distribution of Age')
plt.xlabel('Tuổi')
plt.ylabel('Frequency (Số lượng)')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

def craig_greedy(points, G, k, indices=None):
    R = G.copy()
    G_approx = np.zeros_like(G)
    S = []
    w = {}
    n = len(points)
    candidate_indices = indices if indices is not None else list(range(n))

    for _ in range(k):
        best_score, best_idx = -np.inf, None
        for i in candidate_indices:
            if i in S:
                continue
            g = points[i]
            dot = R @ g
            norm2 = g @ g
            wi = dot / norm2 if norm2 != 0 else 0
            score = wi * dot
            if score > best_score:
                best_score = score
                best_idx = i
        if best_idx is None:
            break
        g = points[best_idx]
        norm2 = g @ g
        wi = R @ g / norm2 if norm2 != 0 else 0
        S.append(best_idx)
        w[best_idx] = wi
        G_approx += wi * g
        R = G - G_approx
    return S, w, G_approx, R

def CRAIG(points, dimension, k):
    print(f"\n=== Simulation for {dimension}D ===")
    G = np.sum(points, axis=0)
    print(f"\nGradient tổng G = {G}\n")

    print("=== CRAIG gốc ===")

    t0 = time.time()
    S, w, G_approx, R = craig_greedy(points, G, k)
    t1 = time.time()

    print(f"Coreset S = {[i+1 for i in S]}")
    print(f"Trọng số w = {{", end="")
    for idx in S:
        print(f"{idx+1}: {w[idx]:.4f}, ", end="")
    print("}")
    print(f"G_approx = {G_approx}")
    print(f"Residual R = {R}")
    relative_error = np.linalg.norm(R) / np.linalg.norm(G)
    print(f"Relative error CRAIG gốc: {relative_error:.6f}")
    print(f"Thời gian CRAIG gốc: {t1 - t0:.4f} giây\n")

def random(points, dimension, k):
    print(f"\n=== Simulation for {dimension}D ===")
    G = np.sum(points, axis=0)
    print(f"\nGradient tổng G = {G}\n")

    print("=== Random chọn k điểm ===")

    t0 = time.time()
    random_indices = np.random.choice(points.shape[0], k, replace=True)
    G_approx_random = np.sum(points[random_indices], axis=0)
    residual_random = G - G_approx_random
    t1 = time.time()

    print(f"S: {[i+1 for i in random_indices]}")
    print(f"G_approx = {G_approx_random}")
    print(f"Residual R = {residual_random}")
    relative_error_random = np.linalg.norm(residual_random) / np.linalg.norm(G)
    print(f"Relative error random: {relative_error_random:.6f}")
    print(f"Thời gian random: {t1 - t0:.4f} giây\n")

def convex_hull(points, dimension, k):
    print(f"\n=== Simulation for {dimension}D ===")
    G = np.sum(points, axis=0)
    print(f"\nGradient tổng G = {G}\n")

    print("=== CRAIG trên Convex Hull (không có mean) ===")

    try:
        t0 = time.time()
        hull = ConvexHull(points, qhull_options='QJ')

        hull_indices = list(hull.vertices)
        S_hull, w_hull, G_approx_hull, R_hull = craig_greedy(points, G, k, indices=hull_indices)
        t1 = time.time()

        print(f"Coreset S = {[i+1 for i in S_hull]}")
        print(f"Trọng số w = {{", end="")
        for idx in S_hull:
            print(f"{idx+1}: {w_hull[idx]:.4f}, ", end="")
        print("}")
        print(f"G_approx = {G_approx_hull}")
        print(f"Residual R = {R_hull}")
        relative_error_hull = np.linalg.norm(R_hull) / np.linalg.norm(G)
        print(f"Relative error CRAIG Convex Hull: {relative_error_hull:.6f}")
        print(f"Thời gian CRAIG Convex Hull: {t1 - t0:.4f} giây\n")
    except Exception as e:
        print(f"Convex Hull calculation failed for {dimension}D: {e}")

def sigmoid(x):
    return 1. / (1 + np.exp(-x))

def softmax(x):
    e = np.exp(x - np.max(x))  # prevent overflow
    if e.ndim == 1:
        return e / np.sum(e, axis=0)
    else:
        return e / np.array([np.sum(e, axis=1)]).T  # ndim = 2


class LogisticRegression(object):
    def __init__(self, dim, num_class):
        self.binary = num_class == 1
        self.W = np.zeros((dim, num_class))
        self.b = np.zeros(num_class)
        self.params = [self.W, self.b]

    # σ(z) = 1 / (1 + e^(-z)) , với z = X·W + b
    def activation(self, input, params=None):
        W, b = params if params is not None else self.params
        if self.binary:
            return sigmoid(np.dot(input, W) + b)
        else:
            return softmax(np.dot(input, W) + b)

    # loss function = -1/N * sum(y * log(σ(z)) + (1 - y) * log(1 - σ(z))
    def loss(self, input, label, l2_reg=0.00, params=None):
        sigmoid_activation = self.activation(input, params)
        cross_entropy = - np.mean(np.sum(label * np.log(sigmoid_activation) + (1 - label) * np.log(1 - sigmoid_activation), axis=1))
        return cross_entropy + l2_reg * np.linalg.norm(self.W) ** 2 / 2

    def predict(self, input, params=None):
        return self.activation(input, params)

    def accuracy(self, input, label, params=None):
        if self.binary:
            return np.mean(np.isclose(np.rint(self.predict(input, params)), label))
        else:
            return np.mean(np.argmax(self.predict(input, params), axis=1) == np.argmax(label, axis=1))

    def gradient(self, input, label, l2_reg=0.00):
        p_y_given_x = self.activation(input)
        d_y = label - p_y_given_x  # shape (n_samples, 1)

        # Tính d_W đúng cách (shape phải khớp với W)
        d_W = -np.dot(input.T, d_y) / input.shape[0]  # shape (n_features, 1)
        d_W += l2_reg * self.W  # Thêm regularization

        d_b = -np.mean(d_y)  # scalar
        return d_W, d_b
    
X = df[['Age', 'Academic_Performance']].values  # shape (n_samples, 2)
y_raw = df['Phone_Checks_Per_Day'].values

# Chuẩn hóa X
# X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

# Chuyển y thành binary: 1 nếu lớn hơn trung bình, 0 nếu ngược lại
y_mean = np.mean(y_raw)
y = (y_raw > y_mean).astype(int)  # shape (n,)

dim = X.shape[1]  # số feature
model = LogisticRegression(dim=dim, num_class=1)  # binary

# Huấn luyện đơn giản (với số epochs và learning rate cố định)
learning_rate = 0.1
epochs = 1000
for epoch in range(epochs):
    # Tính gradient cho toàn bộ dữ liệu (batch gradient descent)
    gradients = model.gradient(X, y.reshape(-1, 1))  # y cần reshape thành (n,1)
    d_W, d_b = gradients
    # Cập nhật tham số
    model.W -= learning_rate * d_W
    model.b -= learning_rate * d_b

    # In loss mỗi 100 epochs
    if epoch % 100 == 0:
        loss = model.loss(X, y.reshape(-1, 1))
        print(f"Epoch {epoch}, Loss: {loss}")

# Lấy W và b từ mô hình
W = model.W
b = model.b
print(f"Trained weights: W={W}, b={b}")

print(model.accuracy(X, y))

def compute_individual_gradients(X, y, W, b):
    y = y.reshape(-1, 1) if y.ndim == 1 else y
    z = np.dot(X, W) + b
    p = 1 / (1 + np.exp(-z))
    error = (p - y)  # shape (n_samples, 1)

    # Tính gradient cho từng điểm: [error_i * X_i]
    individual_grads = error * X  # shape (n_samples, n_features)

    return individual_grads  # Trả về ma trận (n_samples, 2)

if __name__ == "__main__":
    # Đảm bảo X có shape (n_samples, n_features)
    # y có shape (n_samples, 1)
    # W có shape (n_features, 1)
    # b là scalar

    num_points = compute_individual_gradients(X, y, W, b)
    k = 10

    CRAIG(num_points, X.shape[1], k)  # dimension = số features

    random(num_points, X.shape[1], k)

    convex_hull(num_points, X.shape[1], k)
