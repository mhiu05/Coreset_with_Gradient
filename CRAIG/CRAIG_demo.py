import numpy as np
from scipy.spatial import ConvexHull
import time

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

def run_simulation(num_points, dimension, k):
    print(f"\n=== Simulation for {dimension}D ===")
    np.random.seed(dimension)
    points = np.random.uniform(0.5, 3.0, size=(num_points, dimension))
    G = np.sum(points, axis=0)
    print(f"\nGradient tổng G = {G}\n")

    # CRAIG gốc
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

    # Random chọn k điểm
    print("=== Random chọn k điểm ===")
    t0 = time.time()
    random_indices = np.random.choice(num_points, k, replace=False)
    G_approx_random = np.sum(points[random_indices], axis=0)
    residual_random = G - G_approx_random
    t1 = time.time()
    print(f"S: {[i+1 for i in random_indices]}")
    print(f"G_approx = {G_approx_random}")
    print(f"Residual R = {residual_random}")
    relative_error_random = np.linalg.norm(residual_random) / np.linalg.norm(G)
    print(f"Relative error random: {relative_error_random:.6f}")
    print(f"Thời gian random: {t1 - t0:.4f} giây\n")

    # CRAIG trên Convex Hull (không có mean)
    print("=== CRAIG trên Convex Hull (không có mean) ===")
    try:
        t0 = time.time()
        hull = ConvexHull(points)
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

if __name__ == "__main__":
    num_points = 100000
    k = 10

    run_simulation(num_points, 2, k)
    run_simulation(num_points, 3, k)
    run_simulation(num_points, 4, k)