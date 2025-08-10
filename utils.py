import numpy as np

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
