
"""
    Copyright © 2025, Nguyen Kieu Linh and Hoang Xuan Phu
    This code was implemented for the paper titled "Algorithms for finding 
    a desired number of vertices ofthe convex hull of finite sets"
"""

import random
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from numpy.linalg import norm, pinv, matrix_rank
from scipy.linalg import null_space


# ==============================================================================
# I. CÁC HÀM PHỤ TRỢ CHUNG 
# ==============================================================================

def partition_points(P: np.ndarray, p_indices_to_partition: list[int], simplex_indices: list[int]) -> list[list[int]]:
    """
    Hàm phân vùng điểm ( Wang 2013 Algorithm 2)
    :param P: Ma trận tất cả các điểm trong tập dữ liệu.
    :param p_indices_to_partition: Các chỉ số của điểm cần phân vùng.
    :param simplex_indices: Các chỉ số của d+1 đỉnh tạo thành simplex.
    """
    num_vertices = len(simplex_indices)
    d = P.shape[1] 
    if num_vertices != d + 1:
        raise ValueError("Partitioning requires a simplex with d+1 vertices.")

    V_simplex_matrix = P[simplex_indices, :] 
    parts = [[] for _ in range(num_vertices)] 
    def barycentric_coords_full(p_vec, V_mat):
        A_bary = np.vstack((V_mat.T, np.ones(d + 1)))
        b_bary = np.append(p_vec, 1)

        if matrix_rank(A_bary) < d + 1:
            return np.full(d + 1, np.nan)
     
        return pinv(A_bary) @ b_bary

  
    for p_idx in p_indices_to_partition:
        p = P[p_idx, :] 
        α = barycentric_coords_full(p, V_simplex_matrix) 
        if np.isnan(α).any():
            continue

        min_val = np.min(α) 
        min_idx = np.argmin(α) 
        if min_val < -1e-9:
            parts[min_idx].append(p_idx)
            
    return parts 

def get_facets_indices(simplex_indices: list[int]) -> list[list[int]]:
    """
    Hàm lấy các chỉ số đỉnh của các mặt của một simplex
    :param simplex_indices: Các chỉ số của d+1 đỉnh tạo thành simplex.
    """
    d1 = len(simplex_indices) 
    facets = [] 
    for i in range(d1):
 
        facet_vertices = simplex_indices[:i] + simplex_indices[i+1:]
        facets.append(facet_vertices) 
    return facets 

def dist_point_to_linear_subspace(p_vec: np.ndarray, basis_vectors_matrix: np.ndarray) -> float:
    """
    Hàm tính khoảng cách từ một điểm đến một không gian con tuyến tính
    (Dựa trên Wang 2013, Equation 2. Cần cho select_initial_simplex_Wang2013)
    :param p_vec: Vector điểm (u trong bài báo)
    :param basis_vectors_matrix: Ma trận các vector cơ sở của không gian con (U trong bài báo), mỗi cột là một vector cơ sở.
    """
    t = basis_vectors_matrix.shape[1] 
    if t == 0:
        return norm(p_vec)

    Q = basis_vectors_matrix.T @ basis_vectors_matrix 
    c = basis_vectors_matrix.T @ p_vec

    a_star = pinv(Q) @ c

    dist_sq = p_vec @ p_vec - c @ a_star
    
    return np.sqrt(max(0.0, dist_sq))

def compute_distance_to_hyperplane_ding2017(x: np.ndarray, S_points_matrix: np.ndarray) -> float:
    """
    Hàm tính khoảng cách từ một điểm đến một siêu phẳng/bao lồi
    (Dựa trên Algorithm 2 Computation of the Distance Between a Point and a Hyperplane của Ding 2017)
    :param x: Điểm cần tính khoảng cách.
    :param S_points_matrix: Ma trận các điểm định nghĩa siêu phẳng/bao lồi (S trong Algorithm 2), mỗi hàng là một điểm.
    """
    n_S, d = S_points_matrix.shape 

    if n_S < d:
        return np.min([norm(x - S_points_matrix[i,:]) for i in range(n_S)])

    V_hyperplane_for_normal = S_points_matrix[:d, :]
    p1_normal = V_hyperplane_for_normal[0, :] 
    basis_vectors_normal = V_hyperplane_for_normal[1:, :] - p1_normal
 
    if matrix_rank(basis_vectors_normal) < d - 1:
        return np.min([norm(x - S_points_matrix[i,:]) for i in range(n_S)])

    ns = null_space(basis_vectors_normal.T) 
    if ns.shape[1] == 0:
        return np.min([norm(x - S_points_matrix[i,:]) for i in range(n_S)])
    beta = ns[:, 0] 

    x0_center_vec = np.mean(S_points_matrix, axis=0)

    z0 = x - x0_center_vec 

    k_positive_dot = 0  
    for i in range(n_S):
        zi = S_points_matrix[i, :] - x0_center_vec
        if np.dot(z0, zi) >= 0: 
            k_positive_dot += 1

    if k_positive_dot < n_S:
        return np.min([norm(x - S_points_matrix[i,:]) for i in range(n_S)])
    else:
        x1_from_S = S_points_matrix[0, :]

        norm_beta = norm(beta)
        if norm_beta < 1e-12: 
            return np.inf 
        return np.abs(np.dot(beta, x - x1_from_S)) / norm_beta

def generate_random_points_in_hypercube(n: int, d: int, range_min: float = 0.0, range_max: float = 1.0) -> np.ndarray:
    """
    Hàm tạo dữ liệu ngẫu nhiên trong siêu hộp
    :param n: Số lượng điểm
    :param d: Số chiều
    :param range_min, range_max: Khoảng giá trị cho mỗi chiều (ví dụ: [0, 1] cho siêu hộp đơn vị)
    """
    return np.random.uniform(range_min, range_max, (n, d))

# ==============================================================================
# II. CHỌN SIMPLEX BAN ĐẦU (Wang 2013 Algorithm 1)
# ==============================================================================

def select_initial_simplex_Wang2013_sequential(P: np.ndarray, d: int) -> list[int]:
    """
    Hàm: Chọn d+1 đỉnh của một d-simplex theo Algorithm 1 của Wang (2013) - PHIÊN BẢN TUẦN TỰ
    :param P: Ma trận tất cả các điểm trong tập dữ liệu.
    :param d: Số chiều không gian.
    """
    n = P.shape[0]
    rand_idx = random.randint(0, n - 1)
    x0 = P[rand_idx, :]

    dists_x0 = norm(P - x0, axis=1)
    xj0_idx = np.argmax(dists_x0)
    xj0 = P[xj0_idx, :]

    dists_xj0 = norm(P - xj0, axis=1)
    xj1_idx = np.argmax(dists_xj0)

    S_t_indices = {xj0_idx, xj1_idx} 
    
    tilde_U_all = P - xj0
    
    tilde_U_S_t_basis = tilde_U_all[xj1_idx, :].reshape(-1, 1)

    t = 1 

    while t < d:
        all_candidate_dists = np.full(n, -np.inf)
        
        for i in range(n):
            if i not in S_t_indices: 
                u_i = tilde_U_all[i, :] 
                all_candidate_dists[i] = dist_point_to_linear_subspace(u_i, tilde_U_S_t_basis)
        
        max_dist_subspace = np.max(all_candidate_dists)
        if max_dist_subspace == -np.inf:
            break
        best_idx_subspace = np.argmax(all_candidate_dists)

        S_t_indices.add(best_idx_subspace)
        tilde_U_S_t_basis = np.hstack((tilde_U_S_t_basis, tilde_U_all[best_idx_subspace, :].reshape(-1, 1)))
        t += 1
    
    if len(S_t_indices) != d + 1:
        warnings.warn(f"Could not select d+1 independent vertices. Selected {len(S_t_indices)} instead.")

    return list(S_t_indices)

def select_initial_simplex_Wang2013_parallel(P: np.ndarray, d: int) -> list[int]:
    """
    Hàm: Chọn d+1 đỉnh của một d-simplex theo Algorithm 1 của Wang (2013) - PHIÊN BẢN SONG SONG
    :param P: Ma trận tất cả các điểm trong tập dữ liệu.
    :param d: Số chiều không gian.
    """
    n = P.shape[0]

    rand_idx = random.randint(0, n - 1)
    x0 = P[rand_idx, :]
    dists_x0 = norm(P - x0, axis=1)
    xj0_idx = np.argmax(dists_x0)
    xj0 = P[xj0_idx, :]
    dists_xj0 = norm(P - xj0, axis=1)
    xj1_idx = np.argmax(dists_xj0)

    S_t_indices = {xj0_idx, xj1_idx}
    tilde_U_all = P - xj0
    tilde_U_S_t_basis = tilde_U_all[xj1_idx, :].reshape(-1, 1)
    t = 1

    while t < d:
        all_candidate_dists = np.full(n, -np.inf)
        candidate_indices = [i for i in range(n) if i not in S_t_indices]

        with ThreadPoolExecutor() as executor:
            future_to_idx = {executor.submit(dist_point_to_linear_subspace, tilde_U_all[i, :], tilde_U_S_t_basis): i for i in candidate_indices}
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    dist = future.result()
                    all_candidate_dists[idx] = dist
                except Exception as exc:
                    print(f'Point index {idx} generated an exception: {exc}')

        max_dist_subspace = np.max(all_candidate_dists)
        if max_dist_subspace == -np.inf:
            break
        best_idx_subspace = np.argmax(all_candidate_dists)

        S_t_indices.add(best_idx_subspace)
        tilde_U_S_t_basis = np.hstack((tilde_U_S_t_basis, tilde_U_all[best_idx_subspace, :].reshape(-1, 1)))
        t += 1
    
    if len(S_t_indices) != d + 1:
        warnings.warn(f"Could not select d+1 independent vertices. Selected {len(S_t_indices)} instead.")

    return list(S_t_indices)

# ==============================================================================
# III. THUẬT TOÁN CHVS4 (Dựa trên Algorithm 3 CHVS4 của Ding 2017)
# ==============================================================================

def CHVS4_Algorithm3_Ding2017(P: np.ndarray, epsilon: float = 0.0) -> np.ndarray:
    """
    P: Ma trận tất cả các điểm trong tập dữ liệu.
    epsilon: Ngưỡng dừng cho bao lồi xấp xỉ.
    """
    n, d = P.shape

    rand_idx_x0 = random.randint(0, n - 1)
    x0_point = P[rand_idx_x0, :]
    dists_x0 = norm(P - x0_point, axis=1)
    xj0_idx = np.argmax(dists_x0)
    dists_xj0 = norm(P - P[xj0_idx, :], axis=1)
    xj1_idx = np.argmax(dists_xj0)

    final_V_indices = set()

    initial_simplex_vertices_indices = select_initial_simplex_Wang2013_sequential(P, d)
    final_V_indices.update(initial_simplex_vertices_indices)

    candidate_indices_for_partition = [i for i in range(n) if i not in final_V_indices]
    
    P_oc_list = []
    S_oc_list = []

    initial_facets = get_facets_indices(initial_simplex_vertices_indices)
    initial_parts_by_facet = partition_points(P, candidate_indices_for_partition, initial_simplex_vertices_indices)


    m_i_values = []
    x_i_star_indices = []

    for i in range(len(initial_facets)):
        current_facet_indices = initial_facets[i]
        current_part_indices = initial_parts_by_facet[i]
        
        if current_part_indices:
            P_oc_list.append(current_part_indices)
            S_oc_list.append(current_facet_indices)

            max_dist_in_part = -1.0
            best_point_in_part_idx = -1
            facet_matrix = P[current_facet_indices, :]
            for p_idx in current_part_indices:
                dist = compute_distance_to_hyperplane_ding2017(P[p_idx, :], facet_matrix)
                if dist > max_dist_in_part:
                    max_dist_in_part = dist
                    best_point_in_part_idx = p_idx
            
            m_i_values.append(max_dist_in_part)
            x_i_star_indices.append(best_point_in_part_idx)


    m = max(m_i_values) if m_i_values else -np.inf
    active_problem_indices = set(range(len(P_oc_list)))
    
    max_iterations = 2 * n
    current_iteration = 0


    while m > epsilon and active_problem_indices and current_iteration < max_iterations:
        current_iteration += 1

        active_m_tuples = [(m_i_values[i], i) for i in active_problem_indices]
        _, j0_problem_idx = max(active_m_tuples, key=lambda item: item[0])

        x_j0_star_idx = x_i_star_indices[j0_problem_idx]
        
        final_V_indices.add(x_j0_star_idx)
        active_problem_indices.remove(j0_problem_idx)

        old_part_indices = P_oc_list[j0_problem_idx]
        old_facet_indices = S_oc_list[j0_problem_idx]
        old_part_indices = [p for p in old_part_indices if p != x_j0_star_idx]

        new_simplex_current_indices = old_facet_indices + [x_j0_star_idx]
        new_parts_from_split = partition_points(P, old_part_indices, new_simplex_current_indices)
        all_new_facets_from_split = get_facets_indices(new_simplex_current_indices)

        for new_part_indices, new_facet_indices in zip(new_parts_from_split, all_new_facets_from_split):
            if new_part_indices:
                max_dist_in_new_part = -1.0
                best_point_in_new_part_idx = -1
                new_facet_matrix = P[new_facet_indices, :]
                for p_idx in new_part_indices:
                    dist = compute_distance_to_hyperplane_ding2017(P[p_idx, :], new_facet_matrix)
                    if dist > max_dist_in_new_part:
                        max_dist_in_new_part = dist
                        best_point_in_new_part_idx = p_idx
                
                P_oc_list.append(new_part_indices)
                S_oc_list.append(new_facet_indices)
                m_i_values.append(max_dist_in_new_part)
                x_i_star_indices.append(best_point_in_new_part_idx)
                active_problem_indices.add(len(P_oc_list) - 1)
        if active_problem_indices:
            m = max(m_i_values[i] for i in active_problem_indices)
        else:
            m = -np.inf

    if current_iteration >= max_iterations:
        warnings.warn(f"CHVS4 sequential stopped at max iterations ({max_iterations}). May not have fully converged.")

    return P[list(final_V_indices), :]

# ==============================================================================
# V. VÍ DỤ SỬ DỤNG VÀ KIỂM THỬ
# ==============================================================================
if __name__ == '__main__':

    N_POINTS = 500  
    D_DIMENSIONS = 3   
    
    print(f"Generating {N_POINTS} points in {D_DIMENSIONS} dimensions...")
    points = np.random.randn(N_POINTS, D_DIMENSIONS)
    if D_DIMENSIONS > 1:
        points /= np.linalg.norm(points, axis=1)[:, np.newaxis]
    
    print("\nRunning Sequential CHVS4 Algorithm...")
    import time
    start_time_seq = time.time()
    convex_hull_vertices_seq = CHVS4_Algorithm3_Ding2017(points, epsilon=0.0)
    end_time_seq = time.time()
    print(f"Sequential version found {len(convex_hull_vertices_seq)} vertices.")
    print(f"Time taken (Sequential): {end_time_seq - start_time_seq:.4f} seconds.")


    if D_DIMENSIONS >= 2:
        try:
            import matplotlib.pyplot as plt
            from scipy.spatial import ConvexHull

            # Tính bao lồi thực tế để so sánh
            hull_true = ConvexHull(points)

            if D_DIMENSIONS == 2:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

                ax1.scatter(points[:, 0], points[:, 1], s=10, alpha=0.5, label='All Points')

                if len(convex_hull_vertices_par) >= 3:
                    hull_plot = ConvexHull(convex_hull_vertices_seq)
                    for simplex in hull_plot.simplices:
                        ax1.plot(convex_hull_vertices_seq[simplex, 0], convex_hull_vertices_seq[simplex, 1], 'r-')
                ax1.set_title(f'CHVS4 Result ({len(convex_hull_vertices_seq)} vertices)')
                ax1.legend()
                ax1.set_aspect('equal', adjustable='box')

                ax2.scatter(points[:, 0], points[:, 1], s=10, alpha=0.5, label='All Points')
                for simplex in hull_true.simplices:
                    ax2.plot(points[simplex, 0], points[simplex, 1], 'g-')
                ax2.scatter(points[hull_true.vertices, 0], points[hull_true.vertices, 1], c='g', s=40, label='True Hull Vertices')
                ax2.set_title(f'SciPy ConvexHull Result ({len(hull_true.vertices)} vertices)')
                ax2.legend()
                ax2.set_aspect('equal', adjustable='box')

                plt.show()

            elif D_DIMENSIONS == 3:
                from mpl_toolkits.mplot3d import Axes3D
                fig = plt.figure(figsize=(18, 8))

                ax1 = fig.add_subplot(121, projection='3d')
                ax1.scatter(points[:, 0], points[:, 1], points[:, 2], s=5, alpha=0.3, label='All Points')
                ax1.scatter(convex_hull_vertices_seq[:, 0], convex_hull_vertices_seq[:, 1], convex_hull_vertices_seq[:, 2], c='r', s=30, label='CHVS4 Vertices')
                ax1.set_title(f'CHVS4 Result ({len(convex_hull_vertices_seq)} vertices)')
                ax1.legend()

                ax2 = fig.add_subplot(122, projection='3d')
                ax2.scatter(points[:, 0], points[:, 1], points[:, 2], s=5, alpha=0.3, label='All Points')

                ax2.scatter(points[hull_true.vertices, 0], points[hull_true.vertices, 1], points[hull_true.vertices, 2], c='g', s=30, label='True Hull Vertices')
                ax2.set_title(f'SciPy ConvexHull Result ({len(hull_true.vertices)} vertices)')
                ax2.legend()

                plt.show()

        except ImportError:
            print("\nPlease install matplotlib and scipy to visualize the results: `pip install matplotlib scipy`")

