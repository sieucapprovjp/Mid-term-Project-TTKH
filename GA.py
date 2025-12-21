import numpy as np
import matplotlib.pyplot as plt


# ==========================================
# 1. ĐỊNH NGHĨA HÀM MỤC TIÊU
# ==========================================
def rosenbrock_func(x_arr):
    x, y = x_arr
    # Giá trị tối ưu (mong muốn) là 0 tại x=1, y=1
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2


def rastrigin_func(x_arr):
    x, y = x_arr
    A = 10
    # Giá trị tối ưu (mong muốn) là 0 tại x=0, y=0
    return 20 + (x ** 2 - A * np.cos(2 * np.pi * x)) + (y ** 2 - A * np.cos(2 * np.pi * y))


# ==========================================
# 2. THUẬT TOÁN GENETIC ALGORITHM (GA)
# ==========================================
def genetic_algorithm(objective_func, start_center, bounds, pop_size=100, generations=300, mutation_rate=0.15,
                      target_fitness=1e-3):
    start_center = np.array(start_center)
    dim = len(start_center)
    bounds = np.array(bounds)

    # Khởi tạo quần thể tập trung quanh điểm start
    pop = start_center + np.random.uniform(-0.5, 0.5, (pop_size, dim))
    pop = np.clip(pop, bounds[:, 0], bounds[:, 1])

    path_best = []

    for gen in range(1, generations + 1):
        fitness = np.array([objective_func(ind) for ind in pop])

        # Tìm cá thể tốt nhất
        best_idx = np.argmin(fitness)
        best_val = fitness[best_idx]
        best_ind = pop[best_idx].copy()
        path_best.append(best_ind)

        # Điều kiện dừng
        if best_val < target_fitness:
            return np.array(path_best), gen, True

            # Tạo thế hệ mới
        new_pop = [best_ind]  # Elitism

        while len(new_pop) < pop_size:
            # Tournament
            idxs = np.random.choice(pop_size, 4, replace=False)
            p1 = pop[idxs[0]] if fitness[idxs[0]] < fitness[idxs[1]] else pop[idxs[1]]
            p2 = pop[idxs[2]] if fitness[idxs[2]] < fitness[idxs[3]] else pop[idxs[3]]

            # Crossover
            child = 0.5 * p1 + 0.5 * p2

            # Mutation
            if np.random.rand() < mutation_rate:
                child += np.random.normal(0, 0.4, dim)

            child = np.clip(child, bounds[:, 0], bounds[:, 1])
            new_pop.append(child)

        pop = np.array(new_pop)

    return np.array(path_best), generations, False


# ==========================================
# 3. HÀM CHẠY VÀ IN KẾT QUẢ CHI TIẾT
# ==========================================
def run_and_report(func_name, func, start_point, bounds, global_min_true, target_fitness):
    print(f"\n{'=' * 25} BÁO CÁO KẾT QUẢ: {func_name.upper()} {'=' * 25}")

    # --- CHẠY THUẬT TOÁN ---
    path, stopped_gen, success = genetic_algorithm(
        func, start_point, bounds,
        generations=300,
        target_fitness=target_fitness
    )

    final_pos = path[-1]
    final_val = func(final_pos)
    ideal_val = 0.0  # Giá trị hoàn hảo của cả 2 hàm này đều là 0

    # --- IN RA MÀN HÌNH (CONSOLE) ---
    print("I. THÔNG SỐ ĐẦU VÀO & MỤC TIÊU:")
    print(f"   - Vị trí xuất phát           : {start_point}")
    print(f"   - Vị trí ĐÍCH (Thực tế)      : {global_min_true}")
    print(f"   - Giá trị mong muốn đạt được : {ideal_val}")

    print("\nII. KẾT QUẢ THỰC TẾ (GA):")
    print(f"   - Vị trí hội tụ tìm được     : [{final_pos[0]:.5f}, {final_pos[1]:.5f}]")
    print(f"   - Giá trị đạt được (Loss)    : {final_val:.8f}")
    print(f"   - Độ lệch so với đích (Error): {abs(final_val - ideal_val):.8f}")
    print(f"   - Số bước lặp (Generations)  : {stopped_gen}")
    print(f"   - Đánh giá trạng thái        : {'HỘI TỤ (Thành công)' if success else 'DỪNG DO HẾT VÒNG LẶP'}")
    print("=" * 70)

    # --- VẼ HÌNH MINH HOẠ ---
    x_range = np.linspace(bounds[0][0], bounds[0][1], 200)
    y_range = np.linspace(bounds[1][0], bounds[1][1], 200)
    X, Y = np.meshgrid(x_range, y_range)
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = func([X[i, j], Y[i, j]])

    plt.figure(figsize=(9, 7))
    plt.contourf(X, Y, Z, levels=60, cmap='viridis', alpha=0.7)
    plt.colorbar(label='Cost Value')

    # Vẽ
    plt.plot(path[:, 0], path[:, 1], 'w.-', linewidth=1, alpha=0.8, label='GA Path')
    plt.plot(start_point[0], start_point[1], 'o', color='lime', markersize=8, markeredgecolor='k', label='Start')
    plt.plot(global_min_true[0], global_min_true[1], '*', color='red', markersize=15, markeredgecolor='w',
             label='True Target')

    plt.title(f"GA Visualization: {func_name}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.show()


# ==========================================
# 4. CHẠY CHƯƠNG TRÌNH
# ==========================================

start_common = [-1.0, 2.0]

# --- BÀI 1: ROSENBROCK ---
run_and_report(
    func_name="Rosenbrock",
    func=rosenbrock_func,
    start_point=start_common,
    bounds=[[-2, 2], [-1, 3]],
    global_min_true=[1, 1],
    target_fitness=0.001
)

# --- BÀI 2: RASTRIGIN ---
run_and_report(
    func_name="Rastrigin",
    func=rastrigin_func,
    start_point=start_common,
    bounds=[[-3, 3], [-3, 3]],
    global_min_true=[0, 0],
    target_fitness=0.01
)