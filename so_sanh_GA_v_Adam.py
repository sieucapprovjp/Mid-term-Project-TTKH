import numpy as np
import matplotlib.pyplot as plt


# ==========================================
# 1. ĐỊNH NGHĨA HÀM MỤC TIÊU & ĐẠO HÀM
# ==========================================
def rosenbrock_func(x_arr):
    x, y = x_arr
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2


def rosenbrock_grad(x_arr):
    x, y = x_arr
    dx = -2 * (1 - x) - 400 * x * (y - x ** 2)
    dy = 200 * (y - x ** 2)
    return np.array([dx, dy])


def rastrigin_func(x_arr):
    x, y = x_arr
    A = 10
    return 20 + (x ** 2 - A * np.cos(2 * np.pi * x)) + (y ** 2 - A * np.cos(2 * np.pi * y))


def rastrigin_grad(x_arr):
    x, y = x_arr
    A = 10
    dx = 2 * x + 2 * np.pi * A * np.sin(2 * np.pi * x)
    dy = 2 * y + 2 * np.pi * A * np.sin(2 * np.pi * y)
    return np.array([dx, dy])


# ==========================================
# 2. ADAM OPTIMIZER (Gradient Based)
# ==========================================
def adam_optimizer(grad_func, x0, learning_rate=0.05, max_iter=1000, tol=1e-5):
    x = np.array(x0, dtype=float)
    dim = len(x)
    m, v = np.zeros(dim), np.zeros(dim)
    history = [x.copy()]

    beta1, beta2, epsilon = 0.9, 0.999, 1e-8

    for t in range(1, max_iter + 1):
        g = grad_func(x)
        # Điều kiện dừng: Gradient quá nhỏ (đã tới đáy hoặc điểm bằng phẳng)
        if np.linalg.norm(g) < tol:
            return x, np.array(history), t, True

        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * (g ** 2)
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        x = x - learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
        history.append(x.copy())

    return x, np.array(history), max_iter, False


# ==========================================
# 3. GENETIC ALGORITHM (Population Based)
# ==========================================
def genetic_algorithm(objective_func, start_center, bounds, pop_size=50, generations=200, mutation_rate=0.1,
                      target_fitness=1e-3):
    start_center = np.array(start_center)
    dim = len(start_center)
    bounds = np.array(bounds)

    # KHỞI TẠO: Tạo đám mây điểm xung quanh start_center (bán kính nhỏ 0.5)
    # Để GA có sự đa dạng ban đầu nhưng vẫn đảm bảo xuất phát cùng vị trí với Adam
    pop = start_center + np.random.uniform(-0.5, 0.5, (pop_size, dim))
    pop = np.clip(pop, bounds[:, 0], bounds[:, 1])

    best_history = []

    for gen in range(1, generations + 1):
        fitness = np.array([objective_func(ind) for ind in pop])

        # Tìm cá thể tốt nhất
        best_idx = np.argmin(fitness)
        best_val = fitness[best_idx]
        best_ind = pop[best_idx].copy()
        best_history.append(best_ind)

        # Kiểm tra hội tụ (Nếu sai số đủ nhỏ so với 0)
        # Lưu ý: Với hàm tối ưu, ta thường coi < target là đạt
        if best_val < target_fitness:
            return best_ind, np.array(best_history), gen, True

        # --- Tạo thế hệ mới ---
        new_pop = [best_ind]  # Elitism: Giữ lại con tốt nhất

        while len(new_pop) < pop_size:
            # Tournament Selection
            idxs = np.random.choice(pop_size, 4, replace=False)
            # Chọn bố
            p1 = pop[idxs[0]] if fitness[idxs[0]] < fitness[idxs[1]] else pop[idxs[1]]
            # Chọn mẹ
            p2 = pop[idxs[2]] if fitness[idxs[2]] < fitness[idxs[3]] else pop[idxs[3]]

            # Crossover (Lai ghép)
            child = 0.5 * p1 + 0.5 * p2

            # Mutation (Đột biến)
            if np.random.rand() < mutation_rate:
                # Thêm nhiễu để nhảy ra khỏi hố
                child += np.random.normal(0, 0.3, dim)

            child = np.clip(child, bounds[:, 0], bounds[:, 1])
            new_pop.append(child)

        pop = np.array(new_pop)

    return best_history[-1], np.array(best_history), generations, False


# ==========================================
# 4. HÀM CHẠY VÀ VẼ
# ==========================================
def run_comparison(func_name, func, grad, start_point, bounds, adam_lr, ga_gens, target_score):
    print(f"\n{'=' * 20} {func_name.upper()} (Start: {start_point}) {'=' * 20}")

    # --- Chạy Adam ---
    adam_x, adam_path, adam_iters, adam_success = adam_optimizer(
        grad, start_point, learning_rate=adam_lr, max_iter=1000, tol=1e-4
    )
    adam_score = func(adam_x)

    # --- Chạy GA ---
    ga_x, ga_path, ga_gens_count, ga_success = genetic_algorithm(
        func, start_point, bounds, pop_size=100, generations=ga_gens, target_fitness=target_score
    )
    ga_score = func(ga_x)

    # --- In Bảng Kết Quả ---
    print(f"{'Thuật toán':<10} | {'Steps/Gens':<15} | {'Trạng thái':<15} | {'Giá trị Min':<15} | {'Toạ độ cuối'}")
    print("-" * 85)
    print(
        f"{'Adam':<10} | {adam_iters:<15} | {'Hội tụ' if adam_success else 'Dừng (Max)':<15} | {adam_score:.8f}        | [{adam_x[0]:.2f}, {adam_x[1]:.2f}]")
    print(
        f"{'GA':<10}   | {ga_gens_count:<15} | {'Đạt target' if ga_success else 'Dừng (Max)':<15} | {ga_score:.8f}        | [{ga_x[0]:.2f}, {ga_x[1]:.2f}]")

    # --- Vẽ đồ thị ---
    x_range = np.linspace(bounds[0][0], bounds[0][1], 150)
    y_range = np.linspace(bounds[1][0], bounds[1][1], 150)
    X, Y = np.meshgrid(x_range, y_range)
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = func([X[i, j], Y[i, j]])

    plt.figure(figsize=(10, 7))
    plt.contourf(X, Y, Z, levels=50, cmap='viridis', alpha=0.5)  # Vẽ địa hình
    plt.colorbar(label='Cost Value')

    # Vẽ quỹ đạo
    plt.plot(adam_path[:, 0], adam_path[:, 1], 'r.-', linewidth=2, label=f'Adam ({adam_iters} steps)')
    plt.plot(ga_path[:, 0], ga_path[:, 1], 'w.--', linewidth=2, label=f'GA ({ga_gens_count} gens)')

    # Vẽ điểm
    plt.plot(start_point[0], start_point[1], 'ko', markersize=8, label='START')
    if func_name == "Rosenbrock":
        plt.plot(1, 1, 'b*', markersize=15, label='Global Min (1,1)')
    else:  # Rastrigin
        plt.plot(0, 0, 'b*', markersize=15, label='Global Min (0,0)')

    plt.title(f"{func_name}: Adam vs GA (Start tại {start_point})")
    plt.legend()
    plt.show()


# ==========================================
# 5. CHẠY THỰC TẾ
# ==========================================

# Điểm bắt đầu chung theo yêu cầu
COMMON_START = [-1.0, 2.0]

# --- BÀI 1: ROSENBROCK ---
run_comparison(
    func_name="Rosenbrock",
    func=rosenbrock_func,
    grad=rosenbrock_grad,
    start_point=COMMON_START,
    bounds=[[-2, 2], [-1, 3]],
    adam_lr=0.1,
    ga_gens=200,
    target_score=1e-3
)

# --- BÀI 2: RASTRIGIN ---
# Lưu ý: Từ [-1, 2] về [0, 0] có rất nhiều hố chông gai
run_comparison(
    func_name="Rastrigin",
    func=rastrigin_func,
    grad=rastrigin_grad,
    start_point=COMMON_START,
    bounds=[[-2.5, 2.5], [-2.5, 2.5]],  # Zoom vào gần một chút để dễ nhìn
    adam_lr=0.05,
    ga_gens=200,
    target_score=1e-2
)