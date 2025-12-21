import numpy as np
import matplotlib.pyplot as plt


# ==========================================
# 1. ĐỊNH NGHĨA HÀM & ĐẠO HÀM (GRADIENT)
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
# 2. THUẬT TOÁN ADAM OPTIMIZER (Cập nhật Return)
# ==========================================
def adam_optimizer(grad_func, x0, learning_rate=0.05, beta1=0.9, beta2=0.999, epsilon=1e-8, max_iter=2000, tol=1e-6):
    x = np.array(x0, dtype=float)
    dim = len(x)
    m = np.zeros(dim)
    v = np.zeros(dim)
    history = [x.copy()]

    for t in range(1, max_iter + 1):
        g = grad_func(x)

        # Kiểm tra điều kiện dừng (Gradient cực nhỏ -> Đáy)
        if np.linalg.norm(g) < tol:
            return x, np.array(history), t, True  # True = Hội tụ sớm

        # Cập nhật m, v
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * (g ** 2)

        # Bias correction
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)

        # Cập nhật x
        x = x - learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
        history.append(x.copy())

    return x, np.array(history), max_iter, False  # False = Hết vòng lặp


# ==========================================
# 3. HÀM CHẠY VÀ BÁO CÁO KẾT QUẢ (Report & Plot)
# ==========================================
def run_and_report_adam(func_name, func, grad, start_point, bounds, learning_rate, global_min_true, tol=1e-6):
    print(f"\n{'=' * 25} BÁO CÁO ADAM: {func_name.upper()} {'=' * 25}")

    # --- CHẠY THUẬT TOÁN ---
    final_pos, path, iterations, success = adam_optimizer(
        grad, start_point, learning_rate=learning_rate, max_iter=2000, tol=tol
    )

    final_val = func(final_pos)
    ideal_val = 0.0

    # --- IN RA CONSOLE ---
    print("I. THÔNG SỐ & MỤC TIÊU:")
    print(f"   - Xuất phát (Start)     : {start_point}")
    print(f"   - Đích thực tế (Target) : {global_min_true}")
    print(f"   - Learning Rate         : {learning_rate}")

    print("\nII. KẾT QUẢ ADAM:")
    print(f"   - Vị trí hội tụ         : [{final_pos[0]:.5f}, {final_pos[1]:.5f}]")
    print(f"   - Giá trị hàm (Loss)    : {final_val:.8f}")
    print(f"   - Sai số (Error)        : {abs(final_val - ideal_val):.8f}")
    print(f"   - Số vòng lặp (Iter)    : {iterations}")
    print(f"   - Trạng thái            : {'HỘI TỤ (Gradient < tol)' if success else 'DỪNG (Max Iterations)'}")
    print("=" * 70)

    # --- VẼ HÌNH MINH HOẠ ---
    plt.figure(figsize=(9, 7))

    # Tạo lưới vẽ nền
    x_range = np.linspace(bounds[0][0], bounds[0][1], 200)
    y_range = np.linspace(bounds[1][0], bounds[1][1], 200)
    X, Y = np.meshgrid(x_range, y_range)
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = func([X[i, j], Y[i, j]])

    # Tùy chỉnh cách vẽ Contour cho từng hàm để đẹp nhất
    if func_name == "Rosenbrock":
        # Rosenbrock dốc nên dùng logspace để thấy rõ thung lũng
        plt.contourf(X, Y, Z, levels=np.logspace(-1, 3, 30), cmap='jet', alpha=0.6)
    else:
        # Rastrigin dùng linear contour
        plt.contourf(X, Y, Z, levels=50, cmap='viridis', alpha=0.6)

    plt.colorbar(label='Cost Value')

    # Vẽ đường đi của Adam (Màu đỏ)
    plt.plot(path[:, 0], path[:, 1], 'r.-', linewidth=1.5, label='Adam Path')

    # Điểm Start (Xanh lá)
    plt.plot(start_point[0], start_point[1], 'o', color='lime', markersize=8, markeredgecolor='k', label='Start')

    # Điểm Đích mong muốn (Sao vàng viền đen)
    plt.plot(global_min_true[0], global_min_true[1], '*', color='yellow', markersize=15, markeredgecolor='k',
             label='True Target')

    # Điểm Kết thúc thực tế của Adam (Trắng)
    plt.plot(final_pos[0], final_pos[1], 'X', color='white', markersize=10, markeredgecolor='k', label='Adam End')

    plt.title(f"Adam Optimizer Visualization: {func_name}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


# ==========================================
# 4. THỰC THI
# ==========================================

# Điểm xuất phát chung giống bài GA
start_common = [-1.0, 2.0]

# --- BÀI 1: ROSENBROCK ---
run_and_report_adam(
    func_name="Rosenbrock",
    func=rosenbrock_func,
    grad=rosenbrock_grad,
    start_point=start_common,
    bounds=[[-2, 2], [-1, 3]],
    learning_rate=0.5,  # LR lớn chạy cho nhanh xuống thung lũng
    global_min_true=[1, 1],
    tol=1e-6
)

# --- BÀI 2: RASTRIGIN ---
# Lưu ý: Adam rất dễ kẹt ở Rastrigin.
# Từ [-1, 2] về [0,0] có thể nó sẽ bị kẹt ở hố [-1, 2] hoặc [0, 2] gần đó.
run_and_report_adam(
    func_name="Rastrigin",
    func=rastrigin_func,
    grad=rastrigin_grad,
    start_point=start_common,
    bounds=[[-3, 3], [-3, 3]],
    learning_rate=0.1,  # LR vừa phải
    global_min_true=[0, 0],
    tol=1e-5
)