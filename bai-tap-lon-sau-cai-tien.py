import numpy as np
import matplotlib.pyplot as plt

#Rosenbrock
def rosenbrock(x, y):
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2

def rastrigin(x):
    x = np.asarray(x)
    n = len(x)
    A = 10
    return A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x))
# WOA
def whale_optimization(max_iter=500, n_whales=50):
    dim = 2
    lb, ub = -5, 5
    whales = np.random.uniform(-2, 2, (n_whales, dim))
    fitness = np.array([rosenbrock(w[0], w[1]) for w in whales])
    best_idx = np.argmin(fitness)
    best_pos = whales[best_idx].copy()
    best_fit = fitness[best_idx]
    history = []
    for iter in range(max_iter):
        a = 2 - iter * 2 / max_iter
        for i in range(n_whales):
            r1 = np.random.random()
            r2 = np.random.random()
            A = 2 * a * r1 - a
            C = 2 * r2
            p = np.random.random()
            if p < 0.5:
                if abs(A) < 1:
                    D = abs(C * best_pos - whales[i])
                    whales[i] = best_pos - A * D
                else:
                    rand_idx = np.random.randint(n_whales)
                    D = abs(C * whales[rand_idx] - whales[i])
                    whales[i] = whales[rand_idx] - A * D
            else:
                b = 1
                l = np.random.uniform(-1, 1)
                D = abs(best_pos - whales[i])
                whales[i] = D * np.exp(b * l) * np.cos(2 * np.pi * l) + best_pos

            whales[i] = np.clip(whales[i], lb, ub)

            new_fit = rosenbrock(whales[i][0], whales[i][1])
            if new_fit < best_fit:
                best_fit = new_fit
                best_pos = whales[i].copy()

        history.append(best_fit)

    return history, best_fit, best_pos


# Simulated Annealing
def simulated_annealing(max_iter=500):
    x = np.random.uniform(0, 2)
    y = np.random.uniform(0, 2)
    current_fit = rosenbrock(x, y)
    best_fit = current_fit
    best_pos = np.array([x, y])

    T = 200
    cooling_rate = 0.98
    history = []

    for iter in range(max_iter):
        step_size = T / 100
        new_x = x + np.random.uniform(-step_size, step_size)
        new_y = y + np.random.uniform(-step_size, step_size)

        if -5 <= new_x <= 5 and -5 <= new_y <= 5:
            new_fit = rosenbrock(new_x, new_y)
            delta = new_fit - current_fit

            if delta < 0 or np.random.random() < np.exp(-delta / T):
                x, y = new_x, new_y
                current_fit = new_fit

                if current_fit < best_fit:
                    best_fit = current_fit
                    best_pos = np.array([x, y])

        T *= cooling_rate
        history.append(best_fit)

    return history, best_fit, best_pos


# Genetic Algorithm
def genetic_algorithm(max_iter=500, pop_size=100):
    mutation_rate = 0.15
    crossover_rate = 0.85
    lb, ub = -5, 5

    population = np.random.uniform(-1, 2, (pop_size, 2))
    history = []

    for iter in range(max_iter):
        fitness = np.array([rosenbrock(ind[0], ind[1]) for ind in population])
        fitness_scaled = 1 / (1 + fitness)
        elite_count = pop_size // 10
        elite_indices = np.argsort(fitness)[:elite_count]
        elites = population[elite_indices].copy()

        new_population = list(elites)

        while len(new_population) < pop_size:
            total_fit = np.sum(fitness_scaled)
            pick1 = np.random.uniform(0, total_fit)
            pick2 = np.random.uniform(0, total_fit)

            current = 0
            parent1_idx, parent2_idx = 0, 0
            for idx, fit in enumerate(fitness_scaled):
                current += fit
                if current >= pick1 and parent1_idx == 0:
                    parent1_idx = idx
                if current >= pick2 and parent2_idx == 0:
                    parent2_idx = idx

            parent1 = population[parent1_idx]
            parent2 = population[parent2_idx]

            if np.random.random() < crossover_rate:
                alpha = np.random.random()
                child = alpha * parent1 + (1 - alpha) * parent2
            else:
                child = parent1.copy()
            if np.random.random() < mutation_rate:
                mutation_strength = 0.5 * (1 - iter / max_iter)  # Giảm dần
                child += np.random.normal(0, mutation_strength, 2)
            child = np.clip(child, lb, ub)
            new_population.append(child)
        population = np.array(new_population[:pop_size])
        current_fitness = np.array([rosenbrock(ind[0], ind[1]) for ind in population])
        history.append(np.min(current_fitness))
    final_fitness = np.array([rosenbrock(ind[0], ind[1]) for ind in population])
    best_idx = np.argmin(final_fitness)
    return history, final_fitness[best_idx], population[best_idx]
def block_coordinate_descent(max_iter=500):
    x = np.random.uniform(0, 2)
    y = np.random.uniform(0, 2)
    history = []
    lr = 0.01
    momentum = 0.9
    velocity_x = 0
    velocity_y = 0

    for iter in range(max_iter):
        grad_x = -2 * (1 - x) - 400 * x * (y - x ** 2)
        grad_x = np.clip(grad_x, -10, 10)
        grad_y = 200 * (y - x ** 2)
        grad_y = np.clip(grad_y, -10, 10)
        velocity_x = momentum * velocity_x - lr * grad_x
        velocity_y = momentum * velocity_y - lr * grad_y
        x = x + velocity_x
        y = y + velocity_y
        x = np.clip(x, -5, 5)
        y = np.clip(y, -5, 5)

        if iter % 50 == 0 and iter > 0:
            lr *= 0.95

        current_fit = rosenbrock(x, y)
        history.append(current_fit)

    return history, rosenbrock(x, y), np.array([x, y])


# Adam Optimizer
def adam_optimizer(max_iter=500):
    x = np.random.uniform(0, 2)
    y = np.random.uniform(0, 2)
    history = []

    # Adam parameters
    lr = 0.01
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8

    m_x, m_y = 0, 0
    v_x, v_y = 0, 0

    for iter in range(max_iter):
        grad_x = -2 * (1 - x) - 400 * x * (y - x ** 2)
        grad_y = 200 * (y - x ** 2)
        m_x = beta1 * m_x + (1 - beta1) * grad_x
        m_y = beta1 * m_y + (1 - beta1) * grad_y
        v_x = beta2 * v_x + (1 - beta2) * grad_x ** 2
        v_y = beta2 * v_y + (1 - beta2) * grad_y ** 2
        m_x_hat = m_x / (1 - beta1 ** (iter + 1))
        m_y_hat = m_y / (1 - beta1 ** (iter + 1))
        v_x_hat = v_x / (1 - beta2 ** (iter + 1))
        v_y_hat = v_y / (1 - beta2 ** (iter + 1))
        x = x - lr * m_x_hat / (np.sqrt(v_x_hat) + epsilon)
        y = y - lr * m_y_hat / (np.sqrt(v_y_hat) + epsilon)
        x = np.clip(x, -5, 5)
        y = np.clip(y, -5, 5)
        current_fit = rosenbrock(x, y)
        history.append(current_fit)

    return history, rosenbrock(x, y), np.array([x, y])


# Chạy tất cả thuật toán
print("Đang chạy các thuật toán tối ưu hóa (500 iterations)...")
print("=" * 70)

woa_history, woa_best, woa_pos = whale_optimization(500)
print(f"Whale Optimization Algorithm:")
print(f"  Best fitness: {woa_best:.10f}")
print(f"  Best position: x={woa_pos[0]:.6f}, y={woa_pos[1]:.6f}")
print(f"  Khoảng cách đến optimum: {np.sqrt((woa_pos[0] - 1) ** 2 + (woa_pos[1] - 1) ** 2):.6f}")

sa_history, sa_best, sa_pos = simulated_annealing(500)
print(f"\nSimulated Annealing:")
print(f"  Best fitness: {sa_best:.10f}")
print(f"  Best position: x={sa_pos[0]:.6f}, y={sa_pos[1]:.6f}")
print(f"  Khoảng cách đến optimum: {np.sqrt((sa_pos[0] - 1) ** 2 + (sa_pos[1] - 1) ** 2):.6f}")

ga_history, ga_best, ga_pos = genetic_algorithm(500)
print(f"\nGenetic Algorithm:")
print(f"  Best fitness: {ga_best:.10f}")
print(f"  Best position: x={ga_pos[0]:.6f}, y={ga_pos[1]:.6f}")
print(f"  Khoảng cách đến optimum: {np.sqrt((ga_pos[0] - 1) ** 2 + (ga_pos[1] - 1) ** 2):.6f}")

bcd_history, bcd_best, bcd_pos = block_coordinate_descent(500)
print(f"\nBlock Coordinate Descent (với Momentum):")
print(f"  Best fitness: {bcd_best:.10f}")
print(f"  Best position: x={bcd_pos[0]:.6f}, y={bcd_pos[1]:.6f}")
print(f"  Khoảng cách đến optimum: {np.sqrt((bcd_pos[0] - 1) ** 2 + (bcd_pos[1] - 1) ** 2):.6f}")

adam_history, adam_best, adam_pos = adam_optimizer(500)
print(f"\nAdam Optimizer (BONUS):")
print(f"  Best fitness: {adam_best:.10f}")
print(f"  Best position: x={adam_pos[0]:.6f}, y={adam_pos[1]:.6f}")
print(f"  Khoảng cách đến optimum: {np.sqrt((adam_pos[0] - 1) ** 2 + (adam_pos[1] - 1) ** 2):.6f}")

print("\n" + "=" * 70)
print("Global Optimum: f(1, 1) = 0")

# Vẽ đồ thị so sánh
plt.figure(figsize=(14, 10))

plt.subplot(2, 1, 1)
plt.plot(woa_history, label='Whale Optimization', linewidth=2, color='#8b5cf6')
plt.plot(sa_history, label='Simulated Annealing', linewidth=2, color='#ef4444')
plt.plot(ga_history, label='Genetic Algorithm', linewidth=2, color='#10b981')
plt.plot(bcd_history, label='Block Coordinate Descent', linewidth=2, color='#f59e0b')
plt.plot(adam_history, label='Adam Optimizer', linewidth=2, color='#3b82f6', linestyle='--')
plt.xlabel('Iteration', fontsize=12)
plt.ylabel('Fitness Value', fontsize=12)
plt.title('So sánh tốc độ hội tụ - Hàm Rosenbrock (500 iterations)',
          fontsize=14, fontweight='bold')
plt.legend(fontsize=10, loc='upper right')
plt.grid(True, alpha=0.3)
plt.axhline(y=0, color='black', linestyle=':', linewidth=1, label='Global Optimum')

plt.subplot(2, 1, 2)
plt.semilogy(woa_history, label='Whale Optimization', linewidth=2, color='#8b5cf6')
plt.semilogy(sa_history, label='Simulated Annealing', linewidth=2, color='#ef4444')
plt.semilogy(ga_history, label='Genetic Algorithm', linewidth=2, color='#10b981')
plt.semilogy(bcd_history, label='Block Coordinate Descent', linewidth=2, color='#f59e0b')
plt.semilogy(adam_history, label='Adam Optimizer', linewidth=2, color='#3b82f6', linestyle='--')
plt.xlabel('Iteration', fontsize=12)
plt.ylabel('Fitness Value (log scale)', fontsize=12)
plt.title('So sánh tốc độ hội tụ (Log Scale)', fontsize=14, fontweight='bold')
plt.legend(fontsize=10, loc='upper right')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('rosenbrock_comparison_improved.png', dpi=300, bbox_inches='tight')
# Phân tích vấn đề
print("\n" + "=" * 70)
print("PHÂN TÍCH TẠI SAO KHÓ ĐẠT GLOBAL OPTIMUM:")
print("=" * 70)
plt.show()