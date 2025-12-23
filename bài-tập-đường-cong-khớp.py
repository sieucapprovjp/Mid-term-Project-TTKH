import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


# Rosenbrock function
def rosenbrock(x, y):
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2


# Rastrigin function
def rastrigin(x, y):
    A = 150
    return 2 * A + (x ** 2 - A * np.cos(2 * np.pi * x)) + (y ** 2 - A * np.cos(2 * np.pi * y))


# Choose which function to optimize
FUNCTION = rosenbrock
FUNCTION_NAME = "Rosenbrock"
GLOBAL_OPTIMUM = (1, 1, 0) if FUNCTION == rosenbrock else (0, 0, 0)


# WOA
def whale_optimization(max_iter=500, n_whales=50):
    dim = 2
    lb, ub = -5, 5
    whales = np.random.uniform(-2, 2, (n_whales, dim))
    fitness = np.array([FUNCTION(w[0], w[1]) for w in whales])
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
            new_fit = FUNCTION(whales[i][0], whales[i][1])
            if new_fit < best_fit:
                best_fit = new_fit
                best_pos = whales[i].copy()

        history.append(best_fit)

    return history, best_fit, best_pos


# Simulated Annealing
def simulated_annealing(max_iter=500):
    x = np.random.uniform(0, 2)
    y = np.random.uniform(0, 2)
    current_fit = FUNCTION(x, y)
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
            new_fit = FUNCTION(new_x, new_y)
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
        fitness = np.array([FUNCTION(ind[0], ind[1]) for ind in population])
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
                mutation_strength = 0.5 * (1 - iter / max_iter)
                child += np.random.normal(0, mutation_strength, 2)

            child = np.clip(child, lb, ub)
            new_population.append(child)

        population = np.array(new_population[:pop_size])
        current_fitness = np.array([FUNCTION(ind[0], ind[1]) for ind in population])
        history.append(np.min(current_fitness))

    final_fitness = np.array([FUNCTION(ind[0], ind[1]) for ind in population])
    best_idx = np.argmin(final_fitness)
    return history, final_fitness[best_idx], population[best_idx]


# Block Coordinate Descent with Momentum
def block_coordinate_descent(max_iter=500):
    x = np.random.uniform(0, 2)
    y = np.random.uniform(0, 2)
    history = []
    lr = 0.01
    momentum = 0.9
    velocity_x = 0
    velocity_y = 0

    for iter in range(max_iter):
        if FUNCTION == rosenbrock:
            grad_x = -2 * (1 - x) - 400 * x * (y - x ** 2)
            grad_y = 200 * (y - x ** 2)
        else:
            grad_x = 2 * x + 20 * np.pi * np.sin(2 * np.pi * x)
            grad_y = 2 * y + 20 * np.pi * np.sin(2 * np.pi * y)

        grad_x = np.clip(grad_x, -10, 10)
        grad_y = np.clip(grad_y, -10, 10)

        velocity_x = momentum * velocity_x - lr * grad_x
        velocity_y = momentum * velocity_y - lr * grad_y

        x = x + velocity_x
        y = y + velocity_y

        x = np.clip(x, -5, 5)
        y = np.clip(y, -5, 5)

        if iter % 50 == 0 and iter > 0:
            lr *= 0.95

        current_fit = FUNCTION(x, y)
        history.append(current_fit)

    return history, FUNCTION(x, y), np.array([x, y])


# Run algorithms
print(f"{FUNCTION_NAME}")
np.random.seed(42)  # For reproducibility

woa_history, woa_best, woa_pos = whale_optimization(500)
sa_history, sa_best, sa_pos = simulated_annealing(500)
ga_history, ga_best, ga_pos = genetic_algorithm(500)
bcd_history, bcd_best, bcd_pos = block_coordinate_descent(500)

print("\nResults:")
print(f"WOA: {woa_best:.6f} at ({woa_pos[0]:.4f}, {woa_pos[1]:.4f})")
print(f"SA:  {sa_best:.6f} at ({sa_pos[0]:.4f}, {sa_pos[1]:.4f})")
print(f"GA:  {ga_best:.6f} at ({ga_pos[0]:.4f}, {ga_pos[1]:.4f})")
print(f"BCD: {bcd_best:.6f} at ({bcd_pos[0]:.4f}, {bcd_pos[1]:.4f})")

fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

x_range = np.linspace(-2, 3.5, 100)
y_range = np.linspace(-1, 3.5, 100)
X, Y = np.meshgrid(x_range, y_range)
Z = FUNCTION(X, Y)

Z_display = np.minimum(Z, 2500)

surf = ax.plot_surface(X, Y, Z_display, cmap=cm.viridis, alpha=0.6,
                       edgecolor='none', antialiased=True)

marker_size = 150
ax.scatter(woa_pos[0], woa_pos[1], FUNCTION(woa_pos[0], woa_pos[1]),
           color='#8b5cf6', s=marker_size, marker='o', edgecolors='white',
           linewidths=2, label='WOA', zorder=10)

ax.scatter(sa_pos[0], sa_pos[1], FUNCTION(sa_pos[0], sa_pos[1]),
           color='#ef4444', s=marker_size, marker='o', edgecolors='white',
           linewidths=2, label='SA', zorder=10)

ax.scatter(ga_pos[0], ga_pos[1], FUNCTION(ga_pos[0], ga_pos[1]),
           color='#10b981', s=marker_size, marker='o', edgecolors='white',
           linewidths=2, label='GA', zorder=10)

ax.scatter(bcd_pos[0], bcd_pos[1], FUNCTION(bcd_pos[0], bcd_pos[1]),
           color='#f59e0b', s=marker_size, marker='o', edgecolors='white',
           linewidths=2, label='BCD', zorder=10)

ax.scatter(GLOBAL_OPTIMUM[0], GLOBAL_OPTIMUM[1], GLOBAL_OPTIMUM[2],
           color='#1e40af', s=250, marker='*', edgecolors='white',
           linewidths=2, label='Global Optimum', zorder=15)

ax.set_xlabel('X', fontsize=12, labelpad=10)
ax.set_ylabel('Y', fontsize=12, labelpad=10)
ax.set_zlabel(f'f(x, y)', fontsize=12, labelpad=10)
ax.set_title(f'Hàm {FUNCTION_NAME} ',
             fontsize=14, fontweight='bold', pad=20)

# Legend
ax.legend(loc='upper left', fontsize=10, framealpha=0.9)


ax.view_init(elev=20, azim=45)

ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{FUNCTION_NAME.lower()}_3d_visualization.png', dpi=300, bbox_inches='tight')
print(f"\nFigure saved as '{FUNCTION_NAME.lower()}_3d_visualization.png'")

plt.show()

fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

ax1.plot(woa_history, label='Whale Optimization', linewidth=2, color='#8b5cf6')
ax1.plot(sa_history, label='Simulated Annealing', linewidth=2, color='#ef4444')
ax1.plot(ga_history, label='Genetic Algorithm', linewidth=2, color='#10b981')
ax1.plot(bcd_history, label='Block Coordinate Descent', linewidth=2, color='#f59e0b')
ax1.set_xlabel('Iteration', fontsize=12)
ax1.set_ylabel('Fitness Value', fontsize=12)
ax1.set_title(f'So sánh tốc độ hội tụ - Hàm {FUNCTION_NAME} (500 iterations)',
              fontsize=14, fontweight='bold')
ax1.legend(fontsize=10, loc='upper right')
ax1.grid(True, alpha=0.3)
ax1.axhline(y=GLOBAL_OPTIMUM[2], color='black', linestyle=':', linewidth=1)

ax2.semilogy(woa_history, label='Whale Optimization', linewidth=2, color='#8b5cf6')
ax2.semilogy(sa_history, label='Simulated Annealing', linewidth=2, color='#ef4444')
ax2.semilogy(ga_history, label='Genetic Algorithm', linewidth=2, color='#10b981')
ax2.semilogy(bcd_history, label='Block Coordinate Descent', linewidth=2, color='#f59e0b')
ax2.set_xlabel('Iteration', fontsize=12)
ax2.set_ylabel('Fitness Value (log scale)', fontsize=12)
ax2.set_title('So sánh tốc độ hội tụ (Log Scale)', fontsize=14, fontweight='bold')
ax2.legend(fontsize=10, loc='upper right')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{FUNCTION_NAME.lower()}_convergence.png', dpi=300, bbox_inches='tight')

plt.show()