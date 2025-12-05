# ==== main.py ====
import numpy as np
import matplotlib.pyplot as plt

def simulate_logistic(r, K, N0, t):
    """Numerical integration of logistic growth ODE using explicit Euler."""
    N = np.empty_like(t)
    N[0] = N0
    dt = t[1] - t[0]
    for i in range(1, len(t)):
        dN = r * N[i-1] * (1 - N[i-1] / K)
        N[i] = N[i-1] + dN * dt
        # Prevent negative or overflow values
        if N[i] < 0:
            N[i] = 0.0
    return N

def experiment_logistic_growth():
    # Parameters for baseline logistic growth
    r = 0.5               # per hour
    K = 1e9               # carrying capacity (cells)
    N0 = 1e6              # initial population (cells)
    t = np.linspace(0, 20, 400)  # hours
    N = simulate_logistic(r, K, N0, t)

    plt.figure(figsize=(8, 5))
    plt.plot(t, N, label='Logistic growth (numerical)')
    plt.title('Population vs Time (Logistic Growth)')
    plt.xlabel('Time (hours)')
    plt.ylabel('Population size')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('population_vs_time_logistic.png')
    plt.close()

def experiment_nutrient_sweep():
    # Sweep over nutrient concentration which scales carrying capacity
    r = 0.5               # per hour (same as baseline)
    N0 = 1e6              # initial population (cells)
    alpha = 5e8           # conversion factor from nutrient to K
    nutrient_levels = np.linspace(0.1, 2.0, 20)  # arbitrary units
    K_vals = alpha * nutrient_levels
    t_long = np.linspace(0, 30, 600)  # longer time to reach steady state

    final_populations = []
    for K in K_vals:
        N = simulate_logistic(r, K, N0, t_long)
        final_populations.append(N[-1])
    final_populations = np.array(final_populations)

    plt.figure(figsize=(8, 5))
    plt.plot(nutrient_levels, final_populations, marker='o')
    plt.title('Steady‑state Population vs Nutrient Concentration')
    plt.xlabel('Nutrient concentration (arbitrary units)')
    plt.ylabel('Steady‑state population size')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('final_population_vs_nutrient.png')
    plt.close()

    # Primary numeric answer: steady‑state population at highest nutrient level
    answer = final_populations[-1]
    print('Answer:', answer)

if __name__ == '__main__':
    experiment_logistic_growth()
    experiment_nutrient_sweep()
