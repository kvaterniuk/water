# Необхідні бібліотеки
import jax
import jax.numpy as jnp
import time
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------------------
# V.1. Ініціалізація та Параметри
# -----------------------------------------------------
jax.config.update("jax_enable_x64", True)

# Фізичні параметри
L = 120000.0       # м
Nx = 20000        # комірки
Dx = L / Nx       # 0.5 м
u = 0.5           # м/с
Dl = 5.0          # м^2/с
T_env = 298.15    # К

# Перевірка стабільності (Критерій Неймана для дифузії: alpha <= 0.5)
# Твій старий Dt=0.04 давав alpha=0.8 (нестабільно). Зменшуємо Dt.
Dt = 0.01         # с (Зменшено для стабільності)
N_steps = 4000000 # Збільшено кількість кроків, щоб покрити той самий час

# Кінетика
T_ref = 293.15
k_A_20 = 1.0e-4
k_N_20 = 5.0e-5
theta = 1.05

# Джерело
S_max = 1.0e-6
X_start = 1000.0
X_end = 3000.0

# Граничні умови (Вхід)
C_in_A = 100.0
C_in_N = 0.0
C_init_A = 0.0
C_init_N = 0.0

# Індекси джерела
i_start = int(X_start / Dx)
i_end = int(X_end / Dx)

# Масив джерела (constant, тому винесемо його, щоб не створювати щоразу)
S_diff_N_base = jnp.zeros(Nx)
S_diff_N = S_diff_N_base.at[i_start:i_end].set(S_max)

# Кінетичні константи
def calculate_kinetics(T):
    k_A = k_A_20 * (theta ** ((T - T_ref) / 1.0))
    k_N = k_N_20 * (theta ** ((T - T_ref) / 1.0))
    return k_A, k_N

k_nitr, k_denit = calculate_kinetics(T_env)

# -----------------------------------------------------
# V.2. Ядро симуляції (Scan loop)
# -----------------------------------------------------

# Ми передаємо всі константи через closure або arguments,
# але для lax.scan зручно мати функцію кроку, що приймає поточний стан.

@jax.jit
def simulation_step(state, _):
    """
    state: кортеж (C_A, C_N)
    _: фіктивна змінна (номер кроку, не використовується)
    """
    C_A, C_N = state

    # --- 1. Граничні умови (Правильний підхід без roll) ---
    # Для C_i-1 (сусіди зліва):
    # На вході (i=0) беремо C_in (boundary), далі беремо масив до передостаннього елемента
    C_A_left = jnp.concatenate([jnp.array([C_in_A]), C_A[:-1]])
    C_N_left = jnp.concatenate([jnp.array([C_in_N]), C_N[:-1]])

    # Для C_i+1 (сусіди справа):
    # На виході (i=Nx-1) беремо C_i (умова Neumann: dC/dx = 0 -> C_last = C_prev),
    # тобто дублюємо останній елемент.
    C_A_right = jnp.concatenate([C_A[1:], jnp.array([C_A[-1]])])
    C_N_right = jnp.concatenate([C_N[1:], jnp.array([C_N[-1]])])

    # --- 2. Похідні ---
    # Дифузія (Central Difference)
    Dispersion_A = Dl * (C_A_right - 2 * C_A + C_A_left) / (Dx**2)
    Dispersion_N = Dl * (C_N_right - 2 * C_N + C_N_left) / (Dx**2)

    # Адвекція (Upwind: u > 0, тому потік йде зліва направо: C_i - C_i-1)
    Advection_A = u * (C_A - C_A_left) / Dx
    Advection_N = u * (C_N - C_N_left) / Dx

    # --- 3. Реакції ---
    R_nitr = k_nitr * C_A
    R_denit = k_denit * C_N

    RA = -R_nitr
    RN = R_nitr - R_denit + S_diff_N

    # --- 4. Оновлення ---
    C_A_new = C_A + Dt * (-Advection_A + Dispersion_A + RA)
    C_N_new = C_N + Dt * (-Advection_N + Dispersion_N + RN)

    # Примусове закріплення входу (для надійності, хоча Upwind це вже врахував через C_A_left[0])
    C_A_new = C_A_new.at[0].set(C_in_A)
    C_N_new = C_N_new.at[0].set(C_in_N)

    return (C_A_new, C_N_new), None

# -----------------------------------------------------
# V.3. Запуск
# -----------------------------------------------------

# Початковий стан
C_A_init = jnp.full(Nx, C_init_A, dtype=jnp.float64)
C_N_init = jnp.full(Nx, C_init_N, dtype=jnp.float64)
init_state = (C_A_init, C_N_init)

print(f"Початок симуляції: {N_steps} кроків (Scan mode)...")
start_time = time.time()

# Використовуємо lax.scan - це "магія" JAX для циклів
# Вона компілює весь цикл у один kernel XLA
final_state, _ = jax.lax.scan(simulation_step, init_state, length=N_steps)

# Блокуємо до завершення обчислень
final_C_A, final_C_N = final_state
final_C_A.block_until_ready()

end_time = time.time()
print(f"Готово! Час: {end_time - start_time:.4f} с")

# -----------------------------------------------------
# V.4. Візуалізація
# -----------------------------------------------------
x_coords = np.linspace(0, L, Nx)

plt.figure(figsize=(10, 6))
plt.plot(x_coords, final_C_A, label='Амоній (NH4+)', color='blue')
plt.plot(x_coords, final_C_N, label='Нітрат (NO3-)', color='red')
plt.axvspan(X_start, X_end, color='yellow', alpha=0.2, label='Зона джерела NO3')
plt.xlabel('Відстань (м)')
plt.ylabel('Концентрація (мг/м^3)')
plt.title(f'Профіль концентрації після T={N_steps*Dt:.0f}с')
plt.legend()
plt.grid(True)
plt.show()