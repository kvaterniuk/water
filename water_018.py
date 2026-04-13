import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import time

# --- 0. НАЛАШТУВАННЯ TPU/JAX ---
# JAX автоматично знайде TPU. Якщо TPU немає, він використає CPU або GPU.
try:
    print(f"Доступні пристрої: {jax.devices()}")
    # Вмикаємо 64-бітну точність (опціонально, для гідравліки часто вистачає 32)
    jax.config.update("jax_enable_x64", True)
except:
    print("TPU не знайдено, працюємо на CPU.")

# --- 1. ПАРАМЕТРИ МОДЕЛІ ---

# Геометрія
L_RIVER = 40000.0  # 40 км
W_RIVER = 100.0    # 100 м
H_RIVER = 2.5      # 2.5 м

NX = 200
NY = 50
dx = L_RIVER / NX
dy = W_RIVER / NY

# Гідравліка
V = 0.4    # м/с
DL = 15.0  # м²/с
DT_DISP = 0.5  # м²/с (перейменовано, щоб не плутати з кроком часу)

# Кінетика (Nitrification-Denitrification)
DAYS_TO_SEC = 86400.0
T_WATER = 20.0
THETA = 1.07

def k_T(k_20):
    """Температурна корекція констант"""
    return k_20 * (THETA ** (T_WATER - 20.0))

# Константи швидкості реакцій (1/с)
k_N1 = k_T(0.04 / DAYS_TO_SEC)    # NH4 -> NO2
k_N2 = k_T(0.05 / DAYS_TO_SEC)    # NO2 -> NO3
k_D_V = k_T(0.01 / DAYS_TO_SEC) / H_RIVER # Denitrification
k_Sor = 0.001 / DAYS_TO_SEC       # Sorption NH4

# Джерела забруднення
C_SOURCE_NH4 = 25.0       # Концентрація в скиді
DIFFUSE_LOAD_NO3 = 0.0001 # Фонове навантаження

# --- 2. ЧИСЕЛЬНИЙ МЕТОД ---

# Критерій Куранта-Фрідріхса-Леві (CFL) для стабільності
dt_max_diff = 0.5 * min(dx**2 / DL, dy**2 / DT_DISP)
DT_SIM = min(dt_max_diff / 4, 5.0) # Безпечний крок часу
print(f"Розрахунковий крок часу (dt): {DT_SIM:.4f} с")

# Координати джерела (індекси)
source_x = 1
source_y_start = int(NY * 0.4)
source_y_end = int(NY * 0.6)

# --- 3. ЯДРО СИМУЛЯЦІЇ (JAX JIT COMPILATION) ---

@jax.jit
def step_physics(c_nh4, c_no2, c_no3):
    """
    Один крок симуляції. JAX перетворить цю функцію на
    високоефективний код для TPU без циклів Python.
    """

    # 1. Встановлення джерела (Хмельницький)
    # .at[].set() використовується, бо масиви JAX незмінні
    c_nh4 = c_nh4.at[source_x, source_y_start:source_y_end].set(C_SOURCE_NH4)

    # 2. Підготовка зрізів (Slicing) для векторизації
    # c_center - це внутрішня сітка [1:-1, 1:-1]
    nh4_c = c_nh4[1:-1, 1:-1]
    no2_c = c_no2[1:-1, 1:-1]
    no3_c = c_no3[1:-1, 1:-1]

    # Сусідні комірки для дифузії (Laplacian) та адвекції (Upwind)
    # [i-1], [i+1], [j-1], [j+1]
    # Розміри всіх цих зрізів ідентичні (NX-2, NY-2)

    # NH4 сусіди
    nh4_xm1 = c_nh4[0:-2, 1:-1]; nh4_xp1 = c_nh4[2:, 1:-1]
    nh4_ym1 = c_nh4[1:-1, 0:-2]; nh4_yp1 = c_nh4[1:-1, 2:]

    # NO2 сусіди
    no2_xm1 = c_no2[0:-2, 1:-1]; no2_xp1 = c_no2[2:, 1:-1]
    no2_ym1 = c_no2[1:-1, 0:-2]; no2_yp1 = c_no2[1:-1, 2:]

    # NO3 сусіди
    no3_xm1 = c_no3[0:-2, 1:-1]; no3_xp1 = c_no3[2:, 1:-1]
    no3_ym1 = c_no3[1:-1, 0:-2]; no3_yp1 = c_no3[1:-1, 2:]

    # 3. Обчислення реакцій (R)
    # R_N1: NH4 -> NO2
    r_n1 = k_N1 * nh4_c
    r_sor = k_Sor * nh4_c

    # R_N2: NO2 -> NO3
    r_n2 = k_N2 * no2_c

    # R_D: Denitrification
    r_d = k_D_V * no3_c

    # 4. Рівняння переносу (ADR)

    # --- Рівняння для NH4 ---
    adv_x = -V * (nh4_c - nh4_xm1) / dx
    diff_x = DL * (nh4_xp1 - 2*nh4_c + nh4_xm1) / (dx**2)
    diff_y = DT_DISP * (nh4_yp1 - 2*nh4_c + nh4_ym1) / (dy**2)

    dnh4_dt = adv_x + diff_x + diff_y - r_n1 - r_sor
    c_nh4_new = c_nh4.at[1:-1, 1:-1].add(DT_SIM * dnh4_dt)

    # --- Рівняння для NO2 ---
    adv_x = -V * (no2_c - no2_xm1) / dx
    diff_x = DL * (no2_xp1 - 2*no2_c + no2_xm1) / (dx**2)
    diff_y = DT_DISP * (no2_yp1 - 2*no2_c + no2_ym1) / (dy**2)

    # Прихід від NH4 (r_n1) мінус витрата в NO3 (r_n2)
    dno2_dt = adv_x + diff_x + diff_y + r_n1 - r_n2
    c_no2_new = c_no2.at[1:-1, 1:-1].add(DT_SIM * dno2_dt)

    # --- Рівняння для NO3 ---
    adv_x = -V * (no3_c - no3_xm1) / dx
    diff_x = DL * (no3_xp1 - 2*no3_c + no3_xm1) / (dx**2)
    diff_y = DT_DISP * (no3_yp1 - 2*no3_c + no3_ym1) / (dy**2)

    # Прихід від NO2 (r_n2) мінус денітрифікація (r_d) + дифузне навантаження
    dno3_dt = adv_x + diff_x + diff_y + r_n2 - r_d + DIFFUSE_LOAD_NO3
    c_no3_new = c_no3.at[1:-1, 1:-1].add(DT_SIM * dno3_dt)

    # 5. Граничні умови (Inlet x=0)
    c_nh4_new = c_nh4_new.at[0, :].set(0.1)
    c_no2_new = c_no2_new.at[0, :].set(0.01)
    c_no3_new = c_no3_new.at[0, :].set(1.0)

    return c_nh4_new, c_no2_new, c_no3_new

@jax.jit
def run_batch(c_nh4, c_no2, c_no3, steps):
    """
    Виконує 'steps' ітерацій за один виклик ядра TPU.
    Це критично для швидкодії (зменшує overhead Python).
    """
    def body_fun(i, val):
        return step_physics(*val)

    return jax.lax.fori_loop(0, steps, body_fun, (c_nh4, c_no2, c_no3))

# --- 4. ЗАПУСК ПРОГРАМИ ---

# Ініціалізація полів (на TPU)
C_NH4 = jnp.zeros((NX, NY)).at[0, :].set(0.1)
C_NO2 = jnp.zeros((NX, NY)).at[0, :].set(0.01)
C_NO3 = jnp.zeros((NX, NY)).at[0, :].set(1.0)

MAX_ITER = 100000
BATCH_SIZE = 1000 # Кількість кроків за один раз
TOLERANCE = 1e-6

print("Початок симуляції на TPU...")
start_time = time.time()

for k in range(0, MAX_ITER, BATCH_SIZE):
    C_NH4_prev = C_NH4

    # Основний виклик (виконує 1000 кроків миттєво)
    C_NH4, C_NO2, C_NO3 = run_batch(C_NH4, C_NO2, C_NO3, BATCH_SIZE)

    # Перевірка збіжності кожні 10 000 кроків
    if k % 10000 == 0:
        # jnp.max запускає синхронізацію, щоб отримати число
        diff = jnp.max(jnp.abs(C_NH4 - C_NH4_prev))
        print(f"Ітерація {k}: Макс. зміна концентрації = {diff:.8f}")

        # Якщо зміна дуже мала, виходимо
        if diff < TOLERANCE * (BATCH_SIZE / 100) and k > 2000:
            print(f"Стаціонарний стан досягнуто на ітерації {k}!")
            break

total_time = time.time() - start_time
print(f"Симуляцію завершено за {total_time:.2f} секунд.")

# --- 5. ВІЗУАЛІЗАЦІЯ ---
# Перетворення даних з JAX DeviceArray у звичайний NumPy для малювання
C_NH4_np = np.array(C_NH4)
C_NO2_np = np.array(C_NO2)
C_NO3_np = np.array(C_NO3)

X_plot = np.linspace(0, L_RIVER / 1000, NX)
Y_plot = np.linspace(0, W_RIVER, NY)

fig, axs = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

# Графік 1: NH4
cs0 = axs[0].contourf(X_plot, Y_plot, C_NH4_np.T, levels=20, cmap=cm.viridis)
axs[0].set_title(r'$NH_4^+$ (Амоній)')
axs[0].set_ylabel('Ширина річки (м)')
axs[0].set_xlabel('Відстань (км)')
fig.colorbar(cs0, ax=axs[0], label='мг/дм³')

# Графік 2: NO2
cs1 = axs[1].contourf(X_plot, Y_plot, C_NO2_np.T, levels=20, cmap=cm.inferno)
axs[1].set_title(r'$NO_2^-$ (Нітрити)')
axs[1].set_xlabel('Відстань (км)')
fig.colorbar(cs1, ax=axs[1], label='мг/дм³')

# Графік 3: NO3
cs2 = axs[2].contourf(X_plot, Y_plot, C_NO3_np.T, levels=20, cmap=cm.plasma)
axs[2].set_title(r'$NO_3^-$ (Нітрати)')
axs[2].set_xlabel('Відстань (км)')
fig.colorbar(cs2, ax=axs[2], label='мг/дм³')

plt.suptitle(f'Моделювання якості води (ADR-N) на TPU\nЧас розрахунку: {total_time:.2f} с')
plt.show()

# Табличний вивід центральної лінії
print("\n--- Профіль концентрацій по центру річки (Y=50м) ---")
print(f"{'Відстань (км)':<15} {'NH4+':<12} {'NO2-':<12} {'NO3-':<12}")
print("-" * 55)
center_idx = NY // 2
for i in range(0, NX, 20): # Виводимо кожну 20-ту точку
    dist = X_plot[i]
    print(f"{dist:<15.1f} {C_NH4_np[i, center_idx]:<12.4f} {C_NO2_np[i, center_idx]:<12.4f} {C_NO3_np[i, center_idx]:<12.4f}")