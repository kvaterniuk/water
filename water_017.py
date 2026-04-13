import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# --- 1. ПАРАМЕТРИ МОДЕЛІ ---

# Геометричні та гідравлічні параметри
L_RIVER = 40000.0  # Довжина (м)
W_RIVER = 100.0    # Ширина (м)
H_RIVER = 2.5      # Глибина (м)

NX = 200  # Вузли X
NY = 50   # Вузли Y
dx = L_RIVER / NX
dy = W_RIVER / NY

# Гідравліка
V = 0.4    # Швидкість (м/с)
DL = 15.0  # Поздовжня дисперсія
DT = 0.5   # Поперечна дисперсія

# Кінетика (перерахунок в 1/с)
DAYS_TO_SEC = 86400.0
T_WATER = 16.0
THETA = 1.07

def k_T(k_20):
    return k_20 * (THETA ** (T_WATER - 20.0))

k_N1 = k_T(0.04 / DAYS_TO_SEC)    # NH4 -> NO2
k_N2 = k_T(0.05 / DAYS_TO_SEC)    # NO2 -> NO3
k_D_V = k_T(0.01 / DAYS_TO_SEC) / H_RIVER # Денітрифікація (об'ємна)
k_Sor = 0.001 / DAYS_TO_SEC       # Сорбція NH4

# Джерела (мг/дм³)
C_SOURCE_NH4 = 25.0
DIFFUSE_LOAD_NO3 = 0.000001 # Зменшив для реалістичності (кг/м3/с -> мг/л/с потребує уваги до одиниць)
# Якщо DIFFUSE_LOAD_NO3 в кг/м3/с, то для мг/л/с треба помножити на 1000.
# Приймемо, що це вже приведена величина джерела концентрації на секунду.

# --- 2. ЧИСЕЛЬНИЙ МЕТОД ---

# Стабільність
dt_max_diff = 0.5 * min(dx**2 / DL, dy**2 / DT)
dt_max_adv = dx / V
DT_SIM = min(dt_max_diff, dt_max_adv) * 0.9 # Коефіцієнт безпеки 0.9
if DT_SIM > 100: DT_SIM = 100.0 # Обмеження зверху

print(f"Крок по часу (dt): {DT_SIM:.4f} с")

# Ініціалізація полів
C_NH4 = np.zeros((NX, NY))
C_NO2 = np.zeros((NX, NY))
C_NO3 = np.zeros((NX, NY))

# Вхідні граничні умови (Inlet, x=0)
C_NH4[0, :] = 6.0
C_NO2[0, :] = 0.01
C_NO3[0, :] = 0.1

# Координати скиду (індекси)
src_x = 5 # Змістив трохи далі від самого краю для наочності
src_y_start = int(NY * 0.4)
src_y_end = int(NY * 0.6)

# --- 3. СИМУЛЯЦІЯ (Векторизована) ---

MAX_ITER = 50000 # Достатньо для стабілізації при хорошому кроці
TOLERANCE = 1e-8

for k in range(MAX_ITER):
    # Копіюємо старі значення для розрахунку
    # (Використовуємо .copy(), щоб не змінювати масив під час читання)
    NH4_old = C_NH4.copy()
    NO2_old = C_NO2.copy()
    NO3_old = C_NO3.copy()

    # --- ВЕКТОРИЗАЦІЯ ---
    # Ми працюємо зі зрізами [1:-1, 1:-1] (внутрішні вузли)
    # i відповідає [1:-1], i-1 -> [:-2], i+1 -> [2:]

    # 1. Лапласіан (Дифузія)
    # Dxx = (C[i+1] - 2C[i] + C[i-1]) / dx^2
    def laplacian(C):
        d2x = (C[2:, 1:-1] - 2*C[1:-1, 1:-1] + C[:-2, 1:-1]) / dx**2
        d2y = (C[1:-1, 2:] - 2*C[1:-1, 1:-1] + C[1:-1, :-2]) / dy**2
        return DL * d2x + DT * d2y

    # 2. Адвекція (Upwind scheme: V > 0)
    # Adv = -V * (C[i] - C[i-1]) / dx
    def advection(C):
        return -V * (C[1:-1, 1:-1] - C[:-2, 1:-1]) / dx

    # --- Розрахунок змін (Rates) ---

    # NH4
    # Кінетика: -k1*NH4 - kSor*NH4
    R_NH4 = - (k_N1 + k_Sor) * NH4_old[1:-1, 1:-1]
    dNH4 = laplacian(NH4_old) + advection(NH4_old) + R_NH4
    C_NH4[1:-1, 1:-1] = NH4_old[1:-1, 1:-1] + DT_SIM * dNH4

    # NO2
    # Кінетика: +k1*NH4 - k2*NO2
    R_NO2 = k_N1 * NH4_old[1:-1, 1:-1] - k_N2 * NO2_old[1:-1, 1:-1]
    dNO2 = laplacian(NO2_old) + advection(NO2_old) + R_NO2
    C_NO2[1:-1, 1:-1] = NO2_old[1:-1, 1:-1] + DT_SIM * dNO2

    # NO3
    # Кінетика: +k2*NO2 - kD*NO3 + DiffuseSource
    R_NO3 = k_N2 * NO2_old[1:-1, 1:-1] - k_D_V * NO3_old[1:-1, 1:-1] + DIFFUSE_LOAD_NO3
    dNO3 = laplacian(NO3_old) + advection(NO3_old) + R_NO3
    C_NO3[1:-1, 1:-1] = NO3_old[1:-1, 1:-1] + DT_SIM * dNO3

    # --- ГРАНИЧНІ УМОВИ (Оновлення країв) ---

    # 1. Точкове джерело (Dirichlet condition - фіксуємо концентрацію)
    # Застосовуємо після розрахунку фізики, щоб воно діяло як постійне джерело
    C_NH4[src_x, src_y_start:src_y_end] = C_SOURCE_NH4

    # 2. Береги (y=0, y=W) -> Умова Неймана (dC/dy = 0 -> C[0]=C[1])
    # Верхній берег
    C_NH4[:, 0] = C_NH4[:, 1]
    C_NO2[:, 0] = C_NO2[:, 1]
    C_NO3[:, 0] = C_NO3[:, 1]
    # Нижній берег
    C_NH4[:, -1] = C_NH4[:, -2]
    C_NO2[:, -1] = C_NO2[:, -2]
    C_NO3[:, -1] = C_NO3[:, -2]

    # 3. Вихід (x=L) -> Умова Неймана (dC/dx = 0 -> C[-1]=C[-2])
    # Дозволяє забрудненню вільно виходити з домену
    C_NH4[-1, :] = C_NH4[-2, :]
    C_NO2[-1, :] = C_NO2[-2, :]
    C_NO3[-1, :] = C_NO3[-2, :]

    # --- ПЕРЕВІРКА ЗБІЖНОСТІ ---
    if k % 1000 == 0:
        max_change = np.max(np.abs(C_NH4 - NH4_old))
        print(f"Ітерація {k}, Макс. зміна NH4: {max_change:.8f}")
        if max_change < TOLERANCE and k > 100:
            print("Стаціонарний стан досягнуто.")
            break

# --- 4. ВІЗУАЛІЗАЦІЯ ---
X_plot = np.linspace(0, L_RIVER / 1000, NX)
Y_plot = np.linspace(0, W_RIVER, NY)

fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True) # Змінив на 3 рядки, 1 стовпець

# 1. Амоній
im0 = axs[0].contourf(X_plot, Y_plot, C_NH4.T, levels=50, cmap=cm.viridis)
axs[0].set_title(r'$C_{NH_4}$ (Амоній), мг/дм³')
axs[0].set_ylabel('Ширина (м)')
fig.colorbar(im0, ax=axs[0])

# 2. Нітрити (Виправив індекс з 22 на 1)
im1 = axs[1].contourf(X_plot, Y_plot, C_NO2.T, levels=50, cmap=cm.inferno)
axs[1].set_title(r'$C_{NO_2}$ (Нітрити), мг/дм³')
axs[1].set_ylabel('Ширина (м)')
fig.colorbar(im1, ax=axs[1])

# 3. Нітрати (Виправив індекс з 23 на 2)
im2 = axs[2].contourf(X_plot, Y_plot, C_NO3.T, levels=50, cmap=cm.plasma)
axs[2].set_title(r'$C_{NO_3}$ (Нітрати), мг/дм³')
axs[2].set_ylabel('Ширина (м)')
axs[2].set_xlabel('Відстань (км)')
fig.colorbar(im2, ax=axs[2])

plt.suptitle(f'Моделювання якості води (Південний Буг, ділянка {L_RIVER/1000} км)')
plt.tight_layout()
plt.show()

# Вивід профілю
print("\n--- Профіль по центру річки (останній крок) ---")
center_idx = NY // 2
step = NX // 10
print(f"{'X (км)':<10} | {'NH4':<10} | {'NO2':<10} | {'NO3':<10}")
for i in range(0, NX, step):
    print(f"{X_plot[i]:<10.1f} | {C_NH4[i, center_idx]:<10.4f} | {C_NO2[i, center_idx]:<10.4f} | {C_NO3[i, center_idx]:<10.4f}")