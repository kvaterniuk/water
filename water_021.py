import numpy as np
import matplotlib.pyplot as plt

# 1. Параметри русла
U = 0.3
E = 5.0
L_river = 25000
dx = 50
nx = int(L_river / dx) + 1
x = np.linspace(0, L_river, nx)

# 2. Константи (у 1/с)
K1 = 0.25 / 86400.0
Kn = 0.18 / 86400.0
K2 = 0.65 / 86400.0
alpha = 4.57
SOD = 0.5 / 86400.0

# 3. Граничні умови
L_bg, N_bg, DO_bg = 2.5, 0.4, 8.5
DO_sat = 9.2
D_bg = DO_sat - DO_bg

L_mix, N_mix, DO_mix = 14.0, 5.5, 3.5
D_mix = DO_sat - DO_mix

# 4. Ініціалізація
L = np.full(nx, L_bg)
N = np.full(nx, N_bg)
D = np.full(nx, D_bg)

# Початкова умова на вході
L[0], N[0], D[0] = L_mix, N_mix, D_mix

# 5. НОВИЙ розрахунок стабільного кроку dt
# Об'єднаний критерій стійкості для схеми Upwind + Diffusion
denom = (U / dx) + (2 * E / dx**2) + max(K1, Kn, K2)
dt = 0.9 * (1.0 / denom)  # Коефіцієнт 0.9 для запасу міцності

T_total = 86400 * 3
nt = int(T_total / dt)

print(f"--- Стабільні параметри ---")
print(f"Розрахунковий крок dt: {dt:.2f} с (було 133.33 с)")
print(f"Кількість ітерацій: {nt}")

# 6. Цикл розрахунку
for n in range(nt):
    L_old, N_old, D_old = L.copy(), N.copy(), D.copy()

    # Векторизоване оновлення внутрішніх вузлів
    # Адвекція (Upwind) + Дисперсія (Central) + Реакція
    L[1:-1] = L_old[1:-1] \
        - U * (dt/dx) * (L_old[1:-1] - L_old[:-2]) \
        + E * (dt/dx**2) * (L_old[2:] - 2*L_old[1:-1] + L_old[:-2]) \
        - dt * K1 * L_old[1:-1]

    N[1:-1] = N_old[1:-1] \
        - U * (dt/dx) * (N_old[1:-1] - N_old[:-2]) \
        + E * (dt/dx**2) * (N_old[2:] - 2*N_old[1:-1] + N_old[:-2]) \
        - dt * Kn * N_old[1:-1]

    D[1:-1] = D_old[1:-1] \
        - U * (dt/dx) * (D_old[1:-1] - D_old[:-2]) \
        + E * (dt/dx**2) * (D_old[2:] - 2*D_old[1:-1] + D_old[:-2]) \
        + dt * (K1 * L_old[1:-1] + Kn * alpha * N_old[1:-1] - K2 * D_old[1:-1] + SOD)

    # Захист від фізично неможливих (від'ємних) значень
    L = np.clip(L, 0, None)
    N = np.clip(N, 0, None)

    # Граничні умови
    L[0], N[0], D[0] = L_mix, N_mix, D_mix  # Вхід
    L[-1], N[-1], D[-1] = L[-2], N[-2], D[-2] # Вихід

DO = DO_sat - D

# 7. Візуалізація (скорочено)
plt.figure(figsize=(10, 6))
plt.plot(x/1000, L, label='БСК5 (BOD)')
plt.plot(x/1000, DO, label='Кисень (DO)', color='cyan')
plt.axhline(4.0, color='red', linestyle='--', label='Мін. норма DO')
plt.title('Стабільна модель забруднення річки')
plt.xlabel('Відстань, км')
plt.ylabel('Концентрація, мг/л')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()