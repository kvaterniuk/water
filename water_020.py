import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Магічна команда для Jupyter/Colab
# %matplotlib inline

# ==========================================
# 1. Блок ініціалізації вхідних параметрів
# ==========================================
Q_r = 1.2        # Витрата води в річці (куб. м/с)
L_r = 3.4        # БСК5 річкової води (мгО2/л)
C_r = 8.5        # Концентрація кисню в річці (мгО2/л)
T_r = 20.0       # Температура води в річці (°C)

Q_w = 0.2        # Витрата стічних вод (куб. м/с)
L_w = 45.0       # БСК5 у стічних водах (мгО2/л)
C_w = 2.0        # Кисень у стоках (мгО2/л)
T_w = 22.0       # Температура стічних вод (°C)

k1_20 = 0.15     # Швидкість деоксигенації (1/добу)
k2_20 = 0.45     # Швидкість реаерації (1/добу)

v_mps = 0.2      # Швидкість течії (м/с)
distance = 50.0  # Відстань моделювання (км)

# ==========================================
# 2. Розрахунок параметрів змішування
# ==========================================
Q_mix = Q_r + Q_w
L_0 = (Q_r * L_r + Q_w * L_w) / Q_mix
C_0 = (Q_r * C_r + Q_w * C_w) / Q_mix
T_mix = (Q_r * T_r + Q_w * T_w) / Q_mix

# Температурна корекція
k1 = k1_20 * (1.047 ** (T_mix - 20))
k2 = k2_20 * (1.024 ** (T_mix - 20))

# Концентрація насичення (Cs)
C_s = 14.652 - 0.41022 * T_mix + 0.007991 * (T_mix ** 2) - 0.000077774 * (T_mix ** 3)
D_0 = C_s - C_0

print(f"--- Аналітичні характеристики суміші ---")
print(f"БСК0: {L_0:.2f} мгО2/л, РК0: {C_0:.2f} мгО2/л")
print(f"Cs: {C_s:.2f} мгО2/л, k1: {k1:.3f}, k2: {k2:.3f}\n")

# ==========================================
# 3. Диференціальна модель
# ==========================================
def streeter_phelps(y, t, k1, k2):
    L, D = y
    dLdt = -k1 * L
    dDdt = k1 * L - k2 * D
    return [dLdt, dDdt]  # ВИПРАВЛЕНО: Повертаємо список похідних

# Часова шкала
v_mpd = v_mps * 86400
t_max = (distance * 1000) / v_mpd
t = np.linspace(0, t_max, 500)

# Початкові умови
y0 = [L_0, D_0]  # ВИПРАВЛЕНО: Задано початкове БСК та дефіцит

# ==========================================
# 4. Чисельне розв'язання
# ==========================================
solution = odeint(streeter_phelps, y0, t, args=(k1, k2))

L_t = solution[:, 0]
D_t = solution[:, 1]
C_t = C_s - D_t

x = (t * v_mpd) / 1000

# Пошук критичної точки
min_DO_idx = np.argmin(C_t)
crit_distance = x[min_DO_idx]  # ВИПРАВЛЕНО: Додано індексацію
crit_DO = C_t[min_DO_idx]       # ВИПРАВЛЕНО: Додано індексацію

print(f"--- Результати моделювання ---")
print(f"Критична точка: {crit_distance:.2f} км")
print(f"Мінімальний кисень: {crit_DO:.2f} мгО2/л")

# ==========================================
# 5. Візуалізація
# ==========================================

plt.figure(figsize=(12, 6))
plt.plot(x, C_t, 'b-', linewidth=2.5, label='Розчинений кисень (РК)')
plt.plot(x, L_t, 'r--', linewidth=2, label='БСК')
plt.axhline(C_s, color='g', linestyle=':', label='Насичення (Cs)')
plt.axhline(4.0, color='red', linestyle='-.', alpha=0.6, label='ГДК (4.0 мгО2/л)')

plt.plot(crit_distance, crit_DO, 'ro', markersize=8)
plt.annotate(f'Критична точка\n({crit_distance:.1f} км; {crit_DO:.1f} мг/л)',
             xy=(crit_distance, crit_DO), xytext=(crit_distance + 5, crit_DO + 1),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1),
             fontsize=10, bbox=dict(boxstyle="round", fc="white", alpha=0.8))

plt.title('Модель Стрітера-Фелпса: Прогноз стану річки Рів', fontsize=14)
plt.xlabel('Відстань від скиду (км)')
plt.ylabel('Концентрація (мгО2/л)')
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()