# @title Моделювання денітрифікації (Версія Matplotlib)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display, Markdown, clear_output
import ipywidgets as widgets
from ipywidgets import interact

# Налаштування стилю графіків (робить їх красивішими)
plt.style.use('seaborn-v0_8-whitegrid')

# --- 1. Константи та параметри ---
R = 8.314   # Дж/моль/К
Ea = 54500.0 # Дж/моль

# Кінетичні параметри
Km = 3.0     # мг N/Л
K_DO_inh = 0.1 # мг/Л

# Константи для Арреніуса
k_ref = 0.683
T_ref_C = 12.0
A = k_ref * np.exp(Ea / (R * (T_ref_C + 273.15)))

# --- 2. Функції розрахунку (Без змін) ---

def calculate_max_rate_arrhenius(T_celsius):
    T_kelvin = T_celsius + 273.15
    if T_celsius < 2.0 or T_celsius > 45.0: return 0.0
    if T_celsius < 5.0: return k_ref * 0.05
    k_T = A * np.exp(-Ea / (R * T_kelvin))
    return k_T

def calculate_denitrification_rate(C_N, T_celsius, DO_concentration):
    k_T = calculate_max_rate_arrhenius(T_celsius)
    f_Cn = C_N / (Km + C_N) if C_N > 0 else 0
    f_DO = K_DO_inh / (K_DO_inh + DO_concentration)
    if DO_concentration > 0.5:
        f_DO *= 0.05
    return k_T * f_Cn * f_DO

# --- 3. Симуляція та Візуалізація (Matplotlib) ---

def run_simulation(T_initial, DO_initial, C_N_initial, Total_time):
    # Параметри часу
    Time_step = 0.1
    time_points = np.arange(0, Total_time + Time_step, Time_step)

    C_N_history = []
    rates_history = []
    current_C_N = C_N_initial

    # Цикл розрахунку
    for t in time_points:
        C_N_history.append(current_C_N)
        if current_C_N > 0:
            rate = calculate_denitrification_rate(current_C_N, T_initial, DO_initial)
            rates_history.append(rate)
            current_C_N -= rate * Time_step
            if current_C_N < 0: current_C_N = 0.0
        else:
            rates_history.append(0)
            current_C_N = 0.0

    # --- Побудова Графіка ---
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Ліва вісь Y (Концентрація) - Синя
    color_conc = 'tab:blue'
    ax1.set_xlabel('Час (години)', fontsize=12)
    ax1.set_ylabel('Концентрація NO3 (мг N/Л)', color=color_conc, fontsize=12)
    ax1.plot(time_points, C_N_history, color=color_conc, linewidth=2.5, label='NO3-N')
    ax1.tick_params(axis='y', labelcolor=color_conc)
    ax1.set_ylim(bottom=0)

    # Права вісь Y (Швидкість) - Червона
    ax2 = ax1.twinx()  # створює другу вісь, що ділить спільну вісь X
    color_rate = 'tab:red'
    ax2.set_ylabel('Швидкість реакції (мг/Л/год)', color=color_rate, fontsize=12)
    ax2.plot(time_points, rates_history, color=color_rate, linestyle='--', linewidth=2, label='Швидкість')
    ax2.tick_params(axis='y', labelcolor=color_rate)
    ax2.set_ylim(bottom=0)

    # Заголовок та сітка
    plt.title(f"Динаміка денітрифікації (T={T_initial}°C, DO={DO_initial} мг/Л)", fontsize=14, pad=15)
    ax1.grid(True, alpha=0.3)

    # Комбінована легенда
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right', frameon=True)

    plt.tight_layout()
    plt.show()

    # --- Таблиця результатів (Pandas) ---
    efficiency = (C_N_initial - C_N_history[-1]) / C_N_initial * 100 if C_N_initial > 0 else 0

    # Створюємо гарний DataFrame
    data = {
        "Параметр": ["Температура (T)", "Кисень (DO)", "Початковий NO3", "Кінцевий NO3", "Ефективність"],
        "Значення": [f"{T_initial} °C", f"{DO_initial} мг/л", f"{C_N_initial} мг/л", f"{C_N_history[-1]:.2f} мг/л", f"{efficiency:.1f} %"],
        "Статус": [
            "Оптимум" if 25 <= T_initial <= 35 else ("Інгібування" if T_initial > 40 else "Норма"),
            "Аноксичні умови" if DO_initial < 0.3 else "Пригнічення киснем",
            "-",
            "-",
            "Висока" if efficiency > 90 else "Низька"
        ]
    }

    df = pd.DataFrame(data)

    # Стилізація таблиці (робимо її красивою в HTML)
    display(Markdown("#### 📊 Підсумкова статистика"))
    display(df.style.hide(axis="index").set_properties(**{'text-align': 'left', 'font-size': '11pt'}))

# --- Запуск ---
display(Markdown("### 🎛️ Модель (Matplotlib + Pandas)"))
interact(run_simulation,
         T_initial=widgets.FloatSlider(value=20.0, min=0.0, max=45.0, step=1.0, description='T (°C)'),
         DO_initial=widgets.FloatSlider(value=0.2, min=0.0, max=2.0, step=0.05, description='DO (мг/Л)'),
         C_N_initial=widgets.FloatSlider(value=30.0, min=5.0, max=100.0, step=5.0, description='NO3 (мг/Л)'),
         Total_time=widgets.IntSlider(value=72, min=24, max=168, step=12, description='Час (год)'));