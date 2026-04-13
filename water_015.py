# Математична модель нітрифікації
# Імпорт необхідних бібліотек
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from IPython.display import display, Markdown

# --- 1. Кінетичні та стехіометричні константи ---
class NitrificationConstants:
    T_ref = 20.0        # Референсна температура (°C)
    THETA = 1.072       # Температурний коефіцієнт (більш точний для нітрифікації, US EPA)

    # Максимальні швидкості росту (при 20°C, 1/год - питомі швидкості)
    # Тут ми використовуємо швидкість утилізації субстрату (мг N / мг VSS / год)
    # Для спрощеної моделі без біомаси, це умовні коефіцієнти швидкості (мг N/Л/год)
    R_max_AOB_ref = 0.8
    R_max_NOB_ref = 0.9

    # Константи напівнасичення (мг N/Л)
    K_S_NH4 = 1.0
    K_S_NO2 = 1.0

    # Константи напівнасичення для кисню (мг O2/Л)
    K_DO_AOB = 0.5
    K_DO_NOB = 0.8

    # Стехіометрія (ОНОВЛЕНО)
    # 1. NH4 -> NO2: 3.43 г O2 / г N
    # 2. NO2 -> NO3: 1.14 г O2 / г N
    # Разом: 4.57 г O2 / г N
    Y_O2_AOB = 3.43
    Y_O2_NOB = 1.14

    ALR_CACO3 = 7.14  # Потреба в лужності (г CaCO3 / г N окисненого NH4)

# --- 2. Функції кінетичної моделі ---

def calculate_rates(C_NH4, C_NO2, T_celsius, C_DO):
    """
    Розрахунок швидкостей реакцій AOB та NOB.
    Повертає кортеж (Rate_AOB, Rate_NOB)
    """
    Const = NitrificationConstants

    # 1. Температурна корекція
    if T_celsius < 4.0:
        f_T = 0.0
    else:
        f_T = Const.THETA ** (T_celsius - Const.T_ref)

    # 2. Кінетика для AOB (Амоній -> Нітрит)
    f_S_NH4 = C_NH4 / (Const.K_S_NH4 + C_NH4) if C_NH4 > 0 else 0
    f_DO_AOB = C_DO / (Const.K_DO_AOB + C_DO)
    R_AOB = Const.R_max_AOB_ref * f_T * f_S_NH4 * f_DO_AOB

    # 3. Кінетика для NOB (Нітрит -> Нітрат)
    f_S_NO2 = C_NO2 / (Const.K_S_NO2 + C_NO2) if C_NO2 > 0 else 0
    f_DO_NOB = C_DO / (Const.K_DO_NOB + C_DO)
    R_NOB = Const.R_max_NOB_ref * f_T * f_S_NO2 * f_DO_NOB

    return R_AOB, R_NOB

# --- 3. Параметри та Симуляція ---

# Вхідні параметри
params = {
    'T_initial': 18.0,       # Температура (°C)
    'DO_conc': 4.5,          # Кисень (мг/Л)
    'NH4_init': 30.0,        # Початковий амоній (мг N/Л)
    'NO2_init': 0.0,         # Початковий нітрит
    'NO3_init': 5.0,         # Початковий нітрат
    'Total_time': 120.0,     # Час (год)
    'dt': 0.05               # Крок часу (год) - менший крок для точності
}

# Ініціалізація змінних
time_points = np.arange(0, params['Total_time'] + params['dt'], params['dt'])

# Створення словника для збереження історії (швидше, ніж append у циклі)
history = {k: np.zeros(len(time_points)) for k in
           ['Time', 'NH4_N', 'NO2_N', 'NO3_N', 'Alk_Consumed', 'DO_Consumed']}

# Початкові стани
C_NH4 = params['NH4_init']
C_NO2 = params['NO2_init']
C_NO3 = params['NO3_init']
DO_consumed_total = 0.0
Alk_consumed_total = 0.0

# Основний цикл (Метод Ейлера з перевіркою балансу мас)
for i, t in enumerate(time_points):
    # Збереження стану
    history['Time'][i] = t
    history['NH4_N'][i] = C_NH4
    history['NO2_N'][i] = C_NO2
    history['NO3_N'][i] = C_NO3
    history['Alk_Consumed'][i] = Alk_consumed_total
    history['DO_Consumed'][i] = DO_consumed_total

    # Розрахунок потенційних швидкостей
    r_aob, r_nob = calculate_rates(C_NH4, C_NO2, params['T_initial'], params['DO_conc'])

    # --- Обмеження мас (Mass Balance Check) ---
    # Не можна спожити більше NH4, ніж є
    amount_NH4_oxidized = min(C_NH4, r_aob * params['dt'])

    # Оновлюємо реальну швидкість AOB на основі доступного субстрату
    real_rate_AOB = amount_NH4_oxidized / params['dt']

    # Не можна спожити більше NO2, ніж є (поточний + те, що прийде від AOB)
    # У явному методі ми зазвичай беремо поточний C_NO2, але врахуємо прихід для стабільності
    available_NO2 = C_NO2 + amount_NH4_oxidized # Припустимо миттєву доступність на кроці для спрощення
    amount_NO2_oxidized = min(available_NO2, r_nob * params['dt'])

    real_rate_NOB = amount_NO2_oxidized / params['dt']

    # --- Оновлення концентрацій ---
    C_NH4 -= amount_NH4_oxidized
    C_NO2 += (amount_NH4_oxidized - amount_NO2_oxidized)
    C_NO3 += amount_NO2_oxidized

    # --- Розрахунок ресурсів (Виправлена фізика) ---
    # 1. Кисень на окиснення NH4 -> NO2
    do_step_1 = amount_NH4_oxidized * NitrificationConstants.Y_O2_AOB
    # 2. Кисень на окиснення NO2 -> NO3
    do_step_2 = amount_NO2_oxidized * NitrificationConstants.Y_O2_NOB

    DO_consumed_total += (do_step_1 + do_step_2)

    # Лужність витрачається тільки при окисненні NH4 (вивільнення H+)
    Alk_consumed_total += amount_NH4_oxidized * NitrificationConstants.ALR_CACO3

    # Запобіжник від "мінусового нуля"
    if C_NH4 < 1e-6: C_NH4 = 0.0
    if C_NO2 < 1e-6: C_NO2 = 0.0

# Створення DataFrame
df = pd.DataFrame(history)

# --- 4. Візуалізація ---
display(Markdown(f"## Результати моделювання нітрифікації"))
display(Markdown(f"""
**Параметри середовища:** Temp = {params['T_initial']}°C, DO = {params['DO_conc']} мг/Л.
"""))

# Графік 1: Азотні сполуки
fig_nitrogen = go.Figure()
fig_nitrogen.add_trace(go.Scatter(x=df['Time'], y=df['NH4_N'], name='Амоній (NH4)', line=dict(color='crimson', width=3)))
fig_nitrogen.add_trace(go.Scatter(x=df['Time'], y=df['NO2_N'], name='Нітрит (NO2)', line=dict(color='orange', width=2, dash='dash'), fill='tozeroy', fillcolor='rgba(255, 165, 0, 0.1)'))
fig_nitrogen.add_trace(go.Scatter(x=df['Time'], y=df['NO3_N'], name='Нітрат (NO3)', line=dict(color='forestgreen', width=3)))

fig_nitrogen.update_layout(
    title='Динаміка перетворення форм Азоту',
    xaxis_title='Час (години)',
    yaxis_title='Концентрація (мг N/Л)',
    template="plotly_white",
    hovermode="x unified"
)
fig_nitrogen.show()

# Графік 2: Ресурси
fig_resources = go.Figure()
fig_resources.add_trace(go.Scatter(x=df['Time'], y=df['DO_Consumed'], name='Спожитий O2', line=dict(color='blue')))
fig_resources.add_trace(go.Scatter(x=df['Time'], y=df['Alk_Consumed'], name='Спожита лужність', line=dict(color='purple'), yaxis='y2'))

fig_resources.update_layout(
    title='Накопичене споживання ресурсів',
    xaxis_title='Час (години)',
    yaxis=dict(title='Спожитий Кисень (мг O2/Л)', titlefont=dict(color='blue')),
    yaxis2=dict(title='Спожита Лужність (мг CaCO3/Л)', titlefont=dict(color='purple'), overlaying='y', side='right'),
    template="plotly_white",
    hovermode="x unified"
)
fig_resources.show()

# Фінальна таблиця
final_row = df.iloc[-1]
display(Markdown(f"""
### Підсумки процесу (через {params['Total_time']} год):
| Показник | Значення | Од. виміру |
| :--- | :--- | :--- |
| **Залишковий Амоній** | {final_row['NH4_N']:.2f} | мг N/Л |
| **Максимальний Нітрит** | {df['NO2_N'].max():.2f} | мг N/Л |
| **Утворений Нітрат** | {final_row['NO3_N']:.2f} | мг N/Л |
| **Всього спожито O2** | {final_row['DO_Consumed']:.2f} | мг O2/Л |
| **Всього спожито Лужності** | {final_row['Alk_Consumed']:.2f} | мг CaCO3/Л |
"""))