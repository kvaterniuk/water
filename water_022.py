# Імпорт базових наукових бібліотек, оптимізованих для Google Colab
import numpy as np
import matplotlib.pyplot as plt

class StreeterPhelpsModel:
    """
    Математична модель Стрітера-Фелпса для імітаційного моделювання
    динаміки органічного забруднення та кисневого режиму річкових екосистем.
    """
    def __init__(self, L0, D0, k1_20, k2_20, velocity_m_s, temp_celsius):
        """
        Ініціалізація гідрохімічних та гідрологічних параметрів ділянки.

        Вхідні аргументи:
        L0 : float - Початкове біохімічне споживання кисню (БСК) у місці змішування, мг/л
        D0 : float - Початковий дефіцит розчиненого кисню, мг/л
        k1_20 : float - Константа швидкості дезоксигенації за стандартних 20°C (1/доба)
        k2_20 : float - Константа швидкості реаерації за стандартних 20°C (1/доба)
        velocity_m_s : float - Середня швидкість річкового потоку, м/с
        temp_celsius : float - Фактична температура води, °C
        """
        # Збереження базових параметрів моделі
        self.L0 = L0
        self.D0 = D0
        self.U_m_s = velocity_m_s
        # Конвертація швидкості з м/с у км/доба для узгодження розмірностей часу (доба)
        self.U_km_d = velocity_m_s * 60 * 60 * 24 / 1000
        self.T = temp_celsius

        # Застосування термодинамічних поправок Арреніуса для констант швидкості
        theta_1 = 1.047 # температурний коефіцієнт для бактеріального розкладу
        theta_2 = 1.024 # температурний коефіцієнт для фізичної дифузії газу

        self.k1 = k1_20 * (theta_1 ** (self.T - 20))
        self.k2 = k2_20 * (theta_2 ** (self.T - 20))

        # Обчислення насичення киснем за емпіричним поліномом третього степеня
        self.Cs = self._calc_saturation_DO(self.T)

    def _calc_saturation_DO(self, T):
        """Приватний метод для обчислення максимальної розчинності кисню."""
        # Використання стандартного поліноміального рівняння
        return 14.652 - 0.41022 * T + 0.007991 * (T**2) - 0.000077774 * (T**3)

    def run_simulation(self, total_distance_km, points=1000):
        """
        Векторизований розрахунок аналітичних розв'язків системи диференціальних рівнянь.

        Повертає:
        Кортеж масивів NumPy: координати X (км), БСК (L), Дефіцит (D), Розчинений кисень (C)
        """
        # Генерація просторової сітки
        x_array = np.linspace(0, total_distance_km, points)

        # Трансформація просторової координати у часову на основі гіпотези поршневого потоку
        time_array = x_array / self.U_km_d # вимірюється в добах

        # 1. Експоненціальний розпад органічних речовин (БСК)
        L_array = self.L0 * np.exp(-self.k1 * time_array)

        # 2. Розрахунок дефіциту кисню за рівнянням Стрітера-Фелпса
        coefficient = (self.k1 * self.L0) / (self.k2 - self.k1)
        sag_term = np.exp(-self.k1 * time_array) - np.exp(-self.k2 * time_array)
        initial_deficit_term = self.D0 * np.exp(-self.k2 * time_array)

        D_array = coefficient * sag_term + initial_deficit_term

        # 3. Фактична концентрація розчиненого кисню
        C_array = self.Cs - D_array

        return x_array, L_array, D_array, C_array

    def plot_sag_curve(self, x_array, L_array, C_array):
        """
        Побудова професійної візуалізації результатів. Оптимізовано для відображення
        в комірках Google Colab.
        """
        plt.figure(figsize=(14, 8))

        # Основні лінії динаміки
        plt.plot(x_array, C_array, label='Розчинений кисень (О2)', color='#1f77b4', linewidth=3)
        plt.plot(x_array, L_array, label='Органічне забруднення (БСК)', color='#d62728', linestyle='--', linewidth=2.5)

        # Лінії екологічних меж
        plt.axhline(y=self.Cs, color='#2ca02c', linestyle='-.', alpha=0.7,
                    label=f'Граничне насичення ({self.Cs:.2f} мг/л)')
        plt.axhline(y=4.0, color='black', linestyle=':', alpha=0.8,
                    label='Санітарно-екологічний мінімум (4.0 мг/л)')

        # Ідентифікація "Кризової точки" (мінімум розчиненого кисню)
        idx_min_do = np.argmin(C_array)
        min_do_val = C_array[idx_min_do]
        min_x_val = x_array[idx_min_do]

        # Маркування кризової точки на графіку
        plt.scatter(min_x_val, min_do_val, color='darkred', s=100, zorder=5)
        plt.annotate(
            f'Критична точка (Киснева яма)\nКонцентрація: {min_do_val:.2f} мг/л\nВідстань: {min_x_val:.1f} км',
            xy=(min_x_val, min_do_val),
            xytext=(min_x_val + 5, min_do_val + 2),
            arrowprops=dict(facecolor='black', arrowstyle='->', lw=1.5),
            fontsize=12, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9)
        )

        # Оформлення графіка
        plt.title('Моделювання кисневого режиму р. Дністер після транскордонного скиду', fontsize=16, fontweight='bold')
        plt.xlabel('Відстань за течією від джерела забруднення (км)', fontsize=14)
        plt.ylabel('Концентрація (мг/л)', fontsize=14)
        plt.grid(True, which='both', linestyle='--', alpha=0.4)
        plt.legend(loc='center right', fontsize=12)

        # Налаштування відступів для Colab
        plt.tight_layout()
        plt.show()

# ==============================================================================
# Практичне використання моделі для аналізу транскордонної ділянки
# ==============================================================================

# Сценарій: Імітація скиду висококонцентрованих стоків з очисних споруд
# параметри гіпотетичні, але наближені до типових кризових ситуацій на річці
scenario_params = {
    'L0': 20.0,              # Високе БСК після змішування стічних вод і річки
    'D0': 2.5,               # Початковий кисневий дефіцит
    'k1_20': 0.25,           # Константа дезоксигенації (помірна органіка)
    'k2_20': 0.55,           # Константа реаерації (типова для середньої течії)
    'velocity_m_s': 0.6,     # Швидкість течії (вплив гідроенергетики)
    'temp_celsius': 22.0     # Літня межень
}

# Створення інстанції (екземпляру) моделі
dniester_model = StreeterPhelpsModel(**scenario_params)

# Запуск симуляції на ділянці довжиною 150 км
distance_km = 150
x_dist, bod_vals, deficit_vals, do_vals = dniester_model.run_simulation(distance_km)

# Генерація інтерактивного графіка у Google Colab
dniester_model.plot_sag_curve(x_dist, bod_vals, do_vals)