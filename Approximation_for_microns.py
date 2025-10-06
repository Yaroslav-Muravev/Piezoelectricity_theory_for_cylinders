# Численная оценка электрического поля, генерируемого пьезоэлектриком, для одиночного цилиндра в воде
# Упрощённая (по порядку величины) модель.
# Модель:
#  - Единичный (доминирующий) механический отклик создаёт напряжение ~ приложенное акустическое давление p на поверхности.
#  - Мы предполагаем, что пьезоэлектрическая связь создаёт поле смещения D ~ d * p.
#  - Непрерывность нормали D на границе даёт E_out = D / eps_ext = d * p / eps_ext. При упрощённом предположении D ≈ d * p
#  - Внутри материала E_int ≈ d * p / eps_int.
#  - Учитываем конечную проводимость воды, используя комплексную диэлектрическую проницаемость eps_ext_complex = eps_ext - i*sigma/omega.
# Тогда E_ext = d * p / eps_ext_complex.
#
# Ограничения:
#  - не учитываются распределение деформации внутри элемента;
#  - не учитывается структурная жёсткость металлокоутинга;
#  - не учитывается взаимная ёмкость/электрическое соединение элементов.

import numpy as np
import pandas as pd

# ------------------------------
# Глобальные константы
# ------------------------------

eps0 = 8.8541878128e-12  # Фарады на метр (пермиитивность вакуума)

# ------------------------------
# Параметры модели / входные данные
# ------------------------------

d_pmV_list = [10.0, 15.0]            # pm/V (10 и 15 pm/V)
R_big_um_list = [5.0, 30.0]          # радиусы большого цилиндра в микронах
pressures_Pa = [1.0, 1e2, 1e3, 1e5]  # Pa: 1, 100, 1k, 100k
eps_int_factor = 1000.0              # eps_int = eps_int_factor * eps0 (ориентировочно для пьезо-керамики)
eps_ext_factor = 80.0                # вода ≈ 80 * eps0
sigma_ext_values = [0.0]             # S/m: проводимость воды (0 — идеализированная чистая вода)
freq = 1e6                           # Гц — акустическая частота (влияет на sigma/omega в комплексной permittivity)

# ------------------------------
# Вычисляемые постоянные
# ------------------------------

eps_int = eps_int_factor * eps0      # внутренняя диэлектрическая проницаемость материала
eps_ext = eps_ext_factor * eps0      # внешняя (вода)
omega = 2 * np.pi * freq             # циклическая частота, рад/с

rows = []  # сюда собираем результаты для последующего DataFrame

for d_pmV in d_pmV_list:
    d = d_pmV * 1e-12  # перевод pm/V -> m/V (единицы SI)
    for R_big_um in R_big_um_list:
        R_big = R_big_um * 1e-6  # m: радиус "большого" цилиндра

        # Геометрические параметры массива (4×4) укладываются в квадрат размером 2*R_big (диаметр)
        S = 2.0 * R_big
        n_side = 4
        s = S / n_side  # шаг сетки (между центрами)
        area_big = np.pi * R_big ** 2
        area_small = area_big / (n_side ** 2)  # площадь одного маленького цилиндра, если суммарная площадь равна большой
        r_small = np.sqrt(area_small / np.pi)  # радиус маленького цилиндра, эквивалентный по площади

        # координаты центров маленьких цилиндров в плоскости: симметрично вокруг нуля
        offsets = (-(S / 2) + s / 2) + np.arange(n_side) * s
        coords = [(x, y) for x in offsets for y in offsets]

        # высота наблюдения h над центром массива: берем небольшую положительную дистанцию,
        # чтобы не попасть в сингулярность при r->0 (точечный заряд). Здесь 0.25 * r_small или минимум 1e-7 м.
        h = max(1e-7, 0.25 * r_small)

        for p in pressures_Pa:
            # упрощённая оценка "связанного заряда" Q = d * p * area
            Q_big = d * p * area_big     # заряд эквивалентной площади большого цилиндра
            Q_small = d * p * area_small # заряд каждого маленького цилиндра

            for sigma_ext in sigma_ext_values:
                # комплексная внешняя проницаемость, учитываем проводимость: eps - i*sigma/omega
                eps_ext_complex = eps_ext - 1j * sigma_ext / omega

                # потенциал большой одиночной «точки» в точке (0,0,h): phi = Q / (4*pi*eps*r)
                # (мы моделируем круговую/цилиндрическую проблему как точечную для порядковых оценок)
                r_big = np.sqrt(h ** 2)  # расстояние от "большого" центра до точки наблюдения
                phi_big_c = Q_big / (4.0 * np.pi * eps_ext_complex * r_big)

                # потенциал массива — суммарно
                phi_array_c = 0 + 0j
                for (x, y) in coords:
                    r = np.sqrt(x * x + y * y + h * h)
                    phi_array_c += Q_small / (4.0 * np.pi * eps_ext_complex * r)

                # простая оценка E_ext = d*p/eps_ext, затем V = E * R
                E_ext_simple = abs(d * p / eps_ext_complex)
                V_single_simple = E_ext_simple * R_big
                V_array_ideal = E_ext_simple * (n_side ** 2) * r_small  # "идеальная" суммир.оценка N * E * r_small

                rows.append({
                    "d (pm/V)": d_pmV,
                    "R_big (um)": R_big_um,
                    "r_small (um)": r_small * 1e6,
                    "pressure (Pa)": p,
                    "sigma_ext (S/m)": sigma_ext,
                    "phi_big (V) (abs)": abs(phi_big_c),
                    "phi_array (V) (abs)": abs(phi_array_c),
                    "ratio (array/big)": abs(phi_array_c) / (abs(phi_big_c) + 1e-30),  # +eps для защиты от деления на 0
                    "V_single_simple (V)": V_single_simple,
                    "V_array_ideal (V)": V_array_ideal
                })

df = pd.DataFrame(rows)

print("Quick summary (magnitudes).")
for d_pmV in d_pmV_list:
    print(f"\n--- d = {d_pmV} pm/V ---")
    sub = df[df["d (pm/V)"] == d_pmV]
    print(sub.head(12).to_string(index=False))
