# Код для вычисления дебаевского радиуса (lambda_D) и граничной частоты релаксации (f_c = sigma/(2*pi*eps))
#
# Функции:
#   debye_length(concentration_M, epsilon_r=80.0, T=298.15, z2_mean=1.0)
#       concentration_M: концентрация (моль/литр) для 1:1 электролита; при общем составе
#       можно заменить z2_mean = sum(c_i * z_i^2)/sum(c_i) (но неизвестен состав спиномозговой жидкости)
#       Возвращает lambda_D в метрах.
#
#   relaxation_frequency(sigma, epsilon_r=80.0)
#       sigma: проводимость (S/m)
#       Возвращает f_c = sigma/(2*pi*epsilon)

import numpy as np
import pandas as pd

epsilon_0 = 8.8541878128e-12  # F/m
k_B = 1.380649e-23             # J/K
e_charge = 1.602176634e-19     # C
N_A = 6.02214076e23            # 1/mol

def debye_length(concentration_M, epsilon_ext=80.0, T=298.15, z2_factor=2.0):
    """
    Расчёт Debye length для 1:1 электролита или приближённой ионной силы.
    concentration_M: концентрация в моль/л (M). Будет переведена в mol/m^3.
    epsilon_ext: относительная диэлектрическая проницаемость среды.
    T: температура в К.
    z2_factor: фактор суммирования z_i^2 * c_i; для 1:1 электролита z2_factor = 2 (т.к. z^2 * (c_+ + c_-) = 1^2*(c)+1^2*(c) = 2c).
               В общем случае по-другому.
               If concentration_M is the molarity for a 1:1 electrolyte, set z2_factor=2.
    Returns lambda_D in meters.
    """
    # перевод концентрации в mol/m^3
    c_m3 = concentration_M * 1000.0
    eps = epsilon_ext * epsilon_0
    # Для 1:1 электролита: sum_i z_i^2 n_i0 = 2 * n0
    # в терминах молярности: n0 = N_A * c_m3
    denom = z2_factor * N_A * c_m3 * e_charge**2
    # lambda_D^2 = eps * k_B T / (sum_i z_i^2 e^2 n_i0)
    lambda_D = np.sqrt(eps * k_B * T / denom)
    return lambda_D

def relaxation_frequency(sigma, epsilon_ext=80.0):
    """
    Рассчитать граничную частоту f_c = sigma / (2*pi*epsilon)
    sigma: проводимость (S/m)
    epsilon_ext: относительная диэлектрическая проницаемость среды
    Возвращает f_c в Гц.
    """
    eps = epsilon_ext * epsilon_0
    f_c = sigma / (2 * np.pi * eps)
    return f_c

media = [
    ("Sea water (approx)", 0.7, 4.0),      # concentration M, sigma S/m
    ("Typical fresh (1e-3 M)", 1e-3, 1e-3)
]

rows = []
for name, concM, sigma in media:
    lam = debye_length(concM, epsilon_ext=80.0, T=298.15, z2_factor=2.0)
    fc = relaxation_frequency(sigma, epsilon_ext=80.0)
    rows.append({
        "medium": name,
        "concentration_M (mol/L)": concM,
        "conductivity sigma (S/m)": sigma,
        "Debye length lambda_D (nm)": lam*1e9,
        "relaxation freq f_c (KHz)": fc/1e3
    })

df = pd.DataFrame(rows)
print(df.to_string(index=False))