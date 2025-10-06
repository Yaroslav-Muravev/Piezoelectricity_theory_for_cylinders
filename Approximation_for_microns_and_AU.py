import numpy as np
import pandas as pd

# ------------------------------
# Глобальные константы
# ------------------------------

eps0 = 8.8541878128e-12  # F/m
pi = np.pi

# ------------------------------
# Параметры модели / входные данные
# ------------------------------

d_pmV_list = [10.0, 15.0]            # pm/V (10 и 15 pm/V)
R_big_um_list = [5.0, 30.0]          # радиусы большого цилиндра в микронах
pressures_Pa = [1.0, 1e2, 1e3, 1e5]  # Pa: 1, 100, 1k, 100k
eps_ext = 80.0 * eps0                # вода ≈ 80 * eps0
sigma_ext_values = [0.0]             # S/m: проводимость воды (0 — идеализированная чистая вода)
freq = 1e6                           # Гц — акустическая частота (влияет на sigma/omega в комплексной permittivity)
omega = 2 * np.pi * freq
n_side = 4
N = n_side ** 2
chi_mech_values = [1.0, 0.6]

def conductor_voltage_estimate(Q_tot, eps_complex, R_eff):
    C = 8.0 * eps_complex * R_eff
    V = Q_tot / C
    return V

rows = []
for d_pmV in d_pmV_list:
    d = d_pmV * 1e-12
    for R_big_um in R_big_um_list:
        R_big = R_big_um * 1e-6
        S = 2.0 * R_big
        s = S / n_side
        area_big = pi * R_big ** 2
        area_small = area_big / N
        r_small = np.sqrt(area_small / pi)
        offsets = (-(S / 2) + s / 2) + np.arange(n_side) * s
        coords = [(x, y) for x in offsets for y in offsets]
        h = max(1e-7, 0.25 * r_small)

        for p in pressures_Pa:
            Q_big_raw = d * p * area_big
            Q_small_raw = d * p * area_small
            for sigma_ext in sigma_ext_values:
                eps_ext_complex = eps_ext - 1j * sigma_ext / omega
                # uncoated potentials
                r_big = np.sqrt(h ** 2)
                phi_big_c = Q_big_raw / (4.0 * pi * eps_ext_complex * r_big)
                phi_array_c = sum(
                    Q_small_raw / (4.0 * pi * eps_ext_complex * np.sqrt(x * x + y * y + h * h)) for (x, y) in coords)
                E_ext_simple = abs(d * p / eps_ext_complex)
                V_single_simple = E_ext_simple * R_big
                V_array_ideal = E_ext_simple * N * r_small
                # coated base (chi=1)
                Q_tot = Q_big_raw
                V_big_coated = conductor_voltage_estimate(Q_tot, eps_ext_complex, R_big)
                V_array_coated_common = conductor_voltage_estimate(Q_tot, eps_ext_complex, S * 0.5)
                # separate coated small conductors (not connected)
                C_i = 8.0 * eps_ext_complex * r_small
                V_each_coated = Q_small_raw / C_i
                phi_array_coated_separate = sum(
                    (Q_small_raw) / (4.0 * pi * eps_ext_complex * np.sqrt(x * x + y * y + h * h)) for (x, y) in coords)

                rows.append({
                    "d (pm/V)": d_pmV,
                    "R_big (um)": R_big_um,
                    "pressure (Pa)": p,
                    "sigma_ext (S/m)": sigma_ext,
                    "phi_big_uncoated (V)": abs(phi_big_c),
                    "phi_array_uncoated (V)": abs(phi_array_c),
                    "V_single_simple (V)": V_single_simple,
                    "V_array_ideal (V)": V_array_ideal,
                    "V_big_coated (V)": abs(V_big_coated),
                    "V_array_coated_common (V)": abs(V_array_coated_common),
                    "V_each_coated (V)": abs(V_each_coated),
                    "phi_array_coated_separate (V)": abs(phi_array_coated_separate)
                })
                # with mechanical reduction chi
                for chi in chi_mech_values:
                    Q_big_eff = chi * Q_big_raw
                    Q_small_eff = chi * Q_small_raw
                    V_big_coated_chi = conductor_voltage_estimate(Q_big_eff, eps_ext_complex, R_big)
                    V_array_coated_common_chi = conductor_voltage_estimate(Q_big_eff, eps_ext_complex, S * 0.5)
                    V_each_coated_chi = Q_small_eff / (8.0 * eps_ext_complex * r_small)
                    phi_array_coated_separate_chi = sum(
                        (Q_small_eff) / (4.0 * pi * eps_ext_complex * np.sqrt(x * x + y * y + h * h)) for (x, y) in
                        coords)
                    rows.append({
                        "d (pm/V)": d_pmV,
                        "R_big (um)": R_big_um,
                        "pressure (Pa)": p,
                        "sigma_ext (S/m)": sigma_ext,
                        "chi_mech": chi,
                        "phi_big_uncoated (V)": abs(phi_big_c),
                        "phi_array_uncoated (V)": abs(phi_array_c),
                        "V_single_simple (V)": V_single_simple,
                        "V_array_ideal (V)": V_array_ideal,
                        "V_big_coated (V)": abs(V_big_coated_chi),
                        "V_array_coated_common (V)": abs(V_array_coated_common_chi),
                        "V_each_coated (V)": abs(V_each_coated_chi),
                        "phi_array_coated_separate (V)": abs(phi_array_coated_separate_chi)
                    })

df = pd.DataFrame(rows)

print("Quick summary (magnitudes).")
for d_pmV in d_pmV_list:
    print(f"\n--- d = {d_pmV} pm/V ---")
    sub = df[df["d (pm/V)"] == d_pmV]
    print(sub.head(12).to_string(index=False))