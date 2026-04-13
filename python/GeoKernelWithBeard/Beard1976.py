"""
Beard, K. V. (1976). Terminal velocity and shape of cloud and precipitation drops aloft. Journal of Atmospheric Sciences, 33(5), 851-864. DOI: [10.1175/1520-0469(1976)033%3C0851:TVASOC%3E2.0.CO%3B2](https://doi.org/10.1175/1520-0469(1976)033<0851:TVASOC>2.0.CO;2)
"""

import math
import matplotlib.pyplot as plt
from pathlib import Path
import csv

def beard1976_terminal_velocity_m_s(diameter_m, rho_air_kg_m3, temperature_K, pressure_Pa):
    d_cm = min(diameter_m * 100.0, 0.7)   
    rho_air = rho_air_kg_m3 * 1.0e-3      # kg/m^3 -> g/cm^3
    p_hpa = pressure_Pa * 1.0e-2          # Pa -> hPa

    tdeg = temperature_K - 273.15
    if tdeg >= 0.0:
        mu = (1.7180 + 4.9e-3 * tdeg) * 1.0e-4
    else:
        mu = (1.7180 + 4.9e-3 * tdeg - 1.2e-5 * tdeg * tdeg) * 1.0e-4
    # mu: dynamic viscosity [g/(cm s)]

    gxdrow = 9.80665e2 * (1.0 - rho_air)  # grav * (rho_water - rho_air), [cm/s^2 * g/cm^3]
    mean_free_path = 6.62e-6 * (mu / 1.818e-4) * (1013.25 / p_hpa) * math.sqrt(temperature_K / 293.15)

    if d_cm <= 1.9e-3:
        slip = 1.0 + 2.510 * (mean_free_path / d_cm)
        return (gxdrow / (18.0 * mu)) * slip * d_cm * d_cm * 1.0e-2

    if d_cm <= 1.07e-1:
        davies = rho_air * (4.0 * gxdrow) / (3.0 * mu * mu) * d_cm**3
        x1 = math.log(davies)
        x2 = x1 * x1
        x3 = x1 * x2
        y = (
            -3.18657
            + 0.9926960 * x1
            - 0.00153193 * x2
            - 0.000987059 * x3
            - 0.000578878 * x2 * x2
            + 0.0000855176 * x3 * x2
            - 0.00000327815 * x3 * x3
        )
        reynolds = (1.0 + 2.510 * (mean_free_path / d_cm)) * math.exp(y)
        return mu * reynolds / (rho_air * d_cm) * 1.0e-2

    tau = 1.0 - temperature_K / 647.096
    sigma = 0.2358 * math.exp(1.2560 * math.log(tau)) * (1.0 - 0.6250 * tau)
    if temperature_K < 267.5:
        sigma = sigma - 2.854e-3 * math.tanh((temperature_K - 243.9) / 35.35) + 1.666e-3
    sigma *= 1.0e3  # N/m -> g/s^2

    bond = (4.0 * gxdrow) / (3.0 * sigma) * d_cm * d_cm
    npp = ((rho_air * rho_air) * (sigma**3) / (gxdrow * mu**4)) ** (1.0 / 6.0)

    x1 = math.log(bond * npp)
    x2 = x1 * x1
    x3 = x1 * x2
    y = (
        -5.00015
        + 5.23778 * x1
        - 2.04914 * x2
        + 0.47529400 * x3
        - 0.0542819 * x2 * x2
        + 0.00238449 * x3 * x2
    )
    reynolds = npp * math.exp(y)
    return mu * reynolds / (rho_air * d_cm) * 1.0e-2

def main():
    rho_air_kg_m3 = 1.225
    temperature_K = 288.15
    pressure_Pa = 101325

    diameters_um = [i for i in range(5, 601)]
    terminal_velocities_m_s = []

    for d_um in diameters_um:
        d_m = d_um * 1e-6
        v = beard1976_terminal_velocity_m_s(d_m, rho_air_kg_m3, temperature_K, pressure_Pa)
        terminal_velocities_m_s.append(v)

    plt.figure(figsize=(10, 6))
    plt.plot(diameters_um, terminal_velocities_m_s, marker='o', linestyle='-', color='blue')
    plt.xlabel("Diameter (μm)")
    plt.ylabel("Terminal Velocity (m/s)")
    plt.title("Beard (1976) Terminal Velocity")
    plt.grid(True)
    output_path = Path(__file__).with_name("beard1976_terminal_velocity.png")
    plt.savefig(output_path, dpi=300)
    # plt.show()
    
    # save as CSV
    csv_output_path = Path(__file__).with_name("beard1976_terminal_velocity.csv")
    with open(csv_output_path, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Diameter_um", "TerminalVelocity_m_s"])
        for d_um, v in zip(diameters_um, terminal_velocities_m_s):
            writer.writerow([d_um, v])
    
if __name__ == "__main__":
    main()