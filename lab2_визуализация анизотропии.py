# -*- coding: utf-8 -*-
import sys
import subprocess

# Автоматически устанавливаем нужные библиотеки.
try:
    import numpy as np
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    print("Устанавливаю numpy и matplotlib...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy", "matplotlib"])
    import numpy as np
    import matplotlib.pyplot as plt

from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# ============================================================
# ЛАБОРАТОРНАЯ РАБОТА
# Визуализация плотности энергии магнитной анизотропии
# однодоменной ферромагнитной наночастицы
# ============================================================

OUT = Path(__file__).with_name("lab2_results")
OUT.mkdir(exist_ok=True)

# Сетка сферических углов
N_THETA = 80
N_PHI = 160
theta = np.linspace(0.0, np.pi, N_THETA)
phi = np.linspace(0.0, 2.0 * np.pi, N_PHI, endpoint=False)
TH, PH = np.meshgrid(theta, phi, indexing="ij")

AX = np.sin(TH) * np.cos(PH)
AY = np.sin(TH) * np.sin(PH)
AZ = np.cos(TH)


def unit_vector(theta0, phi0):
    return np.array([
        np.sin(theta0) * np.cos(phi0),
        np.sin(theta0) * np.sin(phi0),
        np.cos(theta0),
    ], dtype=float)


def energy_uniaxial(sign=1.0):
    """
    w = sign * (alpha_x^2 + alpha_y^2) = sign * sin^2(theta)
    sign > 0 -> легкая ось z (2 минимума)
    sign < 0 -> легкая плоскость (theta = pi/2)
    """
    return sign * (AX**2 + AY**2)


def energy_cubic(sign=1.0):
    """
    w = sign * (ax^2 ay^2 + ay^2 az^2 + ax^2 az^2)
    sign > 0 -> 6 минимумов вдоль ±x, ±y, ±z
    sign < 0 -> 8 минимумов вдоль диагоналей куба
    """
    return sign * (
        AX**2 * AY**2
        + AY**2 * AZ**2
        + AX**2 * AZ**2
    )


def energy_shape(k_ratio, n):
    """Анизотропия формы: k_ratio * (1 - (alpha·n)^2)."""
    dot = AX * n[0] + AY * n[1] + AZ * n[2]
    return k_ratio * (1.0 - dot**2)


def energy_combined(k_ratio, n, cubic_sign=1.0):
    return energy_cubic(cubic_sign) + energy_shape(k_ratio, n)


def energy_with_field(w_aniso, h, h_direction):
    """
    Полная приведенная энергия:
    w_i = w_eff - 2 h (alpha · h_hat)
    """
    dot = (
        AX * h_direction[0]
        + AY * h_direction[1]
        + AZ * h_direction[2]
    )
    return w_aniso - 2.0 * h * dot


def surface_xyz(w):
    """
    В методичке r(theta,phi) = w(theta,phi).
    Чтобы радиус всегда был положительным, добавляем постоянную.
    """
    r = w - np.min(w) + 0.15
    x = r * AX
    y = r * AY
    z = r * AZ
    return x, y, z


def plot_energy_surface(ax, w, title):
    x, y, z = surface_xyz(w)
    ax.plot_surface(
        x, y, z,
        rstride=2,
        cstride=3,
        linewidth=0,
        antialiased=True
    )
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_box_aspect((1, 1, 1))


def save_four_surfaces():
    fig = plt.figure(figsize=(13, 10))

    cases = [
        (energy_uniaxial(+1), "Одноосная: K1 > 0"),
        (energy_uniaxial(-1), "Одноосная: K1 < 0"),
        (energy_cubic(+1), "Кубическая: K1c > 0"),
        (energy_cubic(-1), "Кубическая: K1c < 0"),
    ]

    for i, (w, title) in enumerate(cases, start=1):
        ax = fig.add_subplot(2, 2, i, projection="3d")
        plot_energy_surface(ax, w, title)

    fig.suptitle("Четыре типа магнитной анизотропии")
    fig.tight_layout()
    fig.savefig(OUT / "01_four_anisotropy_surfaces.png", dpi=160)


def save_combined_aligned():
    """
    Ось формы совпадает с осью z кубической анизотропии.
    Kef/K1c = 0.1, 0.5, 1.0, 2.0
    """
    n = unit_vector(0.0, 0.0)
    ratios = [0.1, 0.5, 1.0, 2.0]

    fig = plt.figure(figsize=(13, 10))

    for i, ratio in enumerate(ratios, start=1):
        w = energy_combined(ratio, n, cubic_sign=1.0)
        ax = fig.add_subplot(2, 2, i, projection="3d")
        plot_energy_surface(ax, w, f"Kef/K1c = {ratio}")

    fig.suptitle("Комбинированная анизотропия: ось формы совпадает с z")
    fig.tight_layout()
    fig.savefig(OUT / "02_combined_aligned.png", dpi=160)


def save_combined_arbitrary():
    """
    Ось формы ориентирована произвольно:
    theta_n = pi/4, phi_n = pi/4.
    """
    n = unit_vector(np.pi / 4.0, np.pi / 4.0)
    ratios = [0.1, 0.5, 1.0, 2.0]

    fig = plt.figure(figsize=(13, 10))

    for i, ratio in enumerate(ratios, start=1):
        w = energy_combined(ratio, n, cubic_sign=1.0)
        ax = fig.add_subplot(2, 2, i, projection="3d")
        plot_energy_surface(ax, w, f"Kef/K1c = {ratio}")

    fig.suptitle("Комбинированная анизотропия: θn = π/4, φn = π/4")
    fig.tight_layout()
    fig.savefig(OUT / "03_combined_arbitrary.png", dpi=160)


def save_field_series():
    """
    Поле направлено перпендикулярно легкой оси z:
    theta_H = pi/2, phi_H = 0 -> вдоль x.
    """
    h_dir = unit_vector(np.pi / 2.0, 0.0)
    h_values = [0.0, 0.3, 0.7, 4.0]

    base_cases = [
        ("Одноосная K1>0", energy_uniaxial(+1)),
        ("Кубическая K1c>0", energy_cubic(+1)),
        ("Кубическая K1c<0", energy_cubic(-1)),
        (
            "Комбинированная",
            energy_combined(
                0.1,
                unit_vector(np.pi / 4.0, np.pi / 4.0),
                cubic_sign=-1.0
            )
        ),
    ]

    for case_index, (case_name, w0) in enumerate(base_cases, start=1):
        fig = plt.figure(figsize=(13, 10))

        for i, h in enumerate(h_values, start=1):
            w = energy_with_field(w0, h, h_dir)
            ax = fig.add_subplot(2, 2, i, projection="3d")
            plot_energy_surface(ax, w, f"h = {h}")

        fig.suptitle(f"Влияние внешнего поля: {case_name}")
        fig.tight_layout()
        fig.savefig(
            OUT / f"04_{case_index:02d}_field_{case_name.replace(' ', '_').replace('>', 'plus').replace('<', 'minus')}.png",
            dpi=160
        )


def candidate_minima(w):
    """
    Поиск локальных минимумов на сферической сетке.
    Затем одинаковые направления, попавшие в несколько соседних узлов,
    объединяются в один минимум.
    """
    # Вдоль phi используем периодические соседи.
    is_min = np.ones_like(w, dtype=bool)

    # theta-соседи
    up = np.vstack([w[0:1, :], w[:-1, :]])
    down = np.vstack([w[1:, :], w[-1:, :]])
    is_min &= w <= up
    is_min &= w <= down

    # phi-соседи и диагонали
    for dtheta in (-1, 0, 1):
        shifted_theta = np.roll(w, dtheta, axis=0)
        if dtheta == -1:
            shifted_theta[-1, :] = w[-1, :]
        elif dtheta == 1:
            shifted_theta[0, :] = w[0, :]

        for dphi in (-1, 1):
            neigh = np.roll(shifted_theta, dphi, axis=1)
            is_min &= w <= neigh

    # Должен быть хотя бы один строго больший сосед, иначе плоское плато.
    strictly_lower_than_some = np.zeros_like(w, dtype=bool)
    neighbors = [
        up, down,
        np.roll(w, 1, axis=1),
        np.roll(w, -1, axis=1)
    ]
    for neigh in neighbors:
        strictly_lower_than_some |= w < neigh
    is_min &= strictly_lower_than_some

    idx = np.argwhere(is_min)
    if idx.size == 0:
        flat = np.argmin(w)
        idx = np.array([np.unravel_index(flat, w.shape)])

    dirs = []
    vals = []
    for i, j in idx:
        dirs.append(np.array([AX[i, j], AY[i, j], AZ[i, j]]))
        vals.append(w[i, j])

    # Сортируем от более глубоких минимумов.
    order = np.argsort(vals)
    clusters = []
    angular_tol = np.deg2rad(10.0)

    for k in order:
        v = dirs[k]
        duplicate = False
        for c in clusters:
            dot = np.clip(np.dot(v, c), -1.0, 1.0)
            angle = np.arccos(dot)
            if angle < angular_tol:
                duplicate = True
                break
        if not duplicate:
            clusters.append(v)

    return clusters


def save_minima_vs_field():
    """
    Строит зависимость числа локальных минимумов полной энергии от h
    для одноосной и кубической (K1c > 0) анизотропии.
    """
    h_dir = unit_vector(np.pi / 2.0, 0.0)
    h_values = np.linspace(0.0, 2.5, 36)

    uniax_counts = []
    cubic_counts = []

    w_uni = energy_uniaxial(+1)
    w_cub = energy_cubic(+1)

    for h in h_values:
        uniax_counts.append(
            len(candidate_minima(energy_with_field(w_uni, h, h_dir)))
        )
        cubic_counts.append(
            len(candidate_minima(energy_with_field(w_cub, h, h_dir)))
        )

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(h_values, uniax_counts, marker="o", label="Одноосная K1>0")
    ax.plot(h_values, cubic_counts, marker="s", label="Кубическая K1c>0")
    ax.set_xlabel("Безразмерное поле h")
    ax.set_ylabel("Число локальных минимумов")
    ax.set_title("Число минимумов полной энергии от величины поля")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "05_minima_count_vs_field.png", dpi=160)


def print_summary():
    print("=" * 70)
    print("ЛАБОРАТОРНАЯ РАБОТА: ВИЗУАЛИЗАЦИЯ МАГНИТНОЙ АНИЗОТРОПИИ")
    print("=" * 70)
    print("Одноосная K1>0: два минимума вдоль +z и -z.")
    print("Одноосная K1<0: легкая плоскость theta = pi/2.")
    print("Кубическая K1c>0: 6 минимумов вдоль декартовых осей.")
    print("Кубическая K1c<0: 8 минимумов вдоль диагоналей куба.")
    print("В сильном поле энергетический ландшафт определяется полем.")
    print()


def main():
    print_summary()

    print("1/5 Строю четыре основные поверхности...")
    save_four_surfaces()

    print("2/5 Комбинированная анизотропия: совпадающая ось...")
    save_combined_aligned()

    print("3/5 Комбинированная анизотропия: произвольная ось...")
    save_combined_arbitrary()

    print("4/5 Серии поверхностей во внешнем поле...")
    save_field_series()

    print("5/5 График числа минимумов от поля...")
    save_minima_vs_field()

    print()
    print("Готово. Результаты сохранены в:")
    print(OUT.resolve())

    plt.show()


if __name__ == "__main__":
    main()
