import numpy as np
from time import time
from numba import jit
import gc
import argparse

from core.refraction_bbo import refr_eff, refr_no, refr_ne
from core.refraction_air import refr as refr_air
from core.refraction_sio2 import refr_no as rerf_sio2
from core.phase_matching import match_angle
import utils.phys_constants as phconst
from core.shmidt import *


parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dist", help="distance d", type=float, required=True)
parser.add_argument("-L1", "--len1", help="1st crystal length", type=float, required=True)
parser.add_argument("-L2", "--len2", help="2nd crystal length", type=float, required=False)
parser.add_argument("-phase", "--phase", help="phase modulation in [pi * rad]", type=float, required=False)
parser.add_argument("-dl", "--dl", help="SiO2 layer thickness", type=float, required=False)
parser.add_argument("-anis", "--anis", help="Anisotropy case", type=str, required=False)
parser.add_argument("-save_path", "--save_path", help="Path to save", type=float, required=True)
args = parser.parse_args()


# Wavelength 354.7 [nm]
pump_wavelen = 0.3547 * 1e-4

omega_p = 2 * np.pi * phconst.c / pump_wavelen
omega_si = omega_p / 2

# d = 1.0 - means 1 [sm].
d = args.dist
# L = 1.0 - means 1 [sm].
L1 = args.len1

save_folder = args.save_path

if args.dl is not None:
    d_sio2 = args.dl
else:
    d_sio2 = 0.0

if args.len2 is None:
    L2 = L1
else:
    L2 = args.len2

print('L1: {}'.format(L1))
print('L2: {}'.format(L2))
print('d: {}'.format(d))

if args.phase is not None:
    phase_modul = args.phase * np.pi
else:
    phase_modul = 0.0

print('phase modulaton in [pi*rad]: {}'.format(phase_modul / np.pi))

# FWHM [sm] of the pump envelope. 70 micrometers == 70 * 1e-4 sm.
fwhm_pump = 70 * 1e-4
pump_sigma = fwhm_pump / (2 * np.sqrt(np.log(2)))
pump_width = pump_sigma

# Angle between optical axis and extraord. beam(pump) wave vector (optical axis angle)[rad]
alpha = match_angle(pump_wavelen, source='eimerl', grd=1000000)

n_si = refr_no(pump_wavelen * 2, source='eimerl')
n_p = refr_eff(pump_wavelen, alpha, source='eimerl')

print('alpha: {}'.format(alpha))
print('ns/np - 1: {}'.format(n_si/n_p - 1))

no = refr_no(pump_wavelen * 2, source='eimerl')
ne = refr_ne(pump_wavelen * 2, source='eimerl')

# Walk-off angle.
if ne < no:
    theta = np.arctan((no / ne)**2 * np.tan(alpha)) - alpha
else:
    theta = -np.arctan((no / ne) ** 2 * np.tan(alpha)) + alpha

# Pumps wave vector
k_p = n_p * omega_p / phconst.c
k_s = n_si * omega_si / phconst.c
k_i = k_s

if k_i == k_s:
    k_si = k_s

dk_cryst = k_p - k_s - k_i

# Air gap.
n_p_air = refr_air(pump_wavelen)
n_si_air = refr_air(pump_wavelen * 2)

k_p_air = n_p_air * omega_p / phconst.c
k_s_air = n_si_air * omega_si / phconst.c
k_i_air = k_s_air

dk_air_cr = k_p_air - k_s_air - k_i_air

# SiO2 layer.
n_p_sio2 = rerf_sio2(pump_wavelen)
n_si_sio2 = rerf_sio2(pump_wavelen * 2)

k_p_sio2 = n_p_sio2 * omega_p / phconst.c
k_s_sio2 = n_si_sio2 * omega_si / phconst.c
k_i_sio2 = k_s_sio2

dk_sio2_cr = k_p_sio2 - k_s_sio2 - k_i_sio2

# Space grid. 201 is minimum. Set (100*n + 1) like numbers.
l_grid = 201

l_array = np.linspace(0.0, L1 + L2, l_grid)
delta_l = l_array[1] - l_array[0]

if L1 == L2:
    ind = np.arange(len(l_array))
    l_ind1 = ind[:l_grid // 2 + 1]
    l_ind2 = ind[l_grid // 2:]
else:
    n1 = round(L1 / float(delta_l))
    ind = np.arange(len(l_array))
    l_ind1 = ind[:n1 + 1]
    l_ind2 = ind[n1:]


# Wave vector grid.
q_grid = 1000
q_max = 12000
q_min = - q_max

q_list = np.linspace(q_min, q_max, q_grid)
delta_q = q_list[1] - q_list[0]

# Pump power.
gamma_arr = np.linspace(0.0001, 0.15, 2)
gain_grid = len(gamma_arr)

ns_list = []
ni_list = []
n2_list = []
ns_1cr_mean_list = []
ns2_1cr_mean_list = []
n_integr_list = []
n2_integr_list = []

# cross corralation <ai_conj(q)*ai(q)*as_conj(q')*as(q')> - <ai_conj(q)*ai(q)><as_conj(q')*as(q')>
ns_ns_avg_list = []
ns_ni_avg_list = []
ni_ns_avg_list = []
ni_ni_avg_list = []
cov_full_list = []

# Auto correl  <as_conj(q)*as(q)*as_conj(q')*as(q')> - <as_conj(q)*as(q)><as_conj(q')*as(q')>
auto_correl_signal_list = []
auto_correl_idler_list = []

# external angle.
angle_arr = np.arcsin((n_si / n_si_air) * (q_list / k_s))

as_array_vs_gain = []
ai_conj_array_vs_gain = []

# Shmidt decomposition.
mode = '710'
shm_list = []

if args.anis is not None:
    anis = args.anis
else:
    anis = 'None'

if anis == 'compens':
    anis_tag = 'anis_compens'
    theta1 = theta
    theta2 = - theta
elif anis == 'non_compens':
    anis_tag = 'anis_non_compens'
    theta1 = theta
    theta2 = theta
elif anis == 'None':
    anis_tag = 'no_anis'
    theta, theta1, theta2 = 0, 0, 0
else:
    raise ValueError

sp = '{}2cr_{}_L1-{:.2f}sm_L2-{:.2f}sm_d-{:.3f}_phase-{:.6f}_dl-{:.7f}.npz'.format(save_folder, anis_tag, L1, L2, d, phase_modul / np.pi, d_sio2)

print('Walk-off theta:', theta)
print('Walk-off theta1:', theta1)
print('Walk-off theta2:', theta2)
print('Save to: ', sp)


def propagate(as_array, ai_conj_array, l_ind, gamma, crystal):
    if mode == '710':
        if crystal == 1:
            @jit
            def f_plus(k, m, l):
                dk_par = np.sqrt(k_p ** 2 - (q_list[m] + q_list[k]) ** 2) - np.sqrt(k_s ** 2 - q_list[m] ** 2) - np.sqrt(k_i ** 2 - q_list[k] ** 2)
                dk_perp = q_list[m] + q_list[k]

                dk1 = dk_par * np.sin(theta) + dk_perp * np.cos(theta)
                dk2 = dk_par - dk_perp * np.tan(theta)

                return np.exp(-0.5 * pump_width ** 2 * dk1**2) * np.exp(1j * l_array[l] * dk2)

            @jit
            def f_minus(k, m, l):
                dk_par = np.sqrt(k_p ** 2 - (q_list[m] + q_list[k]) ** 2) - np.sqrt(k_s ** 2 - q_list[m] ** 2) - np.sqrt(k_i ** 2 - q_list[k] ** 2)
                dk_perp = q_list[m] + q_list[k]

                dk1 = dk_par * np.sin(theta) + dk_perp * np.cos(theta)
                dk2 = dk_par - dk_perp * np.tan(theta)

                return np.exp(-0.5 * pump_width ** 2 * dk1**2) * np.exp(-1j * l_array[l] * dk2)

        if crystal == 2:
            @jit
            def f_plus(k, m, l):
                dk_par = np.sqrt(k_p ** 2 - (q_list[m] + q_list[k]) ** 2) - np.sqrt(k_s ** 2 - q_list[m] ** 2) - np.sqrt(k_i ** 2 - q_list[k] ** 2)
                dk_perp = q_list[m] + q_list[k]

                dk1 = dk_par * np.sin(theta2) + dk_perp * np.cos(theta2)
                dk2 = dk_par - dk_perp * np.tan(theta2)

                qs_air = k_s_air * (n_si / n_si_air) * (q_list[m] / k_s)
                qi_air = k_i_air * (n_si / n_si_air) * (q_list[k] / k_i)
                dk_air = k_p_air - np.sqrt(k_i_air ** 2 - qi_air ** 2) - np.sqrt(k_s_air ** 2 - qs_air ** 2)

                qs_sio2 = qs_air * (k_s_sio2 / k_s_air) * (n_si_air / n_si_sio2)
                qi_sio2 = qi_air * (k_i_sio2 / k_i_air) * (n_si_air / n_si_sio2)
                dk_sio2 = k_p_sio2 - np.sqrt(k_i_sio2 ** 2 - qi_sio2 ** 2) - np.sqrt(k_s_sio2 ** 2 - qs_sio2 ** 2)

                add_ph = np.exp(1j * L1 * (dk_par - dk_perp * np.tan(theta)))

                return np.exp(-0.5 * pump_width ** 2 * dk1 ** 2) * add_ph * np.exp(1j * l_array[l] * dk2) * np.exp(1j * d * dk_air) * np.exp(1j * d_sio2 * dk_sio2) * np.exp(1j * phase_modul)

            @jit
            def f_minus(k, m, l):
                dk_par = np.sqrt(k_p ** 2 - (q_list[m] + q_list[k]) ** 2) - np.sqrt(k_s ** 2 - q_list[m] ** 2) - np.sqrt(k_i ** 2 - q_list[k] ** 2)
                dk_perp = q_list[m] + q_list[k]

                dk1 = dk_par * np.sin(theta2) + dk_perp * np.cos(theta2)
                dk2 = dk_par - dk_perp * np.tan(theta2)

                qs_air = k_s_air * (n_si / n_si_air) * (q_list[m] / k_s)
                qi_air = k_i_air * (n_si / n_si_air) * (q_list[k] / k_i)
                dk_air = k_p_air - np.sqrt(k_i_air ** 2 - qi_air ** 2) - np.sqrt(k_s_air ** 2 - qs_air ** 2)

                qs_sio2 = qs_air * (k_s_sio2 / k_s_air) * (n_si_air / n_si_sio2)
                qi_sio2 = qi_air * (k_i_sio2 / k_i_air) * (n_si_air / n_si_sio2)
                dk_sio2 = k_p_sio2 - np.sqrt(k_i_sio2 ** 2 - qi_sio2 ** 2) - np.sqrt(k_s_sio2 ** 2 - qs_sio2 ** 2)

                add_ph = np.exp(-1j * L1 * (dk_par - dk_perp * np.tan(theta)))

                return np.exp(-0.5 * pump_width ** 2 * dk1 ** 2) * add_ph * np.exp(-1j * l_array[l] * dk2) * np.exp(-1j * d * dk_air) * np.exp(-1j * d_sio2 * dk_sio2) * np.exp(-1j * phase_modul)

    else:
        raise ValueError

    as_array_curr = as_array
    ai_conj_array_curr = ai_conj_array

    for l in l_ind:
        as_array_next = np.zeros((q_grid, 2 * q_grid), dtype=complex)
        ai_conj_array_next = np.zeros((q_grid, 2 * q_grid), dtype=complex)

        for m in range(q_grid):
            a_plus = np.zeros(2 * q_grid, dtype=complex)
            a_minus = np.zeros(2 * q_grid, dtype=complex)
            for k in range(q_grid):
                a_plus += f_plus(k, m, l) * ai_conj_array_curr[k, :]
                a_minus += f_minus(k, m, l) * as_array_curr[k, :]
            as_array_next[m, :] = as_array_curr[m, :] + gamma * a_plus
            ai_conj_array_next[m, :] = ai_conj_array_curr[m, :] + gamma * a_minus

        # keep only current by L.
        as_array_curr = as_array_next
        ai_conj_array_curr = ai_conj_array_next

    return as_array_next, ai_conj_array_next


t = time()
for i in range(gain_grid):
    print('step:', i)

    gamma = delta_l * delta_q * gamma_arr[i]

    as_array = np.zeros((q_grid, 2 * q_grid), dtype=complex)
    ai_conj_array = np.zeros((q_grid, 2 * q_grid), dtype=complex)

    # Boundary(initially) conditions at z = 0
    for m in range(q_grid):
        as_array[m, m] = 1
        ai_conj_array[m, q_grid + m] = 1

    # Propagate through 1st crystal.
    as_array, ai_conj_array = propagate(as_array, ai_conj_array, l_ind1[:-1], gamma, crystal=1)

    # Intensity <Ns>(q), <Ns^2>(q) after 1st crystal.
    ns_1cr_mean = np.zeros(q_grid)
    ns2_1cr_mean = np.zeros(q_grid)

    for m in range(q_grid):
        # Signal intensity <Ns>(q) after 1st crystal for 700 nm.
        ns_1cr_mean[m] = np.sum(np.power(np.abs(as_array[m, q_grid:]), 2))

        # <Ns^2>(q) element after 1st crystal.
        x = np.sum(np.power(np.abs(as_array[m, q_grid:]), 2))
        ns2_1cr_mean[m] = x * (x + np.sum(np.power(np.abs(as_array[m, :q_grid]), 2)))

    ns_1cr_mean_list.append(ns_1cr_mean)
    ns2_1cr_mean_list.append(ns2_1cr_mean)

    # Propagating through 2nd crystal.
    x = l_ind1[:-1]
    as_array, ai_conj_array = propagate(as_array, ai_conj_array, x, gamma, crystal=2)

    as_array_vs_gain.append(as_array)
    ai_conj_array_vs_gain.append(ai_conj_array)

    n_mean = np.zeros(q_grid)  # signal
    n_mean_idler = np.zeros(q_grid)  # idler

    # <Ns^2>(q) element.
    n2_mean = np.zeros(q_grid)  # signal

    # Signal intensity Ns(q) in the z = L(l=l_grid) for 700 nm
    for m in range(q_grid):
        n_mean[m] = np.sum(np.power(np.abs(as_array[m, q_grid:]), 2))

    ns_list.append(n_mean)

    # <Ns^2>(q) element in the z = L(l=l_grid). Depends on angle (q).
    for m in range(q_grid):
        x = np.sum(np.power(np.abs(as_array[m, q_grid:]), 2))
        n2_mean[m] = x * (x + np.sum(np.power(np.abs(as_array[m, :q_grid]), 2)))

    n2_list.append(n2_mean)

    # # <Ns> integral element.
    n_integr = np.sum(n_mean)
    n_integr_list.append(n_integr)

    # # <Ns^2> integral element.
    n2_integr = np.sum(n2_mean)
    n2_integr_list.append(n2_integr)

    # <Ns(q)Ns(q')> - <Ns(q)><Ns(q')>
    ns_ns_avg = np.zeros((q_grid, q_grid), dtype=complex)

    # <Ns(q)Ni(q')> - <Ns(q)><Ni(q')>
    ns_ni_avg = np.zeros((q_grid, q_grid), dtype=complex)

    # <Ni(q)Ns(q')> - <Ni(q)><Ns(q')>
    ni_ns_avg = np.zeros((q_grid, q_grid), dtype=complex)

    # <Ni(q)Ni(q')> - <Ni(q)><Ni(q')>
    ni_ni_avg = np.zeros((q_grid, q_grid), dtype=complex)

    # Covariance.
    for p in range(q_grid):
        for p_ in range(q_grid):
            ns_ns_avg[p, p_] = np.sum(np.multiply(np.conj(as_array[p, q_grid:]), as_array[p_, q_grid:])) * \
                               np.sum(np.multiply(as_array[p, :q_grid], np.conj(as_array[p_, :q_grid])))

            ns_ni_avg[p, p_] = np.sum(np.multiply(np.conj(as_array[p, q_grid:]), ai_conj_array[p_, q_grid:])) * \
                               np.sum(np.multiply(as_array[p, :q_grid], np.conj(ai_conj_array[p_, :q_grid])))

            ni_ns_avg[p, p_] = np.sum(np.multiply(ai_conj_array[p, :q_grid], np.conj(as_array[p_, :q_grid]))) * \
                               np.sum(np.multiply(np.conj(ai_conj_array[p, q_grid:]), as_array[p_, q_grid:]))

            ni_ni_avg[p, p_] = np.sum(np.multiply(ai_conj_array[p, :q_grid], np.conj(ai_conj_array[p_, :q_grid]))) * \
                               np.sum(np.multiply(np.conj(ai_conj_array[p, q_grid:]), ai_conj_array[p_, q_grid:]))

    ns_ns_avg_list.append(ns_ns_avg)
    ns_ni_avg_list.append(ns_ni_avg)
    ni_ns_avg_list.append(ni_ns_avg)
    ni_ni_avg_list.append(ni_ni_avg)

    cov_full = ns_ns_avg + ns_ni_avg + ni_ns_avg + ni_ni_avg
    cov_full_list.append(cov_full)

    # Shmidt decomposition.
    try:
        modes1, modes2, eingvals, phases = shmidt_decomp(cov_full, modes_num=50)
        shm = {'modes1': modes1, 'modes2': modes2, 'eigvals': eingvals, 'phases': phases}
        shm_list.append(shm)
    except:
        shm_list.append(None)

    gc.collect()
print('elapsed time: {}'.format(time() - t))


np.savez_compressed(sp,
     intensity_list=ns_list,
     n2_list=n2_list,
     n_integr_list=n_integr_list,
     n2_integr_list=n2_integr_list,
     ns_1cr_mean_list=ns_1cr_mean_list,
     ns2_1cr_mean_list=ns2_1cr_mean_list,
     cov_full_list=cov_full_list,
     ns_ns_avg_list=ns_ns_avg_list,
     ns_ni_avg_list=ns_ni_avg_list,
     ni_ns_avg_list=ni_ns_avg_list,
     ni_ni_avg_list=ni_ni_avg_list,
     as_array_vs_gain=as_array_vs_gain,
     ai_conj_array_vs_gain=ai_conj_array_vs_gain,
     gamma_arr=gamma_arr,
     q_list=q_list,
     shm_list=shm_list,
     ext_angle_arr=angle_arr,
     L1=L1,
     L2=L2,
     d=d,
     phase_modulation=phase_modul,
     fwhm_pump=fwhm_pump,
     pump_wavelen=pump_wavelen,
     theta=theta,
     alpha=alpha,
     l_array=l_array)
