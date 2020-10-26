import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numba import jit

from core.refraction_bbo import refr_eff, refr_no, refr_ne
from core.refraction_air import refr as refr_air
from core.phase_matching import match_angle
import utils.phys_constants as phconst


# wavelength 354.7nm
pump_wavelen = 0.3547 * 1e-4

omega_p = 2 * np.pi * phconst.c / pump_wavelen
omega_si = omega_p / 2

L1 = 0.05
L2 = L1
L = L1
d = 2.4
phase = 0.0 * np.pi

# FWHM [sm] of pump envelope. 70 micrometers == 70 * 1e-4 sm.
fwhm_pump = 70 * 1e-4
pump_sigma = fwhm_pump / (2 * np.sqrt(np.log(2)))
pump_width = pump_sigma


# Angle between optical axis and extraord. beam(pump) wave vector (optical axis angle) [rad]
alpha = match_angle(pump_wavelen, source='eimerl', grd=int(1e6))

n_si = refr_no(pump_wavelen * 2, source='eimerl')
n_p = refr_eff(pump_wavelen, alpha, source='eimerl')

print(f'ns/np - 1: {n_si/n_p - 1}')

# Walk-off angle.
no = refr_no(pump_wavelen * 2, source='eimerl')
ne = refr_ne(pump_wavelen * 2, source='eimerl')

if ne < no:
    theta = np.arctan((no / ne)**2 * np.tan(alpha)) - alpha
else:
    theta = -np.arctan((no / ne) ** 2 * np.tan(alpha)) + alpha

# pumps wave vector
k_p = n_p * omega_p / phconst.c
k_s = n_si * omega_si / phconst.c
k_i = k_s

if k_i == k_s:
    k_si = k_s

dk_cryst = k_p - k_s - k_i
print('delta k:', dk_cryst)

n_p_air = refr_air(pump_wavelen)
n_si_air = refr_air(pump_wavelen * 2)

k_p_air = n_p_air * omega_p / phconst.c
k_s_air = n_si_air * omega_si / phconst.c
k_i_air = k_s_air

dk_air_cr = k_p_air - k_s_air - k_i_air


# 1 crystal, without gap or anisotropy.
@jit(nopython=True, parallel=True)
def ampl_1cr(qs, qi):
    dk = k_p - np.sqrt(k_s ** 2 - qs ** 2) - np.sqrt(k_i ** 2 - qi ** 2)
    return np.exp(- 0.5 * pump_sigma**2 * (qs + qi)**2) * np.sinc((1/np.pi) * 0.5 * L1 * dk)


# 1 crystal with anisotropy.
@jit(nopython=True, parallel=True)
def ampl_1cr_anis(qs, qi):
    dk1 = (np.sqrt(k_p**2 - (qs + qi)**2) - np.sqrt(k_s**2 - qs**2) - np.sqrt(k_i**2 - qi**2)) * np.sin(theta) + (qs + qi) * np.cos(theta)
    dk2 = (np.sqrt(k_p**2 - (qs + qi)**2) - np.sqrt(k_s**2 - qs**2) - np.sqrt(k_i**2 - qi**2)) * np.cos(theta) - (qs + qi) * np.sin(theta)
    return np.exp(-0.5 * pump_sigma**2 * dk1**2) * np.sinc((1/np.pi) * (L1 / np.cos(theta)) * dk2 * 0.5)


# 2 crystals, no distance with constant phase. No anisotropy.
@jit(nopython=True, parallel=True)
def ampl_2cr_phase(qs, qi):
    dk = k_p - np.sqrt(k_s ** 2 - qs ** 2) - np.sqrt(k_i ** 2 - qi ** 2)
    f1 = np.exp(- 0.5 * pump_sigma ** 2 * (qs + qi) ** 2) * np.sinc((1 / np.pi) * L1 * 0.5 * dk) * np.exp(1j * 0.5 * L1 * dk)
    f2 = np.exp(- 0.5 * pump_sigma ** 2 * (qs + qi) ** 2) * np.sinc((1 / np.pi) * L2 * 0.5 * dk) * np.exp(1j * (3/2) * L2 * dk)
    return f1 + np.exp(1j * phase) * f2


# Two crystals with anisotropy and gap.
@jit(nopython=True, parallel=True)
def ampl_1cr_anis_2cr_gap(qs, qi):
    qs_air = k_s_air * (n_si / n_si_air) * (qs / k_s)
    qi_air = k_i_air * (n_si / n_si_air) * (qi / k_i)

    dk_air = k_p_air - np.sqrt(k_s_air ** 2 - (qs_air) ** 2) - np.sqrt(k_i_air ** 2 - (qi_air) ** 2)

    dk1 = (np.sqrt(k_p**2 - (qs + qi)**2) - np.sqrt(k_s**2 - qs**2) - np.sqrt(k_i**2 - qi**2)) * np.sin(theta) + (qs + qi) * np.cos(theta)
    dk2 = (np.sqrt(k_p**2 - (qs + qi)**2) - np.sqrt(k_s**2 - qs**2) - np.sqrt(k_i**2 - qi**2)) * np.cos(theta) - (qs + qi) * np.sin(theta)

    f1 = np.exp(-0.5 * pump_sigma**2 * dk1**2) * np.sinc((1/np.pi) * (L1 / np.cos(theta)) * dk2 * 0.5) * np.exp(1j * 0.5 * (L1 / np.cos(theta)) * dk2)

    a = - np.sin(theta)
    b = np.cos(theta)
    a1 = np.cos(theta)
    b1 = np.sin(theta)

    dk_par = np.sqrt(k_p**2 - (qs + qi)**2) - np.sqrt(k_s**2 - qs**2) - np.sqrt(k_i**2 - qi**2)
    dk_perp = qs + qi

    f2 = np.exp(-0.5 * pump_sigma**2 * (a*dk_par - a1*dk_perp)**2) * np.sinc((1/np.pi) * (L1 / np.cos(theta)) * 0.5 * (b * dk_par - b1 * dk_perp)) * np.exp(1j * (3/2) * (L1 / np.cos(theta)) * (b * dk_par - b1 * dk_perp))

    air_phase = np.exp(1j * dk_air * d)
    return f1 + air_phase * f2


def sig_intensity(f, x):
    return np.trapz(np.power(np.abs(f), 2), x)


q_max = 14000
q_min = - q_max
q_grd = 1400
q_arr = np.linspace(q_min, q_max, q_grd)

# external angle
ext_angle_arr = np.arcsin((n_si / n_si_air) * (q_arr / k_s))
int_angle_arr = np.arcsin(q_arr / k_s)

x, y = np.meshgrid(q_arr, q_arr)
f = ampl_1cr_anis(x, y)

plt.plot(ext_angle_arr, sig_intensity(f, q_arr) / np.max(sig_intensity(f, q_arr)), label='f2')
plt.xlim([-0.08, 0.08])
plt.title(f'L={L:.2f} sm, d={d:.2f} sm, phase={phase / np.pi:.3f}pi.')
plt.ylim(bottom=0)
plt.grid(True)
plt.xlabel('External angle')
plt.show()
