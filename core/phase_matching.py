from core.refraction_bbo import *


def match_angle(l, source='tamosauskas', grd=int(1e6)):
    """
    :param l: pump's wavelength.
    :param grd: grid.
    :param source: source.
    :return:
    """
    angles = np.linspace(0, np.pi, grd)
    n_diff = np.zeros(grd)
    for i in range(grd):
        n_diff[i] = abs(refr_eff(l, angles[i], source=source) - refr_no(l * 2, source=source))
    return angles[np.argmin(n_diff)]
