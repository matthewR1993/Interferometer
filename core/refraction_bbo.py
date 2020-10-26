import numpy as np


def refr_no(l, source='tamosauskas'):
    """
    Ordinary refraction index.
    :param l: wavelength in [sm]
    :param source: source
    :return: n - ordinary refraction index.
    """

    if source == 'zhang':
        """
        Source: Zhang et al. 2000: n(o) 0.64-3.18 µm
        https://refractiveindex.info/?shelf=main&book=BaB2O4&page=Zhang-o
        """
        l = l * 1e4  # convert into micrometers.
        if l < 0.64 or l > 3.18:
            raise ValueError('The wavelength should be between 0.64 – 3.18 micrometers.')
        n2 = 2.7359 + 0.01878 / (l**2 - 0.01822) - 0.01471 * l**2 + 0.0006081 * l**4 - 0.00006740 * l**6
        n = np.sqrt(n2)
    elif source == 'tamosauskas':
        """
        Source: Tamošauskas et al. 2018: n(o) 0.188-5.2 µm
        https://refractiveindex.info/?shelf=main&book=BaB2O4&page=Tamosauskas-o
        """
        l = l * 1e4  # convert into micrometers.
        if l < 0.188 or l > 5.2:
            raise ValueError('The wavelength should be between 0.188 – 5.2 micrometers.')
        n2 = 1 + 0.90291 * l**2 / (l**2 - 0.003926) + 0.83155 * l**2 / (l**2 - 0.018786) + 0.76536 * l**2 / (l**2 - 60.01)
        n = np.sqrt(n2)
    elif source == 'eimerl':
        """
        Source: Eimerl et al. 1987: n(o) 0.22-1.06 µm
        https://refractiveindex.info/?shelf=main&book=BaB2O4&page=Eimerl-o
        """
        l = l * 1e4  # convert into micrometers.
        if l < 0.22 or l > 1.06:
            raise ValueError('The wavelength should be between 0.22 – 1.06 micrometers.')
        n2 = 2.7405 + 0.0184 / (l**2 - 0.0179) - 0.0155 * l**2
        n = np.sqrt(n2)
    else:
        raise ValueError
    return n


def refr_ne(l, source='tamosauskas'):
    """
    Extraordinary refraction index.
    :param l: wavelength in [sm]
    :param source: source
    :return: n - extraordinary refraction index.
    """
    if source == 'zhang':
        """
        Source: Zhang et al. 2000: n(e) 0.64-3.18 µm
        https://refractiveindex.info/?shelf=main&book=BaB2O4&page=Zhang-e
        """
        l = l * 1e4  # convert into micrometers.
        if l < 0.64 or l > 3.18:
            raise ValueError('The wavelength should be between 0.64 – 3.18 micrometers.')
        n2 = 2.3753 + 0.01224 / (l**2 - 0.01667) - 0.01627 * l**2 + 0.0005716 * l**4 - 0.00006305 * l**6
        n = np.sqrt(n2)
    elif source == 'tamosauskas':
        """
        Source: Tamošauskas et al. 2018: n(e) 0.188-5.2 µm
        https://refractiveindex.info/?shelf=main&book=BaB2O4&page=Tamosauskas-e
        """
        l = l * 1e4  # convert into micrometers.
        if l < 0.188 or l > 5.2:
            raise ValueError('The wavelength should be between 0.188 – 5.2 micrometers.')
        n2 = 1 + 1.151075 * l**2 / (l**2 - 0.007142) + 0.21803 * l**2 / (l**2 - 0.02259) + 0.656 * l**2 / (l**2 - 263)
        n = np.sqrt(n2)
    elif source == 'eimerl':
        """
        Source: Eimerl et al. 1987: n(e) 0.22-1.06 µm
        https://refractiveindex.info/?shelf=main&book=BaB2O4&page=Eimerl-e
        """
        l = l * 1e4  # convert into micrometers.
        if l < 0.22 or l > 1.06:
            raise ValueError('The wavelength should be between 0.22 – 1.06 micrometers.')
        n2 = 2.3730 + 0.0128 / (l ** 2 - 0.0156) - 0.0044 * l ** 2
        n = np.sqrt(n2)
    else:
        raise ValueError
    return n


def refr_eff(l, teta, source='tamosauskas'):
    """
    Effective refractive index.
    :param l: wavelength in [sm]
    :param teta: angle between wave vector and optical axis [rad].
    :param source: source
    :return: n - effective refractive index.
    """
    return 1 / np.sqrt(np.sin(teta)**2 / refr_ne(l, source=source)**2 + np.cos(teta)**2 / refr_no(l, source=source)**2)
