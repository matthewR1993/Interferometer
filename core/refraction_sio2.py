import numpy as np


def refr_no(l, source='ghosh'):
    """
    Ordinary refraction index.
    :param l: wavelength in [sm]
    :param source: source
    :return: n - ordinary refraction index.
    """
    if source == 'ghosh':
        """
        Source: Ghosh 1999: α-Quartz, n(o) 0.198-2.05 µm
        https://refractiveindex.info/?shelf=main&book=SiO2&page=Ghosh-o
        """
        l = l * 1e4  # convert into micrometers.
        if l < 0.198 or l > 2.0531:
            raise ValueError('The wavelength should be between 0.198 – 2.0531 micrometers.')
        n2 = 1.28604141 + 1.07044083 * l**2 / (l**2 - 0.0100585997) + 1.10202242 * l**2 / (l**2 - 100)
        n = np.sqrt(n2)
    else:
        raise ValueError

    return n


def refr_ne(l, source='ghosh'):
    """
    Extraordinary refraction index.
    :param l: wavelength in [sm]
    :param source: source
    :return: n - ordinary refraction index.
    """
    if source == 'ghosh':
        """
        Source: Ghosh 1999: α-Quartz, n(e) 0.198-2.05 µm
        https://refractiveindex.info/?shelf=main&book=SiO2&page=Ghosh-e
        """
        l = l * 1e4  # convert into micrometers.
        if l < 0.198 or l > 2.0531:
            raise ValueError('The wavelength should be between 0.198 – 2.0531 micrometers.')
        n2 = 1.28851804 + 1.09509924 * l ** 2 / (l ** 2 - 0.0102101864) + 1.15662475 * l ** 2 / (l ** 2 - 100)
        n = np.sqrt(n2)
    else:
        raise ValueError

    return n
