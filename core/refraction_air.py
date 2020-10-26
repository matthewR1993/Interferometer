
def refr(l, source='ciddor'):
    """
    Refraction index of the air.
    :param l: wavelength in [sm]
    :param source: source
    :return: n - refraction index.
    """
    if source == 'ciddor':
        """
        Source: Ciddor 1996: n 0.23-1.690 µm
        https://refractiveindex.info/?shelf=other&book=air&page=Ciddor
        """
        l = l * 1e4  # convert into micrometers.
        if l < 0.23 or l > 1.69:
            raise ValueError('The wavelength should be between 0.23 – 1.69 micrometers.')
        n = 1 + 0.05792105 / (238.0185 - l ** (-2)) + 0.00167917 / (57.362 - l ** (-2))
    else:
        raise ValueError
    return n
