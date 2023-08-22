import numpy as np
from astropy.io import fits
from matplotlib import pyplot as plt


xhdu = fits.open('../hlsp_muscles_cxo_acis_gj15a_none_v23_component-spec.fits')
xdata = fits.getdata("../hlsp_muscles_cxo_acis_gj15a_none_v23_component-spec.fits")
print(xhdu.info())
print(xhdu[1].header)