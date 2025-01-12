#from __future__ import division
#from builtins import input
#from builtins import range
from past.utils import old_div
import numpy as np
from matplotlib import pyplot as plt

def igm_transmission(z_source, wavelength_obs):
  import numpy as np

  red_wavelength = np.zeros(5)

  # constants from Madau et al 1996
  # Ly alpha..delta coefficients and wavelength
  A = np.array([0.0036, 1.7e-3, 1.2e-3, 9.3e-4])
  wavelength = np.array([1216., 1026., 973., 950., 912.])


  xc = old_div(wavelength_obs, wavelength[4])
  xem = 1.+z_source

  red_wavelength = xem * wavelength

  # We consider that between 912A and 950A, there is only Ly_delta
  # absorption, because of lacking coefficients in Madau et al 1996.

  igm_absorption = np.zeros(len(wavelength_obs))

  # Ly alpha, beta, gamma, delta
  for il in range(4):
    cond = (wavelength_obs <= red_wavelength[il]) & \
           (wavelength_obs > red_wavelength[il+1])
    for jl in range(il+1):
      igm_absorption[cond] += A[jl] * (old_div(wavelength_obs[cond], wavelength[jl]))**3.46

  # Add the photoelectric effect (shortwards 912A)
  cond = (wavelength_obs <= red_wavelength[4])
  for jl in range(4):
    igm_absorption[cond] += A[jl] * (old_div(wavelength_obs[cond], wavelength[jl]))**3.46
  igm_absorption[cond] += \
      0.25  * xc[cond]**3 * (xem**0.46 - xc[cond]**0.46) \
      + 9.4   * xc[cond]**1.5 * (xem**0.18 - xc[cond]**0.18) \
      - 0.7   * xc[cond]**3 * (xc[cond]**(-1.32) - xem**(-1.32)) \
      - 0.023 * (xem**1.68 - xc[cond]**1.68)

  igm_transmission = np.exp(-igm_absorption)

  return igm_transmission

if __name__ == '__main__':
  x = np.arange(500,5000)
  t = igm_transmission(0., x)
  plt.ion()
  plt.plot(x,t)
  a = eval(input())

