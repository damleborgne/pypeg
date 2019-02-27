from __future__ import print_function
from __future__ import absolute_import
from builtins import input
import numpy as np
import matplotlib.pyplot as plt
import pypeg as pp
import populations as pop
import astropy as ap
import os
from astropy import units as units
import scipy


def example():

  #np.seterr(all='raise')
  #scipy.seterr(all='raise')
  # read filter from ascii transmission
  prefix_filters = os.environ['ZPEG_ROOT']+'/data/filters/'
  fn = prefix_filters + 'u_prime.fil'
  #read and calibrate
  myfilter = pp.Filter(filename = fn)

  # read P2 model
  prefix_templates = os.environ['ZPEG_ROOT']+'/data/templates/'
  fn = prefix_templates + 'Salp_200ages/Sa.dat'
  model = pp.Model()
  model.read_from_p2file(fn, sigma = 10.)
  model.upscale(1e12)

  plt.ion()
  plt.subplot(3,3,1)
  ages = model.props.time
  iage = np.abs(ages - 10000).argmin()
  print(ages[iage])
  plt.plot(model.seds.w, model.seds.fevol[iage,:], label = '{0}'.format(ages[iage]))
  #plt.plot(model.seds.w, model.seds.sed_at_age(10022.), label = '10022')
  plt.plot(model.seds.w, model.seds.sed_at_age(10000.).f, label = '10000')
  plt.plot(model.seds.w, model.seds.sed_at_age(11000.).f, label = '11000')
  plt.plot(model.seds.w, model.seds.sed_at_age(10500.).f, label = '10500')
  plt.legend(loc=0,prop={'size':6})
  
  plt.xlabel("lambda (A)")
  plt.ylabel("Flambda (erg/s/A)")
  plt.xscale('log')
  plt.yscale('log')

  plt.subplot(3,3,2)
  model.plotevol_sed_rf(ages=[40.,400.,1000.,10000.],prop={'size':6})

  #plot absmag(time)
  mag, z = model.seds.absmags(myfilter)
  plt.subplot(3,3,3)
  plt.plot(model.seds.time, mag, label = 'absolute mag')
  plt.xlabel('time (Myr)')
  plt.ylabel('magnitude')

  # plot color(z)
  mag_u, z = model.seds.obsmags(pp.Filter(filename = 'u_prime.fil'), zfor = 10, calibtype='AB')
  mag_i, z = model.seds.obsmags(pp.Filter(filename = 'i_prime.fil'), zfor = 10, calibtype='AB')
  mag_V, z = model.seds.obsmags(pp.Filter(filename = 'V_B90.fil'), zfor = 10, calibtype='AB')
  mag_Z, z = model.seds.obsmags(pp.Filter(filename = 'z_Mosaic.fil'), zfor = 10, calibtype='AB')


  plt.subplot(3,3,4)
  plt.plot(z, mag_V-mag_Z, label = 'observed V-Z')
  plt.xlabel('redshfit')
  plt.ylabel('observed V-Z AB colour')


  plt.subplot(3,3,5)
  plt.plot(z, mag_i, label = 'observed i')
  plt.plot(z, mag_u, label = 'observed u')
  plt.xlabel('redshfit')
  plt.ylabel('observed i AB colour')
  plt.legend(loc=0, prop={'size':6})


  # populations
  p = pop.Mfunction(np.arange(5.,15.,0.1))
  p.Schechter(10.,1e-3,-1.2)
  plt.subplot(3,3,6)
  plt.plot(p.logM, p.N)
  plt.xlabel('log M')
  plt.ylabel(u'N/dex/Mpc$^3$')
  plt.yscale('log')
  plt.ylim(1e-6,1)


  a = input()

if __name__ == '__main__':
  example()
