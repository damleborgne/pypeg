#from __future__ import print_function
#from __future__ import absolute_import
#from builtins import input
import numpy as np
import matplotlib.pyplot as plt
import pypegm as pp
import populations as pop
import astropy as ap
import os
from astropy import units as units
import scipy

def spec_flambda_to_ab(w, flambda):
    # flambda is in erg/s/cm^2/A
    # w is in A
    # convert to erg/s/cm^2/Hz
    c = 2.99792458e18 # A/s
    fnu = flambda * w**2 / c # erg/s/cm^2/Hz
    fab = -2.5*np.log10(fnu) - 48.6
    return fab


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
  mag_u, z, dm = model.seds.obsmags(pp.Filter(filename = 'u_prime.fil'), zfor = 10, calibtype='AB')
  mag_i, z, dm = model.seds.obsmags(pp.Filter(filename = 'i_prime.fil'), zfor = 10, calibtype='AB')
  mag_V, z, dm = model.seds.obsmags(pp.Filter(filename = 'V_B90.fil'), zfor = 10, calibtype='AB')
  mag_Z, z, dm = model.seds.obsmags(pp.Filter(filename = 'z_Mosaic.fil'), zfor = 10, calibtype='AB')


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

  plt.savefig('example.pdf')


  #------------------------------------
  # plot templates at 13 Gyr in AB mag
  plt.figure(figsize=(12,8))
  plt.subplot(1,1,1)

  prefix_templates = os.environ['ZPEG_ROOT']+'/data/templates/'
  prefix_filters = os.environ['ZPEG_ROOT']+'/data/filters/'
  filters = ['1500.fil', '2500.fil', 'u_prime.fil', 'g_prime.fil', 'r_prime.fil', 'i_prime.fil', 'z_prime.fil', 'J.fil', 'H.fil', 'K.fil']

  # choose a color for each template
  colors = {'Sa':'b', 'Sb':'g', 'Sbc':'r', 'Sc':'c', 'Sd':'m'}
  for ht in ['Sa', 'Sb', 'Sbc', 'Sc', 'Sd']:
    fn = prefix_templates + f'Salp_200ages/{ht}.dat'
    model = pp.Model()
    model.read_from_p2file(fn, sigma = 10.)
    model.upscale(1e11)
    time =  model.props.time
    mstars = model.props.mstars
    iage = np.abs(time - 13000.).argmin()

    # spec = model.seds.sed_at_age(13000.)
    spec = model.seds.fevol[iage,:] # erg/s/A
    spec *= 1e10/mstars[iage]
    # dimming with ldist**2
    z = 0.1
    sqdimming = np.sqrt(4.*np.pi)*pp.ldist_z(z) # in cm
    spec /= sqdimming**2

    w = model.seds.w
    ab = spec_flambda_to_ab(w, spec)
    plt.plot(w, ab, color = colors[ht], label = ht)

    # plot filters
    for f in filters:
      myfilter = pp.Filter(filename = prefix_filters + f)
      myfilter.calibrate()
      mag = myfilter.mag(pp.Spectrum(w=w, f=spec), calibtype='AB')
      plt.plot(myfilter.waveeff, mag, 'o', color = colors[ht])
      # draw the same but with an empty black circle to make it more visible
      plt.plot(myfilter.waveeff, mag, 'o', color = 'k', fillstyle='none')
      

  plt.xscale('log')
  plt.gca().invert_yaxis()
  plt.xlim(800, 50000)
  plt.xlabel('Wavelength (A)')
  plt.ylabel('AB mag')
  plt.title(f'Templates at 13 Gyr for 1e10 Msun in stars at z={z}')
  # make a grid with interval = 1 on y axis
  plt.gca().yaxis.set_major_locator(plt.MultipleLocator(1))
  plt.grid()
  
  plt.legend(loc=0, prop={'size':8})
  plt.savefig('example2.pdf')



  
  a = input()


if __name__ == '__main__':
  example()
