# -*- coding:utf-8-*

# module population from pypeg : mass functions etc.

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from builtins import next
from builtins import input
from builtins import str
from builtins import range
from builtins import object
import functools

import numpy as np
import matplotlib.pyplot as plt
import pypeg.pypegm as pypeg
import os
from astropy.cosmology import FlatLambdaCDM, LambdaCDM

#pypeg = reload(pypeg)

def extrap(x, xp, yp):
  """
  np.interp function with linear extrapolation. Sort xp and remove nan or inf values. X must be a list or ndarray
  """

  iv = np.invert((np.isnan(xp+yp) | np.isinf(xp+yp)))
  isort = np.argsort(xp[iv])

  y = np.interp(x, xp[iv][isort], yp[iv][isort])
  y[x < xp[iv][isort][0]] = yp[iv][isort][0] + (x[x<xp[iv][isort][0]]-xp[iv][isort][0]) * (yp[iv][isort][0]-yp[iv][isort][1]) / (xp[iv][isort][0]-xp[iv][isort][1])
  y[x > xp[iv][isort][-1]]= yp[iv][isort][-1] + (x[x>xp[iv][isort][-1]]-xp[iv][isort][-1])*(yp[iv][isort][-1]-yp[iv][isort][-2])/(xp[iv][isort][-1]-xp[iv][isort][-2])

  return y


def get_area(positions, nbins = 100):

  H, xedges, yedges = np.histogram2d(positions[:,0], positions[:,1], bins = nbins)

  narea = 0
  for i in range(len(xedges)-1):
    for j in range(len(yedges)-1  ):
      n1 = np.sum(H[:i+1, :j+1])
      n2 = np.sum(H[i:, :j+1])
      n3 = np.sum(H[:i+1, j:])
      n4 = np.sum(H[i:, j:])
      if n1*n2*n3*n4 > 0:
        narea += 1

  return narea * (xedges[1]-xedges[0]) * (yedges[1]-yedges[0])


class Photo_sample(object):

  def __init__(self, filters, positions, photometry):
    """ 
      positions : ndarray of shape (ndata, 2) for RA, dec
      photometry : ndarray of shape (ndata, nfilters, 2) for AB, ABerr for each galaxy/filter
    """

    self.filters = [] # elements of the list are instances of pypeg.Filter 
    self.ndata = len(positions)
    self.positions = positions 
    self.photometry = photometry

    self.area = get_area(positions)


def STY(data):
  """ Fits the data (instance of Photo_sample) with the STY method, given incompleteness sampling (weights) and limiting magnitue) """

  return best_LF, cov_matrix



def make_schechter_from_zpegdata(zpegfile, zbins = None, 
  magbins = None, vmaxcorr = True, keeponly = None, magoffsets = None):

  # magoffsets are offsets to be applied to all magnitudes before making the LF.
  # magoffsets is a dictionary: len(magoffsets) = number of Hubble types 

  from pyzpeg import pyzpeg

  zpegrun = pyzpeg.Zpeg_run()
  zpegrun.read_zpegres(zpegfile)
  zpd = zpegrun.data
  zph = zpegrun.header

  if keeponly is not None: # e.g. {'fields': ['ypix', 'DEC'], 'mins': [10, 3.33], 'maxs': [15., 4.3]}
    for k,myfield in enumerate(keeponly['fields']):
      zpd = zpd[ (zpd[myfield] >= keeponly['mins'][k]) &  (zpd[myfield] < keeponly['maxs'][k]) ]

  if zbins is None:
    zbins = np.linspace(0.,zph['zmax'], zph['zmax']/(0.5)+1)
  if magbins is None:
    magbins = np.arange(-25.,-15., 0.5)
  nbands = len(zph['Filters'])

  LFs = np.zeros((nbands,len(magbins)-1, len(zbins)-1))
  LFerrs = np.zeros((nbands,len(magbins)-1, len(zbins)-1))

  # compute Volume ponderation for each galaxy
  Vmaxs = np.zeros(len(zpd))
  zmins = np.zeros(len(zpd))
  zmaxs = np.zeros(len(zpd))

  izinsert = np.searchsorted(zbins, zpd['0_z']) # galaxies would take place at index "izinsert" in new array with z inserted

  izmins = np.copy(izinsert)-1
  izmins[izmins <= 0] = 0   # the galaxy has a lower z than the lowest zbin.... don't bother as it won't be used anyway

  izmaxs = np.copy(izinsert)
  izmaxs[izmaxs >= len(zbins)] = 0   # the galaxy has a higher z than the highest zbin.... don't bother as it won't be used anyway

  zmins = zbins[izmins]
  zmaxs = zbins[izmaxs]
  if vmaxcorr:
    zmaxs = np.min([zpd['zmax'], zmaxs], axis=0) # this is the Vmax correction !
    iabnormal = np.where(zmaxs < zmins)[0]

    if len(iabnormal)>0:
      ibad = iabnormal[0]
      print('zbins = ', zbins)
      print(ibad)
      print('HUHU.....', zpd['zmax'][ibad], zpd['0_z'][ibad], zmins[ibad], zmaxs[ibad])
    else :
      print('Looks good !')

  cosmo = FlatLambdaCDM(H0=zph['h100']*100., Om0=zph['Omega0'])

  Vmaxs = zph['Area of the survey (in sqdeg)'] /  41253. * (cosmo.comoving_volume(zmaxs) - cosmo.comoving_volume(zmins)) # 41253 deg2 = 4pi sr * (180 deg/pi sr)**2



  data_types = np.array(zpd['ypix']).astype(int)
  for iband in range(nbands):
    data_magarr = zpd['0_absmags'][:,iband]
    # deal with type-dependent mag offset to correct for stuff 1.26 bug
    if magoffsets is not None:
      for htype in np.unique(data_types):
        index_istype = np.where(data_types == htype)[0]
        if htype in list(magoffsets.keys()):

          print('TYPEEEEEEEEE : ', htype, magoffsets[htype], 'index=', index_istype)
          data_magarr[index_istype] += magoffsets[htype]
        else:
          print('oups....  we have type',htype)
          print('using means offset....')
          data_magarr[index_istype] += np.mean(list(magoffsets.values()))


    for imag in range(len(magbins)-1):
      magmin = min([magbins[imag], magbins[imag+1]])
      magmax = max([magbins[imag], magbins[imag+1]])
      for iz in range(len(zbins)-1):
        bool_galsok = ((zpd['0_z'] >= zbins[iz]) & (zpd['0_z'] < zbins[iz+1]) & (data_magarr >= magmin) & (data_magarr < magmax))
        LFs[iband, imag, iz] = np.sum(bool_galsok[bool_galsok] * 1./Vmaxs.value[bool_galsok])
        LFerrs[iband, imag, iz] = np.sqrt(np.sum(bool_galsok[bool_galsok] * 1./Vmaxs.value[bool_galsok]**2))

  return LFs, LFerrs, magbins, zbins, zph['Filters']

  if False:
    stuff_file = '../test_lf/bj.list'
    area = (0.2*16394./3600.)**2 # (pixel size * width)**2 square degrees
    LFs, LFerrs, m, z = populations.make_schechter_from_stuffcat(stuff_file, area, zbins = [0.3,0.4], frommabs = True)
    plt.ion()
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    mags = (m[:-1]+m[1:])/2. 
    iz = 0
    #mykcorr = 2.75*((z[iz]+z[iz+1])/2./0.7) #z=0.5 for E, in band B
    mykcorr = 0.
    #ABVega = -0.08
    ABVega = 0.
    #adhoc = -2.5*np.log10(0.5)
    adhoc =  2.5*np.log10(1+0.8)
    ax1.errorbar(mags-mykcorr-ABVega + adhoc, LFs[:,iz], ls='-.', yerr = LFerrs[:,iz], label=' z= '+str(z[iz])+'-'+str(z[iz+1])+' ')
    ax1.set_yscale('log')
    ax1.set_xlim(-15,-25)
    ax1.set_ylim(1e-6,1e-1)
    
    H0 = 70.

    slf = Carassou_LF(mags, (z[iz]+z[iz+1])/2., H0, 
      lfevol = {'LF_PHISTAR': [5e-1], 'LF_MSTAR' : [-20.], 'LF_ALPHA' : [-1.], 
      'LF_PHISTAREVOL' : [1.5], 'LF_MSTAREVOL' : [-1.]})

    ax1.plot(mags, slf[0])
    
    a = input('press')

def get_zmax(z, sed, filters, mlim, filterlim, templates):
  """ Returns the maximum redshift at which the object lying at redshift z with apparent magnitude mapp in filter filterapp 
  would still be detectable given its sed in the filters, and the fitting templates 'templates' 

  Parameters:
    - z = real redshift of the galaxy
    - mlim = limiting magnitude of the survey
    - filterlim = pypeg.Filter object defining filter for mlim
    - sed = sed of the galaxy
    - filters = array of pypeg.Filter objects defining filters of the sed
    - templates = array of pypeg.Spectrum objects defing the SED to be used to fit the object at redshift z and red-shift it to zmax

  Returns:
    - zmax = max redshift where object is detectable
  """


  for it in range(len(templates)):
    for iz in range(len(zbins-1)):
      pass
      #mapp_z = filterlim.mag(templates[it])
  return zmax

def get_stuff_param(stuff_conf, param):

  with open(stuff_conf,'r') as f:
    lines = f.readlines()
    for l in lines:
      ls = l.split()
      if len(ls) > 1:
        if ls[0] == param:
          mylist = ls[1].split(',')
  return mylist

def make_LF_fromstuffmabs(stuff_file, stuff_conf = None, zbins = None, magbins = None):
  """ Assumes stuff file has 14 columns, with columns 11 begin z, 3 being mapp, 13 being mabs """

  if stuff_conf is not None:
    H0 = float(get_stuff_param(stuff_conf,'H_0')[0])
    Om0 = float(get_stuff_param(stuff_conf,'OMEGA_M')[0])
    field_size = float(get_stuff_param(stuff_conf,'FIELD_SIZE')[0])
    pixel_size = float(get_stuff_param(stuff_conf,'PIXEL_SIZE')[0])
    area = (field_size*pixel_size/3600.)**2
  else:
    H0 = 70.
    Om0 = 0.3
    area = 1.

  print('AREA (sq deg)=',area)
  s = np.loadtxt(stuff_file) #posx,posy, z, app_mag, abs_mag, hubble_type
  z = s[:,11]
  mapp = s[:,3]
  mabs = s[:,13]

  # define z and absolute magnitude bins
  if zbins is None:
    zbins = np.linspace(0.,5., 5./(0.5)+1)
  if magbins is None:
    magbins = np.arange(-25.,-15., 0.5)

  # initialize LF
  LFs = np.zeros((len(magbins)-1, len(zbins)-1))
  LFerrs = np.zeros((len(magbins)-1, len(zbins)-1))

  # compute Volume ponderation for each galaxy
  Vs = np.zeros(len(s))
  zmins = np.zeros(len(s))
  zmaxs = np.zeros(len(s))

  izinsert = np.searchsorted(zbins, z) # galaxies would take place at index "izinsert" in new array with z inserted
  # zmin  = lower bound of the redshift bin
  izmins = np.copy(izinsert)-1
  izmins[izmins <= 0] = 0   # the galaxy has a lower z than the lowest zbin.... don't bother as it won't be used anyway
  zmins = np.array(zbins)[izmins]
  # zmax = maximum redshift at which the galaxy would still be detectable given the apparent magnitude selection. Default = end of redshift bin
  izmaxs = np.copy(izinsert)
  izmaxs[izmaxs >= len(zbins)] = 0   # the galaxy has a higher z than the highest zbin.... don't bother as it won't be used anyway
  zmaxs = np.array(zbins)[izmaxs]

  # compute Vs : volume within the z bin 
  cosmo = FlatLambdaCDM(H0=H0, Om0=Om0)
  Vs = area /  41253. * (cosmo.comoving_volume(zmaxs) - cosmo.comoving_volume(zmins)) # 41253 deg2 = 4pi sr * (180 deg/pi sr)**2

  for imag in range(len(magbins)-1):
    magmin = min([magbins[imag], magbins[imag+1]])
    magmax = max([magbins[imag], magbins[imag+1]])
    for iz in range(len(zbins)-1):
      bool_galsok = ((z >= zbins[iz]) & (z < zbins[iz+1]) & (mabs >= magmin) & (mabs < magmax))
      LFs[imag, iz] = np.sum(bool_galsok[bool_galsok] * 1./Vs.value[bool_galsok])
      LFerrs[imag, iz] = np.sqrt(np.sum(bool_galsok[bool_galsok] * 1./Vs.value[bool_galsok]**2))

  return LFs, LFerrs, magbins, zbins


def make_schechter_from_stuffcat_with_mabs(stuff_file, stuff_conf, zbins = None, magbins = None, H0 = 70., Om0 = 0.3):

  from pprint import pprint 
  """ No Vmax correciton is done, and no k-correction either !!!! So it only works at z=0 in principle """

  from pyzpeg import pyzpeg
  from astropy.cosmology import FlatLambdaCDM

  # read stuff catalog
  if frommabs:
    s = np.loadtxt(stuff_file, usecols = ((1, 2, 11, 3, 12, 13))) #posx,posy, z, app_mag, abs_mag, hubble_type
  else:
    s = np.loadtxt(stuff_file, usecols = ((1, 2, 11, 3, 12))) #posx,posy, z, app_mag, hubble_type

  # define z and absolute magnitude bins
  if zbins is None:
    zbins = np.linspace(0.,5., 5./(0.5)+1)
  if magbins is None:
    magbins = np.arange(-25.,-15., 0.5)

  # initialize LF
  LFs = np.zeros((len(magbins)-1, len(zbins)-1))
  LFerrs = np.zeros((len(magbins)-1, len(zbins)-1))

  # compute Volume ponderation for each galaxy
  Vmaxs = np.zeros(len(s))
  zmins = np.zeros(len(s))
  zmaxs = np.zeros(len(s))

  izinsert = np.searchsorted(zbins, s[:,2]) # galaxies would take place at index "izinsert" in new array with z inserted

  # zmin  = lower bound of the redshift bin
  izmins = np.copy(izinsert)-1
  izmins[izmins <= 0] = 0   # the galaxy has a lower z than the lowest zbin.... don't bother as it won't be used anyway
  zmins = np.array(zbins)[izmins]

  # zmax = maximum redshift at which the galaxy would still be detectable given the apparent magnitude selection. Default = end of redshift bin
  izmaxs = np.copy(izinsert)
  izmaxs[izmaxs >= len(zbins)] = 0   # the galaxy has a higher z than the highest zbin.... don't bother as it won't be used anyway
  zmaxs = np.array(zbins)[izmaxs]

  # get real zmax
  ngals = s.shape[0]
  for i in range(ngals):
    real_zmax = get_zmax(s[i,2], )

  # compute Vmax : volume within the z bin where the galaxy would be detectable, given its magnitude
  cosmo = FlatLambdaCDM(H0=H0, Om0=Om0)
  Vmaxs = area /  41253. * (cosmo.comoving_volume(zmaxs) - cosmo.comoving_volume(zmins)) # 41253 deg2 = 4pi sr * (180 deg/pi sr)**2

  #print s[:,3], s[:,2]
  # ABS_MAG = APP_MAG - distance modulous
  if frommabs:
    absmags = s[:,5]
  else:
    absmags = s[:,3] - 5.*(np.log10(cosmo.luminosity_distance(s[:,2]).value)+6-1) # warning : no k-correction here !

  for imag in range(len(magbins)-1):
    magmin = min([magbins[imag], magbins[imag+1]])
    magmax = max([magbins[imag], magbins[imag+1]])
    for iz in range(len(zbins)-1):
      bool_galsok = ((s[:,2] >= zbins[iz]) & (s[:,2] < zbins[iz+1]) & (absmags >= magmin) & (absmags < magmax))
      LFs[imag, iz] = np.sum(bool_galsok[bool_galsok] * 1./Vmaxs.value[bool_galsok])
      LFerrs[imag, iz] = np.sqrt(np.sum(bool_galsok[bool_galsok] * 1./Vmaxs.value[bool_galsok]**2))

  return LFs, LFerrs, magbins, zbins

def Schechter_mag(magarr, *params):
  Log10Phi = params[0]
  Mstar = params[1]
  alpha = params[2]
  return 0.4 * np.log(10.) * 10.**Log10Phi * 10.**(-0.4*(magarr-Mstar)*(alpha+1)) * np.exp(-10.**(-0.4*(magarr-Mstar))) 

class Popfunc(object):
  """ Generic function for galaxy population distrbution : 
      dN / dex(mass or flux) (/ Mpc^3 or /sqdeg)
      or 
      dN / mag (/ Mpc^3 or /sqdeg)
     """

  def __init__(self, *logxarr):
    """
    logxarr is log10(M) or log10(Flambda) or log10(Fnu) or mag. 
    """

    if len(logxarr) ==0 :
      self.logxarr = np.arange(0., 20., 0.01)
    else:
      self.logxarr = logxarr[0]
    #
    #self.logxarr = logxarr
    self.N       = np.zeros(len(self.logxarr)) # function N(logx) in # / dex, independant of logx sampling
    self.type   = 'dex_mass' # can be 'mag' or 'dex_mass' or 'dex_sfr' or...


  def Schechter(self,logxknee = None, Phi = None, alpha = None):
    """ 
    in lum  : N(x)dx       = phi * x^alpha e^{-x} dx with x = L/L*  or M/M* or ...
    in mag  : N(m)dm       = 0.4 ln(10) phi* [10^{-0.4(m-m*)}]^(alpha+1) e^{-10^{-0.4(m-m*)}} dm
    in logx : N(logx)dlogx = ln(10) phi* 10^(DLX(alpha+1)) e^(-10^DLX) dlogx avec DLX = log10(L/L*) 
    """
    if self.type == 'dex_mass': # default
      if logxknee is None: logxknee = 9.
      if Phi is None: Phi = 1e-3
      if alpha is None: alpha = -1.3

    if self.type == 'mag':
      if logxknee is None: logxknee = -18.
      if Phi is None: Phi = 1e-3
      if alpha is None: alpha = -1.3

    self.logxknee = logxknee
    self.Phi      = Phi
    self.alpha    = alpha

    if self.type == 'dex_mass':
      # number per dex =  "Nlog" = N(logx) = #/dex = dN= N(logx)/(dlogx=1dex)
      self.N = np.log(10.) * Phi *\
      10.**((self.logxarr-logxknee)*(alpha+1)) * \
      np.exp(-10.**(self.logxarr-logxknee)) 

    if self.type == 'mag':
      # number per mag = N(m) = #/mag = dN= N(mag)/(dmag=1mag)
      self.N = 0.4 * np.log(10.) * Phi *\
      (10.**(-0.4*(self.logxarr-logxknee)))**(alpha+1) * \
      np.exp(-10.**(-0.4*(self.logxarr-logxknee))) 

  def integrate(self, logxmin = None, logxmax = None, weights = None, precise = False, integrator = 'trapz'):
    """ Returns the integral xtot = of x*N dx between logx=xmin and logx=logxmax"""
    import scipy.integrate

    if logxmin is None: logxmin = np.min(self.logxarr)
    if logxmax is None: logxmax = np.max(self.logxarr)
    if weights is None: weights = np.ones_like(self.logxarr)
    # weights = 10.**self.logxarr if you want the total mass for a mass function
    # weights = np.ones_like(self.N) if you want the number of objects for a mass function

    myintegrator = getattr(scipy.integrate, integrator)

    if not(precise):
      iok = (self.logxarr >= logxmin-1e-10) & (self.logxarr <= logxmax+1e-10)
      res = myintegrator(weights[iok]*self.N[iok], self.logxarr[iok])
    else:
      logxarr = np.sort(np.unique(np.hstack((self.logxarr,[logxmin, logxmax]))))
      #if (logmin not in self.logxarr):
      iok = (logxarr >= logxmin) & (logxarr <= logxmax)
      y = np.interp(logxarr[iok], self.logxarr, weights*self.N)
      res = myintegrator(y,logxarr[iok])
      # or faster with searchsorted and insert ???
    return res

  #def Nperdex_to_Npermag(self):
  #  """ Converts N(log) (#/dex) to N(mag) (#/mag) """
  #  return self.N * 0.4

  #def Npermag_to_Nperdex(self):
  #  """ Converts N(mag) (#/mag) to N(log) (#/dex)"""
  #  return self.N / 0.4
      
  def dN(self):
    """ Number in each logxarr bin """
    #return self.N*self.dlogx
    return self.N*np.gradient(self.logxarr) # second order finite differences; same size as self.N


class Lfunction(Popfunc):
  """ Number of objects / ABmag / Mpc3 in a given filter, given a specific filter calibration (AB, ...)"""

  def __init__(self, logxarr = None, **kwargs):
    if logxarr is None:
      logxarr = np.arange(-25.,-5.,0.1)
    Popfunc.__init__(self, logxarr, **kwargs)
    self.type = 'mag'
    try:
      self.filter = myfilter
    except: # myfilter not defined ?    
      try:
        prefix_filters = os.environ['ZPEG_ROOT']+'/data/filters/'
        fn = prefix_filters + 'u_prime.fil'
        self.filter = pypeg.Filter(filename = fn)
      except:
        print("Error : ZPEG environment variable not defined ?")
        self.filter = pypeg.Filter()


  # fancy stuff to define a marr attribute for class Lfunction which mirrors self.logxarr
  @property
  def marr(self):
    return self.logxarr
  @marr.setter
  def marr(self, value):
    self.logxarr = value

  def to_Mfunction(self, log10M_0, ABmag_0, mf = None):
    """ Convert a lum func to a mass function. Assuming a cosmology. Really basic."""
    if mf is None: #the MF obkect does not exists yet
      mf = Mfunction(self.logxarr)
    
    mf.N = self.N[::-1] * 2.5 # N/mag to N/dex

    # shift the logxarr array
    AB_to_logM = lambda m: log10M_0 - 0.4*(m-ABmag_0)
    mf.logxarr = AB_to_logM(self.logxarr)[::-1]

    return mf

class Mfunction(Popfunc):

  def __init__(self, logxarr = None, **kwargs):
    if logxarr is None:
      logxarr = np.arange(2.,15.,0.1)    
    Popfunc.__init__(self, logxarr, **kwargs)
    self.type   = 'dex_mass'
    self.logM = self.logxarr

  # fancy stuff to define a logM attribute for class Mfunction which mirrors self.logxarr
  @property
  def logM(self):
    return self.logxarr
  @logM.setter
  def logM(self, value):
    self.logxarr = value

  def to_Lfunction(self, log10M_0, ABmag_0, lf = None):
    """ Convert a mass func to a lum function. Assuming a cosmology. Really basic."""

    if lf is None: #the MF obkect does not exists yet
      lf = Lfunction(self.logxarr)
    
    lf.N = self.N[::-1] / 2.5 # N/dex to N/mag

    # shift the logxarr array
    logM_to_AB = lambda log10M: ABmag_0 + 2.5 * (log10M_0 - log10M)
    lf.logxarr = logM_to_AB(self.logxarr)[::-1]

    return lf

# class SFRfunction(Popfunc):
# 
#   def __init__(self, **args):
#     Popfunc.__init__(self, **args)
#     self.type   = 'dex_sfr'
#       
#   def to_SFRfunction(logSFR):
#     """ Multiplies a mass function by SFR (scalar) and return SFR function (#/dex_sfr/Mpc3) """
#     from copy import deepcopy
# 
#     sfr_popfunc = deepcopy(mass_popfunc)
#     sfr_popfunc.type    = 'SFR Function'
#     sfr_popfunc.logxarr += logSFR
#     #sfr_popfunc.logxmin += logSFR
#     #sfr_popfunc.logxmax += logSFR
#     try:
#       sfr_popfunc.logxknee += logSFR
#     except:
#       pass
# 
#     return sfr_popfunc
#       

class Counts(object):
  """
  Galaxy counts : number per square degree per mag or per dex_flux"
  """

  def __init__(self, zarr = None, marr = None, carr = None, **args):
    from astropy import constants as c
    from astropy import units as u

    self.type   = 'Differential Counts per dex S' # per sq degree per Mpc³

    if marr is None: # magnitude bins
      self.marr = np.arange(10. , 40., 0.05)
    else:
      self.marr = marr
    self.dmarr = np.gradient(self.marr)
    if carr is None: # color bins
      self.carr = np.arange(-5.,10.,0.1)
    else:
      self.carr = carr
    self.dcarr = np.gradient(self.carr) # to be checked : use diff or gradient ??

    if zarr is None:
      ## appropriate z scale for dN/dzdm : ~400 values from 0.01 to ~18
      #self.zarr=np.hstack((
      #  np.linspace(0.01,2.,200, endpoint = False),
      #  np.linspace(2.,5., 60, endpoint = False),
      #  np.linspace(5., 18., 130, endpoint = False)
      #  ))
      npoints = 200 # 200 is good low end
      self.zarr = np.array([20.*((i+1)*1./npoints)**(3.) for i in np.arange(npoints)])
    else:
      self.zarr = zarr

    self.dzarr = np.gradient(self.zarr) # to be checked : use diff or gradient ??

    if pypeg.cosmo_dict is None:
      pypeg.define_cosmo()
    mycos = pypeg.cosmo_dict['cosmo']
    ## comobile volume for a steradian in a dz=1 slice at each zarr
    ez = np.sqrt(mycos.Om0*(1.+self.zarr)**3+mycos.Ok0*(1.+self.zarr)**2+mycos.Ode0) #sqrt(OmegaM*cube(1.0+z)+omegaR*sqr(1.0+z)+olh.omegalambda)
    self.dVc = (c.c / mycos.H0 * (pypeg.ldist_z(self.zarr)*u.cm)**2 / (1.+self.zarr)**2. / ez).to(u.Mpc**3)

  def dndmdz(self, model, mfunction, myfilter, zfor = 10., extrapolate_MF = False, verbose = 1, **kwargs):
    """
    Returns N/mag/sqdeg in bins of redshift, apparent magnitude (for dz=1 and dmag = 1)
    """
    from scipy import interpolate
    import time

    d3N = np.zeros((len(self.zarr),len(self.marr))) # N(z,mobs) / (dz=1) / (dm = 1mag) / 1 sq degree

    #t1 = time.time()

    mmag, zmodel, *_ = model.seds.obsmags(myfilter, zfor = zfor, **kwargs)
    sqdeg_per_sr=(180./np.pi)**2  

    #t2 = time.time() ;  print("t mags:", t2-t1) ; t1 = time.time()

    for iz, z in enumerate(self.zarr):
      mymags = -2.5*(mfunction.logM-np.log10(model.norm)) + extrap(np.array([np.log10(z)]), np.log10(zmodel), mmag) # CHECKED ! interpolate in log(z) because m ~ log(d) ~ log(z) at low z
      inozero = (mfunction.N>0.) #N(logM)/Mpc3/dlogM
      try:
        f = interpolate.interp1d(mymags[inozero], np.log10(mfunction.N[inozero]), bounds_error = True, fill_value = -1000.) # N/Mpc3/dlogm at each appmag(z)
        log10Ninterpolated = f(self.marr)

      except:
        # probably a problem with magnitude interpolation bounds
        if np.max(mymags[inozero]) < np.max(self.marr): # it only matters at the faint end for reasonable Schechter-like functions.

          if extrapolate_MF:
            if verbose > 1:
              print('Bypassing the issue of not having defined MF for faintest observable galaxies by extrapolating MF (linear in log N vs log M) !')
            log10Ninterpolated = extrap(self.marr, mymags[inozero], np.log10(mfunction.N[inozero]))

          else:
            if verbose > 1:
              print('WARNING: dndmdz at z=',z)
              print('mymags= model obs mags at z, for all masses=',np.min(mymags[inozero]), np.max(mymags[inozero]))
              print('to be interp on observed marr range which is=',np.min(self.marr), np.max(self.marr))
              #lost_number = (10.**f(mymags[inozero][0]) * self.dVc[iz] / sqdeg_per_sr / 2.5).value
              #print('Please Try to extend the GSMF low-mass range ! '+
              #    'It seems that at this redshift, some galaxies in the observable magnitude range are not described by the GSMF.'+
              #    ' As it is, masses out of the GSMF are assumed to have 0 galaxies, instead of the closest value of log10_N= ',\
              #    np.log10(mfunction.N[inozero][0]), 'at mmodel=', mymags[inozero][0], 'accounting for ',lost_number,' galaxies /sqdeg/mag in the counts this magnitude and redshift bin')
              #print('This is a faction of 10^(',np.log10(lost_number/computed_number_if_bypass),') of the currently computed counts at this magnitude')
            if verbose :
              print('Warning in pypeg.populations.dndmdz, lack of MF definition. Assuming 0 galaxies outside of definition range !')
            fbypass = interpolate.interp1d(mymags[inozero], np.log10(mfunction.N[inozero]), bounds_error = False, fill_value = -1000.) # N/Mpc3/dlogm at each appmag(z)
            log10Ninterpolated = fbypass(self.marr)             

        else: # The issue is only at the bright end, where the contribution of galaxies is negligible anyway for any reasonable Schechter function...
          #The way to deal with missing bound does not matter. Lets interpolate ?
          if True:
            log10Ninterpolated = extrap(self.marr, mymags[inozero], np.log10(mfunction.N[inozero]))
          else:
            fbypass = interpolate.interp1d(mymags[inozero], np.log10(mfunction.N[inozero]), bounds_error = False, fill_value = -1000.) # N/Mpc3/dlogm at each appmag(z)
            log10Ninterpolated = fbypass(self.marr)


      #t2 = time.time() ;  print("t loop:", t2-t1) ; t1 = time.time()

      d3N[iz,:] = 10.**log10Ninterpolated * self.dVc[iz] / sqdeg_per_sr / 2.5 # 2.5 for N/dex to N/mag
  
    return d3N # N(z,mobs) / (dz=1) / (dm = 1mag) / 1 sq degree

  def counts(self, model, mfunction, myfilter, zrange = None, **kwargs): #zfor = 10 assumed in dndmdz
    """
    Returns N/mag/sq deg
    """
    d3N = self.dndmdz(model, mfunction, myfilter, **kwargs) #N/mag/sqdeg/dz=1
    res = d3N * \
      self.dzarr.reshape(len(self.dzarr),1) #N/mag/sqdeg/zbin

    res[(res == np.inf) | (res == np.nan)] = 0.

    if zrange is None:   
      return np.sum(res, axis=0) #sum over redshifts
    else:
      izok = (self.zarr >= zrange[0]) & (self.zarr < zrange[1])
      return np.sum(res[izok], axis=0) #sum over redshifts


    #return np.sum(d3N * np.vstack(self.dzarr), axis=0)


  def dndmdzdc(self, model, mfunction, myfilter1, myfilter2, zfor = 10., **kwargs):
    """
    Returns N/mag/mag/sq deg
    """
    from scipy import interpolate
    #import time

    d4N = np.zeros((len(self.zarr),len(self.marr), len(self.carr))) # N(z,mobs,color) / (dz=1) / (dm = 1mag) / (dcolor=1mag) / 1 sq degree

    mmag1, zmodel, *_ = model.seds.obsmags(myfilter1, zfor = zfor, **kwargs)
    mmag2, zmodel, *_ = model.seds.obsmags(myfilter2, zfor = zfor, **kwargs)

    #t2 = time.time() ;  print("t mags:", t2-t1) ; t1 = time.time()

    sr_in_sqdeg=(180./np.pi)**2  

    for iz, z in enumerate(self.zarr):
      mymag1 = extrap(np.array([np.log10(z)]), np.log10(zmodel), mmag1)
      mymag2 = extrap(np.array([np.log10(z)]), np.log10(zmodel), mmag2)
      mycol = mymag1 - mymag2
      ic = np.digitize(mycol, self.carr)[0]-1
      mymag1 = -2.5*mfunction.logxarr + mymag1

      mydn = mfunction.N #N(logM)/dlogM
      f = interpolate.interp1d(mymag1, mydn, bounds_error = False, fill_value = 0.)
      d4N[iz,:,ic] += f(self.marr)*self.dVc[iz]/sr_in_sqdeg * 0.4 / self.dcarr[ic] # 0.4 for N/dex to N/mag

    #t2 = time.time() ;  print("t loop:", t2-t1) ; t1 = time.time()
    return d4N

  def colormagcounts_incells(self, model, mfunction, myfilter1, myfilter2, **kwargs):
    """ Returns the number of galaxies in each bin of m, color (/sqdeg)"""

    d4N = self.dndmdzdc(model, mfunction, myfilter1, myfilter2, **kwargs)
    res = d4N * \
          self.dzarr.reshape(len(self.dzarr),1,1) * \
          self.dmarr.reshape(1,len(self.dmarr),1) * \
          self.dcarr.reshape(1,1,len(self.dcarr))

    res[(res == np.inf) | (res == np.nan)] = 0.
    return np.sum(res, axis=0) #sum over redshifts

  def colormagcounts_atz_incells(self, model, mfunction, myfilter1, myfilter2, z, **kwargs):
    """ Returns the number of galaxies in each bin of m, color (/sqdeg)"""

    d4N = self.dndmdzdc(model, mfunction, myfilter1, myfilter2, **kwargs)
    res = d4N * \
          self.dzarr.reshape(len(self.dzarr),1,1) * \
          self.dmarr.reshape(1,len(self.dmarr),1) * \
          self.dcarr.reshape(1,1,len(self.dcarr))
    res[(res == np.inf) | (res == np.nan)] = 0.
    izok = np.argmin(abs(self.zarr - z))
    return res[izok,:,:]


  def test_dndmdz(self):    
    import os
    import time

    prefix_templates = os.environ['ZPEG_ROOT']+'/data/templates/'
    fn = prefix_templates + 'Salp_200ages/Sb.dat'
    model = pypeg.Model()
    model.read_from_p2file(fn, sigma = 10.)

    prefix_filters = os.environ['ZPEG_ROOT']+'/data/filters/'
    fn = prefix_filters + 'u_prime.fil'
    myfilter = pypeg.Filter(filename = fn)

    TMF = Mfunction(np.arange(5.,13.,0.1))
    #TMF.Schechter(11.3,1e-3,-1.3)
    TMF.Schechter(11.3,1e-3,-1.3)
    #TMF.Schechter(11., 10**(-2.3), -0.6)

    if False:
      #self.dndmdz(model, TMF, myfilter)
      plt.figure(1)
      plt.clf()
      t1 = time.time()
      c = self.counts(model, TMF, myfilter, igm = False) 
      print('counts done in ',time.time()-t1)
      t1 = time.time()
      c2 = self.counts(model, TMF, myfilter, igm = True) 
      print('counts done in ',time.time()-t1)
      plt.ion()
      plt.plot(self.marr,(c2-c)/c, label = 'relative diff  w/wo IGM /sqdeg/dm=1')
      plt.yscale('log')

      t1 = time.time()
      n_incells = self.colormagcounts_incells(model, TMF, 
        pypeg.Filter(filename = prefix_filters+'u_prime.fil'), 
        pypeg.Filter(filename = prefix_filters+'i_prime.fil'), igm = True)

      print('counts done in ',time.time()-t1)
      print("nz, nm, nc, nM =",len(self.zarr), len(self.marr), len(self.carr), len(TMF.logM))

      #print n.shape
      #plt.plot(self.marr, np.sum( n_incells / 
      #  self.dmarr.reshape(len(self.marr),1)  
      #  , axis=1), label = 'sum(counts_col)*dc : N/mag')
      plt.legend(loc=0)

      plt.figure(3)
      plt.clf()
      n_incells[(n_incells<1e-5)&(n_incells>0.)] = 1e-6
      n_incells[n_incells==0.] = 1e-6
      plt.pcolormesh(self.marr, self.carr, np.log10(n_incells).transpose())
      plt.colorbar()
      plt.xlabel('mag')
      plt.ylabel('color')

      t1 = time.time()
      n = self.colormagcounts_incells(model, TMF, 
        pypeg.Filter(filename = prefix_filters+'u_prime.fil'), 
        pypeg.Filter(filename = prefix_filters+'i_prime.fil'), igm = True)
      print('counts done in ',time.time()-t1)
      t1 = time.time()
      n2 = self.colormagcounts_incells(model, TMF, 
        pypeg.Filter(filename = prefix_filters+'u_prime.fil'), 
        pypeg.Filter(filename = prefix_filters+'i_prime.fil'), igm = False)
      print('counts done in ',time.time()-t1)

      print('max diff=', np.max(np.abs(n2-n)))
      plt.figure(4)    
      plt.clf()
      n[(n<1e-5)&(n>0.)] = 1e-6
      n[n==0.] = 1e-6
      plt.pcolormesh(self.marr, self.carr, (n-n2).transpose())
      plt.colorbar()
      plt.title ('color mag diagram ')
      plt.xlabel('mag')
      plt.ylabel('color')

    plt.figure(5)
    plt.clf()
    plt.plot(self.marr, self.counts(model, TMF, 
      pypeg.Filter(filename = prefix_filters+'i_prime.fil')), '+-')
    plt.yscale('log')


def LF_to_MF(lf, model, filter0, z = 0., stellar = False, zfor = 10., mf = None):
  """ Converts the luminoisty function to a mass function.
  lf : Lfunction object (i.e. N(mag) / Mpc3 at a given redshift)
  model : Model object (evolving model SEDs)
  filter0 : Filter object (filter0 for the LF)
  z : redshift of the observed LF (lf)
  stellar : if True, the mass function is in stellar mass, not total mass
  zfor : redshift of formation for the model
  mf
  """

  mabs, zm = model.seds.absmags(filter0, zfor = zfor) #AB
  inozero = (zm>0.)
  mabsz0 = extrap(np.log10(np.array([1+z])),np.log10(1.+zm[inozero]), mabs[inozero])
  mstarsz0 = extrap(np.log10(np.array([1+z])),np.log10(1.+zm[inozero]), model.props.mstars[inozero])
  if not stellar:
    return lf.to_Mfunction(np.log10(model.norm), mabsz0, mf = mf) # 1Msun_total corresponds to mabsz0 in the model
  else:
    return lf.to_Mfunction(np.log10(mstarsz0), mabsz0, mf = mf) # 1Msun_total corresponds to mabsz0 in the model

def MF_to_LF(mf, model, filter0, z = 0., stellar = False, zfor = 10., lf = None):

  mabs, zm = model.seds.absmags(filter0, zfor = zfor) #AB
  inozero = (zm>0.)
  mabsz0 = extrap(np.log10(np.array([1+z])),np.log10(1.+zm[inozero]), mabs[inozero])
  mstarsz0 = extrap(np.log10(np.array([1+z])),np.log10(1.+zm[inozero]), model.props.mstars[inozero])
  if not stellar:
    return mf.to_Lfunction(np.log10(model.norm), mabsz0, lf = lf) # 1Msun_total corresponds to mabsz0 in the model
  else:
    return mf.to_Lfunction(np.log10(mstarsz0), mabsz0, lf = lf) # 1Msun_total corresponds to mabsz0 in the model

def TMF_to_SMF(TMF, model, z0, zfor = 10.):
  from copy import deepcopy

  z = pypeg.cosmic_z(model.props.time, zfor)
  inozero = (z>0)
  mstarsz0 = extrap(np.log10(np.array([1.+z0])),np.log10(1.+z[inozero]), model.props.mstars[inozero])

  SMF = deepcopy(TMF)
  SMF.N *= mstarsz0

  return SMF


def SMF_to_TMF(mfin, model, z0, zfor = 10.):
  from copy import deepcopy

  z = pypeg.cosmic_z(model.props.time, zfor)
  inozero = (z>0)
  mstarsz0 = extrap(np.log10(np.array([1+z0])),np.log10(1.+z[inozero]), model.props.mstars[inozero])

  TMF = deepcopy(mfin)
  TMF.N /= mstarsz0

  return TMF

def LF_to_LF(lfin, filterin, filterout, model, z0, zfor=10.):
  # shift the logxarr axis by the color difference between 2 filters, for a model at a giver redshift z0
  from copy import deepcopy

  mabsin, z = model.seds.absmags(filterin, zfor = zfor) #AB
  mabsout, z = model.seds.absmags(filterout, zfor = zfor) #AB

  inozero = (z>0)
  mabsinz0 = extrap(np.log10(np.array([1.+z0])),np.log10(1.+z[inozero]), mabsin[inozero])
  mabsoutz0 = extrap(np.log10(np.array([1.+z0])),np.log10(1.+z[inozero]), mabsout[inozero])

  lfout = deepcopy(lfin)
  lfout.logxarr += mabsoutz0 - mabsinz0

  return lfout


def SFRD(model, mfunction, zfor = 10.):

  """ returns model sfrd(z) for a given model with a given total mass function"""

  z = pypeg.cosmic_z(model.props.time, zfor)
  totalmass = mfunction.integrate(weights = 10.**mfunction.logM)
  SFRDarr = model.props.SFR*1e-6 * totalmass  # Msun/yr/Mpc3
  inozero = (z>0)

  return SFRDarr[inozero], z[inozero]

def plot_SFRD(models, TMFs, zfors, scenarios, withdata = True):

  from scipy import interpolate

  if withdata:
    zdata, logsfrddata, logsfrddata_err = pypeg.cosmic_sfh() # Hopkins compilation with Chabrier IMF
    plt.fill_between(zdata, logsfrddata-logsfrddata_err, logsfrddata+logsfrddata_err, facecolor='blue', alpha=0.5)
    plt.plot(zdata, logsfrddata, label= 'Hopkins compilation (Chabrier)')

  zarr = np.arange(0.,10.,0.1)
  sfrdtot = np.zeros(len(zarr))

  sfrds = []
  zs = []
  for i in range(len(models)):
    sfrd, z = SFRD(models[i], TMFs[i], zfor = zfors[i])
    sfrds.append(sfrd)
    zs.append(z)
    #interpolate onto zarr for later sum
    f = interpolate.interp1d(np.log10(1.+z), sfrd, bounds_error = False, fill_value = 0.)
    sfrdtot += f(np.log10(1.+zarr))

  if len(models) > 0:
    color=iter(plt.cm.rainbow(np.linspace(1,0,len(models)+1)))
    c = next(color)
    plt.plot(zarr, np.log10(sfrdtot), label = 'total', c=c)
    plt.xlabel('z')
    plt.ylabel('log10(SFRD (Msun/yr/Mpc$^3$)')
    plt.ylim([-3.,0.])  
    plt.xlim([0.,8.])  
    #plt.legend(loc=1,prop={'size':4})

  for i in range(len(models)):
    c = next(color)
    plt.plot(zs[i], np.log10(sfrds[i]), label = 'model {0} : {1}'.format(i, scenarios[i]), c=c)


  return zdata, logsfrddata, logsfrddata_err, zs, sfrds


def rhostar(model, mfunction, zfor = 10.):

  """ returns model rhostar(z) for a given model with a given total mass function"""

  z = pypeg.cosmic_z(model.props.time, zfor)
  totalmass = mfunction.integrate(weights = 10.**mfunction.logM)
  rhostararr = model.props.mstars * totalmass  # Msun/Mpc3
  inozero = (z>0)

  return rhostararr[inozero], z[inozero]

def plot_rhostar(models, TMFs, zfors, scenarios, withdata = True):

  from scipy import interpolate

  if withdata:
    elsner = np.genfromtxt('rhostar_Elsner08.tbl', dtype = "f8,f8,f8,f8", names = ('zmin','zmax','rhostar','err') ,invalid_raise=False, skip_header=0)
    perezg = np.genfromtxt('rhostar_PG08.dat', dtype = "f8,f8,f8,f8", names = ('zmin','zmax','rhostar','err') ,invalid_raise=False, skip_header=1)
    plt.errorbar((elsner['zmin']+elsner['zmax'])/2., elsner['rhostar'], elsner['err'], marker='^')
    plt.errorbar((perezg['zmin']+perezg['zmax'])/2., perezg['rhostar'], perezg['err'], marker='v')

  zarr = np.arange(0.,10.,0.1)
  rhostartot = np.zeros(len(zarr))

  rhostars = []
  zs = []
  for i in range(len(models)):
    myrhostar, z = rhostar(models[i], TMFs[i], zfor = zfors[i])
    rhostars.append(myrhostar)
    zs.append(z)
    #interpolate onto zarr for later sum
    f = interpolate.interp1d(np.log10(1.+z), myrhostar, bounds_error = False, fill_value = 0.)
    rhostartot += f(np.log10(1.+zarr))

  color=iter(plt.cm.rainbow(np.linspace(1,0,len(models)+1)))
  c = next(color)
  plt.plot(zarr, np.log10(rhostartot), label = 'total', c=c)

  for i in range(len(models)):
    c = next(color)
    plt.plot(zs[i], np.log10(rhostars[i]), label = 'model {0} : {1}'.format(i, scenarios[i]), c=c)

  plt.xlabel('z')
  plt.ylabel(r'log10($\rho$* (Msun/Mpc$^3$)')
  plt.ylim([6.,9.5])  
  plt.xlim([0.,8.])  
  plt.legend(loc=1,prop={'size':4})


















