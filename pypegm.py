from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import input
from builtins import str
from builtins import range

import functools
import numpy as np
import re
import pypeg.pypeg_io as io 
import pypeg.pypeg_utils as utils
from astropy import constants as const
from astropy import units as u
import astropy as ap
from copy import deepcopy
from scipy import interpolate, integrate
from astropy import cosmology


""" Natural units:
time : Myr
flux density : erg/s/A or erg/s/cm2/A
wavelength : angstrom

Routines : 
- def igm_transmission(wavelength, redshift)
- ldist_z(z) : in cm
- cosmic_z(age, zfor) : input in Myr
- cosmic_sfh() :   return z, log_CSFR, log_CSFR_err # log(Msun/yr/Mpc3)
- Class SSP
  - self.imf_name = ''
    self.imf_file = ''
    self.imf_number = 4
    self.m_min= 0.1
    self.m_max = 120.
    self.SNII_model = 'B'
    self.stellar_winds = 'y'
    self.stellib = 'stellibLCBcor.fits'
    self.stellib_number = 1
    self.ssp_prefix = 'Salp_B_sw_LCB'
  - write_input_file(file)
  - build()
  - run_ssp(file)
- Class Scenario
  - self.header = []
    self.output_file = 'test.fits'
    self.SSPs_file = 'Salp_B_sw_LCB_SSPs.dat'
    self.stellib = 'stellibLCBcor.fits'
    self.type_sf_dict = {'Instantaneous':0, 'Constant':1, 'Exponential':2, 'Schmidt':3}
    self.cb_fraction = 0.05
    self.init_metal=0.
    self.infall= True
    self.tau_infall= 1e4 # Myr
    self.infall_metal= 0.
    self.type_sf=self.type_sf_dict['Schmidt']
    self.p1=1.
    self.p2=1e4 #Myr
    self.sfr_file= ''
    self.tau_winds= 20001. # Myr
    self.consistent_metal= True
    self.stellar_metal= 0.
    self.substellar_mass= 0.
    self.nebular= True
    self.extinction= 0
    self.inclination= 0.
- write_scenarios(file, scenarios) : make .scn file from array of Scenarios
- run_scenarios_file(file)
- compute_scenarios(scenarios) : directly run array of scenarios

- Class Spectrum
  - w, nw, f
  - w,f = fromfile(file) : read from ascii 2-column file
  - normalize(w0,f0) : so that f(w0) = f0
  - smooth(wscale, w0 = median(w)) 

- Class Sedevol
  - self.time = time
    self.w    = w
    self.fevol = np.zeros((len(time),len(w)))
    self.dirac = dirac
    self.sigma = 10. # width of emission lines
  - __add__ : so that s1+s2 = addition of evolving spectra, 
              even with different w sampling. Times must be identical, though.
              Uses self.sigma to smooth Dirac lines.
  - smooth(wscale, w0)  : smooth evolving spectrum (uses Spectrum.smooth)
  - absmag(myfilter, zfor=10., calibtype='AB')
  - sed_in_ab(sfor=10, igm = True)
  - obsmag(myfilter, zfor = 10, calibtype = 'AB', igm = True)
  - sed_at_age(age) : interpolate, returns Spectrum with s.w, s.f being the spectrum

- Class Properties
  - self.nages
    self.time=np.zeros(self.nages)
    self.mgal=np.zeros(self.nages)
    self.mstars=np.zeros(self.nages)
    self.mWD=np.zeros(self.nages)
    self.mNSBH=np.zeros(self.nages)
    self.msubstell=np.zeros(self.nages)
    self.mgas=np.zeros(self.nages)
    self.Zism=np.zeros(self.nages)
    self.Zstars_mass=np.zeros(self.nages)
    self.Zstars_lumbol=np.zeros(self.nages)
    self.Lbol=np.zeros(self.nages)
    self.tauv=np.zeros(self.nages)
    self.Ldust_Lbol=np.zeros(self.nages)
    self.SFR=np.zeros(self.nages)
    self.nLym=np.zeros(self.nages)
    self.nSNII=np.zeros(self.nages)
    self.nSNIa=np.zeros(self.nages)
    self.age_stars_mass=np.zeros(self.nages)
    self.age_stars_Lbol=np.zeros(self.nages) 

- Class Model
  - self.scen = Scenario()
    self.props = Properties()
    self.norm = 1.
  - upscale(norm) : recompute parameters with normalisation !=1.
  - read_from_p2file(file, sigma=None): also do normalisation accroding to self.norm
  - read_from_fitsfile(file, sigma=None): idem
  - plotevol_sed_rf(ages = [10,100,1000])

- Class Filter(filename, fildir, transtype)
  - self.filename = 'unknown'
    self.fildir   = ''
    self.nw       = 10000
    self.transorig = Spectrum(nw = self.nw) # photon or energy transmission
    if transtype is None:
      self.transtype = -1 # O = energy ; 1 = photon
    else:
      self.transtype = transtype # force transmission type
    self.trans = Spectrum(nw = self.nw) # energy transmission
    self.area      = 0. # units = A
    self.areanu    = 0. # units = Hz
    self.wavemean  = 0. # units = A
    self.waveeff   = 0. # units = A
    self.fluxVega  = 0. # units = erg/s/cm^2/Angstrom
    self.fluxSun   = 0. # 
    self.ABVega    = 0. # units = AB mag
    self.ABSun     = 0. # units = AB mag
    self.error     = 0
    self.fildir = fildir
  - multintegrate(spectrum, weight)
  - calibrate() : same as calib.f (computes ABVega, fluxVega, etc.)
  - read_pegase_filter(pref, filename) :  looks in $ZPEG_ROOT if necessary, and calibrate
  - mag_to_flambda(m, calibtype='AB')
  - mag(spectrum, calibtype = 'AB') : from erg/s/cm2/A to AB
  - show() : plot f(w)
  - crop(wmin,wmax) : crop filter transmission
  


"""

cosmo_dict = None

def define_cosmo(mycosmo = None):
  """ Defines a default or cusotm comology (must be an astropy.cosmology one). And defines the module global cosmo_dict dictionary"""

  print('init cosmo...')

  if mycosmo is None:
    mycosmo = cosmology.Planck13

  #from astropy.cosmology import FlatLambdaCDM
  #mycosmo = FlatLambdaCDM(H0=70., Om0=1.)

  global cosmo_dict 

  cosmo_dict= {}

  #mycosmo = cosmology.WMAP7
  cosmo_dict['zscale0'] = np.logspace(0.,2.,100)-1.
  cosmo_dict['tscale0'] = mycosmo.age(cosmo_dict['zscale0']).to(u.Myr).value # Myr #Slow !!! Do only once....
  cosmo_dict['zscale'] = np.logspace(1e-10,2.,100)-1.
  cosmo_dict['tscale'] = mycosmo.age(cosmo_dict['zscale']).to(u.Myr).value # Myr #Slow !!! Do only once....
  cosmo_dict['ldistscale']  = mycosmo.luminosity_distance(cosmo_dict['zscale']).to(u.cm).value
  cosmo_dict['cosmo']  = mycosmo
  print('done init cosmo....')

# load cosmology on import, needed by other routines
#define_cosmo()

def memoize(f):
  class MemoizeMutable(object):
    """Memoize(fn) - an instance which acts like fn but memoizes its arguments
       Will work on functions with mutable arguments (slower than Memoize, but of for np.arrays)
    """
    def __init__(self, fn):
      self.fn = fn
      self.memo = {}
    def __call__(self, *args):
      import pickle
      str = pickle.dumps(args)
      if str not in self.memo:
        self.memo[str] = self.fn(*args)
      return self.memo[str]

  return MemoizeMutable(f)

@memoize
def make_and_read_from_p2file(*args, **kwargs):
  model = Model()
  model.read_from_p2file(*args,**kwargs)
  return model

@memoize
def igm_transmission_madauH(wavelength_obs, z_source):

  red_wavelength = np.zeros(5)

  # constants from Madau et al 1996
  # Ly alpha..delta coefficients and wavelength
  A = np.array([0.0036, 1.7e-3, 1.2e-3, 9.3e-4])
  wavelength = np.array([1216., 1026., 973., 950., 912.])


  xc = wavelength_obs / wavelength[4]
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
      igm_absorption[cond] += A[jl] * (wavelength_obs[cond] / wavelength[jl])**3.46

  # Add the photoelectric effect (shortwards 912A)
  cond = (wavelength_obs <= red_wavelength[4])
  for jl in range(4):
    igm_absorption[cond] += A[jl] * (wavelength_obs[cond] / wavelength[jl])**3.46
  igm_absorption[cond] += \
      0.25  * xc[cond]**3 * (xem**0.46 - xc[cond]**0.46) \
      + 9.4   * xc[cond]**1.5 * (xem**0.18 - xc[cond]**0.18) \
      - 0.7   * xc[cond]**3 * (xc[cond]**(-1.32) - xem**(-1.32)) \
      - 0.023 * (xem**1.68 - xc[cond]**1.68)

  igm_transmission = np.exp(-igm_absorption)

  return igm_transmission

@memoize
def igm_transmission(wavelengthin, redshift):
  from scipy.special import factorial

  """Intergalactic transmission (Meiksin, 2006)

  Compute the intergalactic transmission as described in Meiksin, 2006.

  Parameters
  ----------
  wavelength: array like of floats
     The wavelength(s) in Angstrom.
  redshift: float
     The redshift. Must be strictly positive.

  Returns
  -------
  igm_transmission: numpy array of floats
      The intergalactic transmission at each input wavelength.
  
  Taken from "pcigale" : 
  Copyright 2014 Yannick Roehlly, Mederic Boquien, Denis Burgarella
  Licensed under the CeCILL-v2 licence - see Licence_CeCILL_V2-en.txt
  Author: Yannick Roehlly, Mederic Boquien, Denis Burgarella

  """
  wavelength = 0.1 * wavelengthin # A to nm
  n_transitions_low = 10
  n_transitions_max = 31
  gamma = 0.2788  # Gamma(0.5,1) i.e., Gamma(2-beta,1) with beta = 1.5
  n0 = 0.25
  lambda_limit = 91.2  # Lyman limit in nm

  lambda_n = np.empty(n_transitions_max)
  z_n = np.empty((n_transitions_max, len(wavelength)))
  for n in range(2, n_transitions_max):
    lambda_n[n] = lambda_limit / (1. - 1. / float(n*n))
    z_n[n, :] = (wavelength / lambda_n[n]) - 1.

  # From Table 1 in Meiksin (2006), only n >= 3 are relevant. 
  # fact has a length equal to n_transitions_low.
  fact = np.array([1., 1., 1., 0.348, 0.179, 0.109, 0.0722, 0.0508, 0.0373, 0.0283])

  # First, tau_alpha is the mean Lyman alpha transmitted flux,
  # Here n = 2 => tau_2 = tau_alpha
  tau_n = np.zeros((n_transitions_max, len(wavelength)))
  if redshift <= 4:
    tau_a =       0.00211 * np.power(1. + redshift,  3.7)
    tau_n[2, :] = 0.00211 * np.power(1. + z_n[2, :], 3.7)
  elif redshift > 4:
    tau_a =       0.00058 * np.power(1. + redshift,  4.5)
    tau_n[2, :] = 0.00058 * np.power(1. + z_n[2, :], 4.5)

  # Then, tau_n is the mean optical depth value for transitions
  # n = 3 - 9 -> 1
  for n in range(3, n_transitions_max):
    if n <= 5:
      w = np.where(z_n[n, :] < 3)
      tau_n[n, w] = (tau_a * fact[n] *
        np.power(0.25 * (1. + z_n[n, w]), (1. / 3.)))
      w = np.where(z_n[n, :] >= 3)
      tau_n[n, w] = (tau_a * fact[n] *
        np.power(0.25 * (1. + z_n[n, w]), (1. / 6.)))
    elif 5 < n <= 9:
      tau_n[n, :] = (tau_a * fact[n] *
        np.power(0.25 * (1. + z_n[n, :]), (1. / 3.)))
    else:
      tau_n[n, :] = (tau_n[9, :] * 720. /
        (float(n) * (float(n*n - 1.))))

  for n in range(2, n_transitions_max):
    w = np.where(z_n[n, :] >= redshift)
    tau_n[n, w] = 0.
  
  z_l = wavelength / lambda_limit - 1.
  tau_l_igm = np.zeros_like(wavelength)
  w = np.where(z_l < redshift)
  tau_l_igm[w] = (0.805 * np.power(1. + z_l[w], 3) *
    (1. / (1. + z_l[w]) - 1. / (1. + redshift)))

  term1 = gamma - np.exp(-1.)

  n = np.arange(n_transitions_low - 1)
  term2 = np.sum(np.power(-1., n) / (factorial(n) * (2*n - 1)))

  term3 = ((1.+redshift) * np.power(wavelength[w]/lambda_limit, 1.5) -
           np.power(wavelength[w]/lambda_limit, 2.5))

  term4 = np.sum(np.array(
    [((2.*np.power(-1., n) / (factorial(n) * ((6*n - 5)*(2*n - 1)))) *
      ((1.+redshift) ** (2.5-(3 * n)) *
        (wavelength[w]/lambda_limit) ** (3*n) -
        (wavelength[w]/lambda_limit) ** 2.5))
    for n in np.arange(1, n_transitions_low)]), axis=0)

  tau_l_lls = np.zeros_like(wavelength)
  w = np.where(z_l < redshift)
  tau_l_lls[w] = n0 * ((term1 - term2) * term3 - term4)

  tau_taun = np.sum(tau_n[2:n_transitions_max, :], axis=0)
   
  lambda_min_igm = (1+redshift)*70.
  weight = np.ones_like(wavelength)
  w = np.where(wavelength < lambda_min_igm)
  weight[w] = np.power(wavelength[w]/lambda_min_igm, 2.)
  # Another weight using erf function can be used.
  # However, you would need to add: from scipy.special import erf
  #    weight[w] = 0.5*(1.+erf(0.05*(wavelength[w]-lambda_min_igm)))

  tau = tau_taun + tau_l_igm + tau_l_lls
  igm_transmission = np.exp(-tau) * weight

  return igm_transmission

def ldist_z(z):
  if cosmo_dict is None:
    define_cosmo()

  return 10.**np.interp(np.log10(z), 
    np.log10(cosmo_dict['zscale']), np.log10(cosmo_dict['ldistscale'])) #cm

def cosmic_z(galaxy_age, zfor):
  if cosmo_dict is None:
    define_cosmo()
  # input age in Myr. Returns z, given zfor
  tfor = np.interp(zfor, cosmo_dict['zscale'], cosmo_dict['tscale'])
  #print(cosmo_dict['zscale'], cosmo_dict['tscale'])
  #print(zfor,tfor)
  # time is increasing in the interp: good
  return np.interp(galaxy_age, np.array(cosmo_dict['tscale0'][::-1]) - tfor, cosmo_dict['zscale0'][::-1],
    right = np.nan, left = np.nan)

def cosmic_sfh():
  # Chabrier IMF
  z = np.logspace(0.,1.)-1.
  A = -0.997
  B = 0.241
  C = 0.180
  z0 = 1.243
  CSFR = np.array(C / (10.**(A*(z-z0)) + 10.**(B*(z-z0))))
  log_CSFR = np.log10(CSFR)

  log_CSFR_err = 0.*CSFR
  log_CSFR_err[z<0.5] = 0.13
  log_CSFR_err[(z>=0.5) & (z<0.9)] = 0.13
  log_CSFR_err[(z>=0.9) & (z<1.6)] = 0.17
  log_CSFR_err[(z>=1.6) & (z<3)] = 0.19
  log_CSFR_err[z>=3] = 0.27    
  return z, log_CSFR, log_CSFR_err # log(Msun/yr/Mpc3)

# ------------------------------------------------------------------
class SSP(object):
  """Class for PEGASE SSP"""
  def __init__(self):
    self.imf_name = ''
    self.imf_file = ''
    self.imf_number = 4
    self.m_min= 0.1
    self.m_max = 120.
    self.SNII_model = 'B'
    self.stellar_winds = 'y'
    self.stellib = 'stellibLCBcor.fits'
    self.stellib_number = 1
    self.ssp_prefix = 'Salp_B_sw_LCB'

  def write_input_file(self, file):
    with open(file, 'w') as f:
      f.write(str(self.imf_number)+'\n')
      f.write(str(self.m_min)+'\n')
      f.write(str(self.m_max)+'\n')
      f.write(self.SNII_model+'\n')
      f.write(self.stellar_winds+'\n')
      f.write(str(self.stellib_number)+'\n')
      f.write(self.ssp_prefix+'\n')

  def build(self, verbose = True, overwrite = False):
    import subprocess, os, glob

    if os.path.isfile(self.ssp_prefix+'_SSPs.dat'):
      if verbose:
        print("Warning : the SSPs {0} already exists.".format(self.ssp_prefix))
        if overwrite:
          print("Overwriting the SSPs...")
        else :
          print("Not doing anything...")

      if overwrite: # delete SSP files
        os.remove(self.ssp_prefix+'_SSPs.dat')
        for filename in glob.glob(self.ssp_prefix+'_tracks*.dat'):
          os.remove(filename)


    if not(os.path.isfile(self.ssp_prefix+'_SSPs.dat')):
      myfile = 'tmp_ssp_file.in'
      self.write_input_file(myfile)
      if verbose:
        subprocess.call(["SSPs_HR"], stdin = open(myfile))
      else:
        subprocess.call(["SSPs_HR"], stdin = open(myfile),stdout = open("/dev/null", "w"))
      os.remove(myfile)

 #-------------------------------------------------------------------
        
def run_ssp(file):
  import subprocess
  #subprocess.call(["spectra_HR", "file"], stdout = open("/dev/null", "w"))
  subprocess.call(["SSPs_HR"], stdin = open(file))

 #-------------------------------------------------------------------
 # 
 # ------------------------------------------------------------------
        
#-------------------------------------------------------------------
# ------------------------------------------------------------------
class Scenario(object):
  """Class for PEGASE scenario"""
  def __init__(self):
      self.header = []
      self.output_file = 'test.fits'
      self.SSPs_file = 'Salp_B_sw_LCB_SSPs.dat'
      self.stellib = 'stellibLCBcor.fits'
      self.type_sf_dict = {'time_SFR_Z':-2, 'time_SFR':-1, 'Instantaneous':0, 'Constant':1, 'Exponential':2, 'Schmidt':3}
      self.cb_fraction = 0.05
      self.init_metal=0.
      self.infall= True
      self.tau_infall= 1e4 # Myr
      self.infall_metal= 0.
      self.type_sf=self.type_sf_dict['Schmidt']
      self.p1=1.
      self.p2=1e4 #Myr
      self.sfr_file= ''
      self.tau_winds= 20001. # Myr
      self.consistent_metal= True
      self.stellar_metal= 0.
      self.substellar_mass= 0.
      self.nebular= True
      self.extinction= 0
      self.inclination= 0.
                     
  def make_from_header(self):
      """ Input = array of strings (PEGASE 2 header).-> stores the information into scenario"""
      pass #TBD#

  def make_from_fitsheader(self, header_str):
      """ Input = dictionary (fits header) -> into scenario information """
      pass #TBD#

def write_scenarios(file, scenarios):

  nscens = len(scenarios)
  # check all SSPs are the same
  ssps = [s.SSPs_file for s in scenarios]
  
  if len(set(ssps)) > 1:
    raise 

  with open(file, 'w+') as f:
    f.write('SSPs file: {0}\n'.format(ssps[0]))
    f.write('Fraction of close binary systems: {0:e}\n'.format(scenarios[0].cb_fraction))
    f.write('libstell : {0}\n'.format(scenarios[0].stellib))
    for i,s in enumerate(scenarios):
      #print s.p1, s.p2, s.tau_infall, s.tau_winds
      f.write("************************************************************\n")
      f.write('{0:>5d}: {1}\n'.format(i,s.output_file))

      f.write('Initial metallicity: {0:e}\n'.format(s.init_metal))

      f.write('{0}\n'.format(['No infall','Infall'][s.infall]))

      if s.infall:
          f.write('Infall timescale (Myr): {0:e}\n'.format(s.tau_infall))
          f.write('Metallicity of the infalling gas: {0:e}\n'.format(s.infall_metal))

      f.write('Type of star formation: {0:d}\n'.format(s.type_sf))

      if s.type_sf >= 1 and s.type_sf <=3 :    
          f.write('p1: {0:e}\n'.format(s.p1))
          f.write('p2: {0:e}\n'.format(s.p2))

      if s.type_sf == -1:
          f.write('SFR file: {0}\n'.format(s.sfr_file))

      f.write('{0}\n'.format(['No evolution of the stellar metallicity',\
                              'Consistent evolution of the stellar metallicity']\
                              [s.consistent_metal]))

      if not s.consistent_metal:
          f.write('Stellar metallicity: {0:e}\n'.format(s.stellar_metal))

      f.write('Mass fraction of substellar objects: {0}\n'.format(s.substellar_mass))

      if s.tau_winds > 20000.:
          f.write('No galactic winds\n')
      else:
          f.write('Galactic winds\n')
          f.write('Age of the galactic winds: {0:e}\n'.format(s.tau_winds))
      
      f.write('{0}\n'.format(['No nebular emission','Nebular emission'][s.nebular]))

      if s.extinction == 0:
          f.write('No extinction\n')
      elif s.extinction == 1:
          f.write('Extinction for a spheroidal geometry\n')
      elif s.extinction == 2:
          f.write('Extinction for a disk geometry: inclination-averaged\n')
      elif s.extinction == 3:
          f.write('Extinction for a disk geometry: specific inclination\n')
          f.write('Inclination: {0:e}\n'.format(s.inclination))

def run_scenarios_file(file, verbose = True):
  import subprocess

  if verbose:
    subprocess.call(["spectra_HR", file])
  else:
    subprocess.call(["spectra_HR", file], stdout = open("/dev/null", "w"))

def delete_files(scenarios, overwrite_without_prompt = True):
  import os
  # remove files first if needed ("overwrite")
  for f in [s.output_file for s in scenarios]:    
    if os.path.exists(f):
      if overwrite_without_prompt:
        os.remove(f)
      else:
        yy = input('Warning : the file '+f+' already exists ! delete it ? (y/n)')
        if (yy == 'y') or (yy == ''):
          os.remove(f)
        else:
          print('Try again with another filename, then.....')
          return


def compute_scenarios(scenarios, tmpfile = None, overwrite = False, verbose = True, skip_existing = False):
  import os

  if tmpfile is None:
    tmpfile = '/tmp/temp.scn'

  # skip some scenarios if wanted
  print('writing scenarios...')
  scens = scenarios[:]
  if skip_existing:
    for s in scens:
      if os.path.exists(s.output_file):
        print("removing "+s.output_file+" from scenarios to be computed")
        scenarios.remove(s)

  if len(scenarios)>0:
    write_scenarios(tmpfile, scenarios)
    delete_files(scenarios, overwrite_without_prompt = overwrite)

    print('computing scenarios...')
    run_scenarios_file(tmpfile, verbose = verbose)


  # Finally, read and return models from fits files
  print('reading models from fits files')
  models = []
  for f in [s.output_file for s in scens]:
    m = Model()
    m.read_from_fitsfile(f)
    models.append(m)

  return models

#-------------------------------------------------------------------
#-------------------------------------------------------------------
class Spectrum(object):

  def __init__(self,nw = None,w = None, f = None):
    if nw is None:
      if w is None:
        self.nw = 1
      else:   
        self.nw = len(w)
    else:
      self.nw = nw

    if (w is None) or (f is None):
      self.w = np.zeros(self.nw)
      self.f = np.zeros(self.nw)
    else:
      self.w = w
      self.f = f

  def fromfile(self, file, myskip = None):
    self.w, self.f = np.loadtxt(file, usecols=(0,1), unpack=True, skiprows = myskip)

  def normalize(self,w0,f0):
    i0 = (np.abs(self.w - w0)).argmin()
    self.f /= self.f[i0]

  def smooth(self, wscale, w0 = None):

    if w0 is None:
      w0 = np.median(self.w)

    i0 = (np.abs(self.w - w0)).argmin()
    nwind = (int(wscale / (self.w[i0+1]-self.w[i0])))/2*2+1 # odd number
    yhat = utils.smooth(self.f, window_len = nwind, window = 'hanning')
    self.f = yhat

  def dim_to_absolute(self):
    dimming = 4*np.pi*(10.*u.pc.to(u.cm))**2 # surface of a 10pc radius sphere in cm^2 
    self.f /= dimming


#-------------------------------------------------------------------
# ------------------------------------------------------------------        


class Sedevol(object):
  """Class for PEGASE evolutionary SED"""

  def __init__(self, time, w, dirac = False, sigma = 10.):

    self.time = time
    self.w    = w
    self.fevol = np.zeros((len(time),len(w)))
    self.dirac = dirac
    self.sigma = sigma # width of emission lines

  #def lineprofile(self, w, w0, sigma):
  #  return 1./(sigma * np.sqrt(2*np.pi)) * np.exp(-1./2.*((w-w0)/sigma)**2)
  

  def __add__(self, sed2): 

    from time import time
    """  
    Adds 2 evolutive spectra, which can have
    different wavelength sampling, in which case the resulting wavelength scale
    is the two scales joint and sorted, and the fluxes are interpolated before
    summation. 
    """ 

    from scipy.ndimage.filters import gaussian_filter1d

    if not isinstance(sed2, Sedevol):
      raise TypeError('unsupported operand type(s) for : \''+str(type(self))+'\' and \''+str(type(sed2))+'\'')

    #assert self.time == sed2.nt, 'Sedevol.__add__: operand self('+str(self)+') has different number of timesteps than other ('+str(sed2)+')'
    assert not((self.time-sed2.time).any()), 'Sedevol.__add__: operand self('+str(self)+') has some different timesteps than other ('+str(sed2)+')'

    allw = []
    if self.dirac is True:
      warray = [-1.*self.sigma, 0., self.sigma]
      for wline in self.w:
        allw.append(wline+warray)
    else:
      allw.append(self.w)

    if sed2.dirac is True:
      warray = [-1.*sed2.sigma, 0., sed2.sigma]
      for wline in sed2.w:
        allw.append(wline+warray)
    else:
      allw.append(sed2.w)


    #print(allw)
    #allw = np.unique(np.hstack(np.array(allw)))
    allw = np.unique(np.hstack(allw))
    #print(allw, allw.shape)
    newsed =  Sedevol(self.time, allw, dirac = False)


    if (self.dirac is True) and (self.sigma > 0):
      for iline, wline in enumerate(self.w):
        iok = np.where(np.abs(allw-wline) < self.sigma)[0]
        if len(iok)==1:          
          newsed.fevol[:,iok[0]] += 1./self.sigma * self.fevol[:,iline]#.reshape(len(self.time),1)
        else:
          #line_profile = 1./(self.sigma * np.sqrt(2*np.pi)) * np.exp(-1./2.*((allw[iok]-wline)/self.sigma)**2)
          warray = [-1.*self.sigma, 0., self.sigma]
          #for it in range(len(self.time)):
          #  f = interpolate.interp1d(wline+warray, [0.,1.,0.])
          #  newsed.fevol[it,iok] += 1.//self.sigma * self.fevol[it,iline] * f(allw[iok])
          fint = np.interp(allw[iok], wline+warray, [0.,1.,0.])
          newsed.fevol[:,iok] += 1./self.sigma * np.outer(self.fevol[:,iline], fint)

    else: # first operand spectra is not lines : store it, interpolated on allw
      for it in range(len(self.time)):
        finterp = interpolate.interp1d(self.w,self.fevol[it,:])
        newsed.fevol[it,:] += finterp(allw)


    if (sed2.dirac is True) and (sed2.sigma > 0): # second operand is lines: broaden lines, store them, interpolated on allw
      for iline, wline in enumerate(sed2.w):

        iokdebug = np.where(np.abs(allw-wline) < 20*sed2.sigma)[0] # where returns a tuple !
        iok = np.where(np.abs(allw-wline) < sed2.sigma)[0] # where returns a tuple !

        if len(iok)==1: # same as in "else", but faster
          newsed.fevol[:,iok[0]] += 1./sed2.sigma * sed2.fevol[:,iline]#.reshape(sed2.nt,1)
        else:
          #full line_profile = 1./(self.sigma * np.sqrt(2*np.pi)) * np.exp(-1./2.*((allw[iok]-wline)/self.sigma)**2)
          #for it in range(len(sed2.time)):
          #  newsed.fevol[it,iok] += 1./sed2.sigma * sed2.fevol[it,iline] * f(allw[iok])
          warray = [-1.*sed2.sigma, 0., sed2.sigma]
          #finterp = interpolate.interp1d(wline+warray, [0.,1.,0.], copy=False)
          #fint = finterp(allw[iok])
          fint = np.interp(allw[iok],wline+warray, [0.,1.,0.])
          newsed.fevol[:,iok] += 1./sed2.sigma * np.outer(sed2.fevol[:,iline], fint)
        if False:
          print('------------------------')
          print() 
          print(allw[iokdebug])
          print('f1:',np.interp(allw[iokdebug], self.w, self.fevol[3,:]))
          print('f2:',wline,sed2.fevol[3,iline])
          print('ff:',np.interp(allw[iokdebug], newsed.w, newsed.fevol[3,:]))
          print('ff:',newsed.fevol[3,iokdebug])
          print('------------------------')


    else:
      for it in range(len(sed2.time)):
        finterp = interpolate.interp1d(sed2.w, sed2.fevol[it,:])
        newsed.fevol[it,:] += finterp(allw)


    return newsed


  def smooth(self, wscale, w0 = None):
    for i in range(len(self.time)):
      self.fevol[i,:] = Spectrum(w = self.w, f = self.fevol[i,:]).smooth(wscale, w0 = w0).f

  # memoize:
  @functools.lru_cache(maxsize=128)
  def absmags(self, myfilter, zfor = None, calibtype = 'AB'):
    from copy import deepcopy
    #from astropy import unit
    # computes absolute magnitudes in a filter as a function of time

    if zfor is None:
      zfor = 10.
    z = cosmic_z(self.time, zfor)

    mag = np.zeros(len(self.time))
    for it in range(len(self.time)):
      sp = deepcopy(Spectrum(w = self.w, f = self.fevol[it,:]))
      sp.dim_to_absolute()
      mag[it] = myfilter.mag(sp, calibtype = calibtype) #AB magnitude is the default
    return mag, z



  def sed_in_ab(self, zfor = None, igm = True, fixed_age = None):
    #from astropy import units as units
    # computes observed magnitudes in a filter as a function of time
    # fixed_age in Myr if not None
    

    igm_trans = np.ones_like(self.w)
    sp_ab = np.repeat(Spectrum, len(self.time))

    if zfor is None:
      zfor = 10.

    c=2.99792458e18 # A/s

    z = cosmic_z(self.time, zfor)
    iok = (np.invert(np.isnan(z))) #booleans
    sqdimming = 0.*z
    sqdimming[iok] = np.sqrt(4.*np.pi)*ldist_z(z[iok])
    for itok in range(np.sum(iok)):
      it = np.arange(len(self.time))[iok][itok]
      if fixed_age is None:
        it_origin = it
      else:
        it_origin = np.abs(self.time - fixed_age).argmin()

      if igm:
        igm_trans = igm_transmission(self.w*(1.+z[it]), z[it]) 

      sp_ab[it] = Spectrum(w = self.w * (1.+z[it]), f = igm_trans * self.fevol[it_origin,:] / (1.+z[it]) )
      fnu = sp_ab[it].f * (sp_ab[it].w)**2 / c
      #sp_ab[it].f = -2.5 * np.log10(fnu) - 48.60 + 5.*np.log10(sqdimming[it])
      sp_ab[it].ab = -2.5 * np.log10(fnu) - 48.60 + 5.*np.log10(sqdimming[it])
      #print(sp_ab[it].ab)
      #print(z[iok][itok], z[it])
      
    return sp_ab[iok], z[iok]
  
  # memoize:
  @functools.lru_cache(maxsize=128)
  def obsmags(self, myfilter, zfor = 10., calibtype = 'AB', igm = True, fixed_age = None):
    #from astropy import units as units
    # computes observed magnitudes in a filter as a function of time
    #from astropy import cosmology
    #import time

    igm_trans = np.ones_like(self.w)
    z = cosmic_z(self.time, zfor)
    iok = (np.invert(np.isnan(z)))
    sqdimming = 0.*z
    sqdimming[iok] = np.sqrt(4.*np.pi)*ldist_z(z[iok])
    mag = np.zeros(len(self.time))
    distmod = np.zeros(len(self.time))
    for itok in range(np.sum(iok)):
      it = np.arange(len(self.time))[iok][itok]
      if fixed_age is None:
        it_origin = it
      else:
        it_origin = np.abs(self.time - fixed_age).argmin()

      if igm:
        igm_trans = igm_transmission(self.w*(1.+z[it]), z[it]) 
      sp = Spectrum(w = self.w * (1.+z[it]), f = igm_trans * self.fevol[it_origin,:] / (1.+z[it]) )
      mag[it] = myfilter.mag(sp, calibtype=calibtype) +5.*np.log10(sqdimming[it]) #AB magnitude is the default
      distmod[it] = 5.*np.log10(sqdimming[it])
   # mag[(np.isnan(z))] = np.nan
    return np.flip(mag[iok]), np.flip(z[iok]), np.flip(distmod[iok])

  def sed_at_age(self, age):
    if age in self.time:
      #iage = np.abs(self.time - age).argmin()
      #return self.fevol[iage,:]
      return Spectrum(w=self.w, f = self.fevol[np.searchsorted(self.time, age),:])
    else:
      # need to interpolate...
      itok = (self.time > 0)
      myfevol = deepcopy(self.fevol)
      iflow = (myfevol == 0.)
      myfevol[iflow] = 1e-99
      f = interpolate.interp1d(np.log10(self.time[itok]), np.log10(myfevol[itok,:]), axis = 0 ,kind = "slinear")
      myf = 10.**f(np.log10(age))
      iflow = ( np.abs(myf/1e-99 -1) < 1e-5)
      myf[iflow] = 0.
      return Spectrum(w = self.w, f = myf)

  
#-------------------------------------------------------------------
# ------------------------------------------------------------------
class Properties(object):
  """Class for PEGASE evolving physical properties : SFR, etc."""

  def __init__(self, nages=None):
    """Initialising properties"""
    if nages is None:
      self.nages=200
    else:
      self.nages=nages                        

    self.time=np.zeros(self.nages)
    self.mgal=np.zeros(self.nages)
    self.mstars=np.zeros(self.nages)
    self.mWD=np.zeros(self.nages)
    self.mNSBH=np.zeros(self.nages)
    self.msubstell=np.zeros(self.nages)
    self.mgas=np.zeros(self.nages)
    self.Zism=np.zeros(self.nages)
    self.Zstars_mass=np.zeros(self.nages)
    self.Zstars_lumbol=np.zeros(self.nages)
    self.Lbol=np.zeros(self.nages)
    self.tauv=np.zeros(self.nages)
    self.Ldust_Lbol=np.zeros(self.nages)
    self.SFR=np.zeros(self.nages)
    self.nLym=np.zeros(self.nages)
    self.nSNII=np.zeros(self.nages)
    self.nSNIa=np.zeros(self.nages)
    self.age_stars_mass=np.zeros(self.nages)
    self.age_stars_Lbol=np.zeros(self.nages) 


  def getprop(self, propname, zfor = 10.):
    print("zfor=",zfor)
    z = cosmic_z(self.time, zfor)
    #print self.time, z
    iok = (np.invert(np.isnan(z)))
    myprop = np.zeros(len(self.time))
    for itok in range(np.sum(iok)):
      it = np.arange(len(self.time))[iok][itok]
      myprop[it] = eval('self.'+propname)[it]

    return myprop, z


#-------------------------------------------------------------------
# ------------------------------------------------------------------

 #-------------------------------------------------------------------
 # 
 # ------------------------------------------------------------------
        
#-------------------------------------------------------------------
# ------------------------------------------------------------------
class Model(object):
  """Class for a PEGASE evolutionary model"""

  def __init__(self):
    # define scenario and props
    self.scen = Scenario()
    self.props = Properties()
    self.norm = 1.

  def upscale(self, norm, oldnorm = None):
    if oldnorm is None: # default case : we rescale respectively to previous norm.
      oldnorm = self.norm

    self.norm             = norm
    self.props.mgal      *= norm/oldnorm
    self.props.mstars    *= norm/oldnorm
    self.props.mWD       *= norm/oldnorm
    self.props.mNSBH     *= norm/oldnorm
    self.props.msubstell *= norm/oldnorm
    self.props.mgas      *= norm/oldnorm
    self.props.Lbol      *= norm/oldnorm
    self.props.SFR       *= norm/oldnorm
    self.props.nLym      *= norm/oldnorm
    self.props.nSNII     *= norm/oldnorm
    self.props.nSNIa     *= norm/oldnorm
    self.props.nLym      *= norm/oldnorm
    try:
      self.seds_cont.fevol   *= norm/oldnorm
    except:
      pass # no SED defined yet.

    try:
      self.seds_lines.fevol      *= norm/oldnorm
    except:
      pass # no SED defined yet.

    try:
      self.seds.fevol      *= norm/oldnorm
    except:
      pass # no SED defined yet.

  def read_from_p2file(self,filename, sigma = None, verbose = True):
    print('Reading scenario...')
    with open(filename,'r') as f:
      # read header
      self.scen=Scenario()
      isheader = True
      while isheader:
        self.scen.header.append(f.readline())
        if self.scen.header[-1].find('****',0) >= 0:
          isheader=False

      # make scenario from header
      self.scen.make_from_header() # TBD
      self.scen.output_file = filename

      # read array dimensions
      (nt,nl,nllines) = tuple(int(v) for v in re.findall("[0-9]+", f.readline()))

      # define seds and props
      warr = io.myread(f, nl)
      self.seds_cont = Sedevol(np.zeros(nt), warr)
      warr = io.myread(f, nllines)
      self.seds_lines = Sedevol(np.zeros(nt), warr, dirac = True)
      self.props = Properties(nages=nt)

      # loop over timesteps
      for it in np.arange(nt):
        line=f.readline()
        arrline=[np.float(i) for i in line.rstrip().split()]
        self.props.time[it]          = arrline[0]
        self.props.mgal[it]          = arrline[1]
        self.props.mstars[it]        = arrline[2]
        self.props.mWD[it]           = arrline[3]
        self.props.mNSBH[it]         = arrline[4]
        self.props.msubstell[it]     = arrline[5]
        self.props.mgas[it]          = arrline[6]
        self.props.Zism[it]          = arrline[7]
        self.props.Zstars_mass[it]   = arrline[8]
        self.props.Zstars_lumbol[it] = arrline[9]

        line=f.readline()
        arrline=[np.float(i) for i in line.rstrip().split()]
        self.props.Lbol[it]          = arrline[0]
        self.props.tauv[it]          = arrline[1]
        self.props.Ldust_Lbol[it]    = arrline[2]
        self.props.SFR[it]           = arrline[3]
        self.props.nLym[it]          = arrline[4]
        self.props.nSNII[it]         = arrline[5]
        self.props.nSNIa[it]         = arrline[6]
        self.props.age_stars_mass[it]= arrline[7]
        self.props.age_stars_Lbol[it]= arrline[8]

        self.seds_cont.time[it] = self.props.time[it]
        self.seds_lines.time[it] = self.props.time[it]
        self.seds_cont.fevol[it,:]=io.myread(f,nl)
        self.seds_lines.fevol[it,:]=io.myread(f,nllines)

      # add emission lines to seds
      if sigma is not None:
        self.seds_lines.sigma = sigma
      # magic addition involving interpolation, cf __add__ in Sedsevol classs
      self.seds = self.seds_cont + self.seds_lines 
                
      if self.norm != 1. :
        self.upscale(self.norm, oldnorm = 1.)

    if verbose:
      print('Done...')

  def read_from_fitsfile(self, filename, sigma = None):
    from astropy.io import fits

    hdulist = fits.open(filename)

    #header
    self.scen=Scenario() # To be filled in later
    self.scen.make_from_fitsheader(hdulist[0].header)
    nl = hdulist[0].header['NAXIS1']
    nt = hdulist[0].header['NAXIS2']
    nllines = hdulist[1].header['NAXIS2']

    warr = hdulist[3].data.field(0) # continuum for LCB lib... TBD for HR
    self.seds_cont = Sedevol(np.zeros(nt), warr)
    warr = hdulist[1].data.field(0) # lines
    self.seds_lines = Sedevol(np.zeros(nt), warr, dirac = True)


    self.props = Properties(nages=nt)
    self.props.time          = hdulist[2].data['AGE']
    self.seds_lines.time     = self.props.time
    self.seds_cont.time      = self.props.time
    self.props.mgal          = hdulist[2].data['MGAL']
    self.props.mstars        = hdulist[2].data['Mstars']
    self.props.mWD           = hdulist[2].data['MWD']
    self.props.mNSBH         = hdulist[2].data['MBHNS']
    self.props.msubstell     = hdulist[2].data['Msub']
    self.props.mgas          = hdulist[2].data['sigmagas']
    self.props.Zism          = hdulist[2].data['Zgas']
    self.props.Zstars_mass   = hdulist[2].data['Zstars']
    self.props.Zstars_lumbol = hdulist[2].data['Zbol']
    self.props.Lbol          = hdulist[2].data['fluxbol']
    self.props.tauv          = hdulist[2].data['tauV']
    self.props.Ldust_Lbol    = hdulist[2].data['fluxext']
    self.props.SFR           = hdulist[2].data['SFR']
    self.props.nLym          = hdulist[2].data['NLymtot']
    self.props.nSNII         = hdulist[2].data['nSNIItot']
    self.props.nSNIa         = hdulist[2].data['nSNIatot']
    self.props.age_stars_mass= hdulist[2].data['agestars']
    self.props.age_stars_Lbol= hdulist[2].data['agebol']

    Lsun=3.826e33 # erg/s BECAUSE in FITS files, spectra are normalized to Lsun
    self.seds_cont.fevol = Lsun*hdulist[0].data # ntimes  x nlambda array
    self.seds_lines.fevol = Lsun*hdulist[1].data.field(1).transpose() # ntimes  x nlines array

    # add lines to continuum
    if sigma is not None:
      self.seds_lines.sigma = sigma

    # magic addition involving interpolation, cf __add__ in Sedsevol classs
    self.seds = self.seds_cont + self.seds_lines
    self.seds.time           = self.props.time
                
    if self.norm != 1. :
      self.upscale(self.norm, oldnorm = 1.)

    hdulist.close()


  def plotevol_sed_rf(self, ages=[10,100,1000,10000], **kwargs):
    """ Plot the evolution of the SED in the RestFrame"""
    import matplotlib.pyplot as plt
        
    #plt.ion()
    #fig=plt.figure()
    #ax=fig.add_subplot(1,1,1)
    plt.xscale("log",nonposx='clip')
    plt.yscale("log",nonposy='clip')
    plt.xlim([700.,20000.])

    for i in ages:
      mysed = self.seds.sed_at_age(i)
      plt.plot(mysed.w, mysed.f,linewidth=2.)

    leg=plt.legend([str(a)+' Myr' for a in ages], loc = 0, **kwargs)
    for l in leg.get_lines():
      l.set_linewidth(2)  # the legend line width

    #plt.show()

 #-------------------------------------------------------------------
 # 
 # ------------------------------------------------------------------
        
#-------------------------------------------------------------------
# ------------------------------------------------------------------
class Filter(object):

  import os

  """
  Class for photometric filter
  """
  def __init__(self, filename = None, fildir = None, transtype = None):
    import os
    self.filename = 'unknown'
    self.fildir   = ''
    self.nw       = 10000
    self.transorig = Spectrum(nw = self.nw) # photon or energy transmission
    if transtype is None:
      self.transtype = -1 # O = energy ; 1 = photon. Defined from first value in file. 
    else:
      self.transtype = transtype # force transmission type, 0 or 1.
    self.trans = Spectrum(nw = self.nw) # energy transmission
    self.area      = 0. # units = A
    self.areanu    = 0. # units = Hz
    self.wavemean  = 0. # units = A
    self.waveeff   = 0. # units = A
    self.fluxVega  = 0. # units = erg/s/cm^2/Angstrom
    self.fluxSun   = 0. # 
    self.ABVega    = 0. # units = AB mag
    self.ABSun     = 0. # units = AB mag
    self.error     = 0
        
    pref = ''
    if  fildir != None:
      self.fildir = fildir
      pref = fildir + '/'
        
    if filename != None:
      self.filename = filename
      self.read_pegase_filter(pref,filename)
      self.name = os.path.splitext(os.path.basename(filename))[0]

  def multintegrate(self, spectrum,  weight = None, integ_type = 'double_interp'): # 'interp_on_trans', 'double_interp'
    """ Assumes spectrum.w is sorted increasingly """

    #if tofnu is None:
    #  tofnu = False # spectrum is flambda

    if weight is None:
      weight = np.ones(self.nw)

    # return nan if spectrum wavelengths have some nans....
    if np.isnan(np.sum(spectrum.w)):
      return np.nan

    is_sorted = np.all(np.diff(spectrum.w)>=0)
    if not(is_sorted):
      print("WARNING : wavelengths not sorted in call to multintegrate !!!!")

    #integ_type = 'interp_on_trans'
    #integ_type = 'double_interp'

    if integ_type == 'interp_on_trans': # interpolate input spectrum on transmission curve

      fluxtot = 0.
      # spectrum flux interpolated on transmission wavelengths. Not good is there are emssion lines
      fint = np.interp(self.trans.w, spectrum.w, spectrum.f) # faster than interp1d. extrapolations set to limits
      fint[self.trans.w < spectrum.w[0]] = 0.
      fint[self.trans.w > spectrum.w[-1]] = 0.
      try:
       #      finterp = interpolate.interp1d(spectrum.w,spectrum.f)
       # if tofnu is False:
       #        fluxtot = integrate.trapz(weight * finterp(self.trans.w) * self.trans.f, self.trans.w)
        fluxtot = integrate.trapz(weight * fint * self.trans.f, self.trans.w) # check tofnu keyword ? is it different when we integrate over fnu ?
        #else:
        #  c=2.99792458e18 # A/s
        #  fnu = self.trans.f * self.trans.w ** 2 / c
        ## fluxtot = integrate.trapz(weight * finterp(self.trans.w) * fnu, self.trans.w)
        #  fluxtot = integrate.trapz(weight * fint * fnu, self.trans.w)
      except ValueError:
        print("Error in integrating spectrum on filter transmission")
        raise

    if (integ_type == 'double_interp'): # interpolate input spectrum on transmission curve
      fluxtot = 0.
      allw = np.sort(np.unique(np.hstack((self.trans.w, spectrum.w))))
      f1 = np.interp(allw, spectrum.w, spectrum.f) # faster than interp1d. extrapolations set to limits
      f2 = np.interp(allw, self.trans.w, weight * self.trans.f) # faster than interp1d. extrapolations set to limits
      ioutcommon = (allw<np.min(self.trans.w)) | (allw<np.min(spectrum.w)) | (allw>np.max(self.trans.w)) | (allw>np.max(spectrum.w))
      f1[ioutcommon] = 0.
      f2[ioutcommon] = 0.
      #fint=f1*f2
      try:
        fluxtot = integrate.trapz(f1*f2, allw) # check tofnu keyword ? is it different when we integrate over fnu ?
      except:
        print("Error in integrating spectrum on filter transmission")
        raise


    return fluxtot #erg/s/cm^2/A*A in filter

  def calibrate(self):
        
    import os

    self.area = integrate.trapz(self.trans.f, self.trans.w) # Angstrom
    self.wavemean = integrate.trapz(self.trans.w * self.trans.f, self.trans.w) / self.area

    c=2.99792458e18 # A/s
    transfnu = self.trans.f / self.trans.w ** 2 * c # Hz/A
    self.areanu = integrate.trapz(transfnu, self.trans.w) # Hz

    vega = Spectrum()
    vega.fromfile(os.getenv('ZPEG_ROOT')+'/data/VegaLCB_IR.dat', myskip = 1)
    self.fluxVega = self.multintegrate(vega) / self.area # erg/s/cm^2/Angstrom
    self.waveeff = self.multintegrate(vega, weight = self.trans.w) / self.fluxVega / self.area
    self.ABVega = -2.5 * np.log10(self.fluxVega * self.area/ self.areanu ) - 48.60

    sun = Spectrum()
    sun.fromfile(os.getenv('ZPEG_ROOT')+'/data/SunLCB.dat', myskip = 1)
    self.fluxSun = self.multintegrate(sun) / self.area # erg/s/cm^2/Angstrom
    self.ABSun = -2.5 * np.log10(self.fluxSun * self.area/ self.areanu ) - 48.60


  def read_pegase_filter(self,pref,filename):
    import os

    # Check file exists and looks in $ZPEG_ROOT if necessary
    mypref = ''
    try:
      with open(pref+filename):
        mypref = pref
    except:
      try:
        with open(os.getenv('ZPEG_ROOT')+'/data/filters/'+filename):
          mypref = os.getenv('ZPEG_ROOT')+'/data/filters/'
      except:
        print("impossible to open the filter in dir "+pref+" or $ZPEG_ROOT")
        raise

    # read file
    myfile = mypref + filename
    i = 0
    with open(myfile,'r') as f:
      for line in f:
        tab = line.strip().split()

        if (len(tab)==1) and ((tab[0] == '0') or (tab[0] == '1')) and (self.transtype==-1):
          self.transtype = np.int(tab[0])

        if (len(tab) > 1) and (not tab[0].startswith('#')):
          self.trans.w[i] = np.float(tab[0])
          self.trans.f[i] = np.float(tab[1])
          i = i+1

    # clip and calibrate
    self.nw = i
    self.trans.w = self.trans.w[:i]
    self.trans.f = self.trans.f[:i]

    if self.transtype == -1:
      print("Setting filter ", filename, " to photon transmission (=default)")
      self.transtype = 1 # default = photon transmission

    self.transorig = deepcopy(self.trans)
    if self.transtype == 1: # photon transmission
      self.trans.f = self.trans.f * self.trans.w

    self.calibrate()

  def mag_to_flambda(self, m, calibtype = 'AB'):

    if (calibtype == 'Vega'):
      try:
        f = self.fluxVega * 10.**(-0.4*(m - 0.03)) #erg/s/cm^2/A in filter
      except:                
        self.error
        raise
    elif (calibtype == 'AB'):
      try:
        f = self.areanu / self.area * 10.**(-0.4*(m + 48.60)) #erg/s/cm^2/A in filter
      except:
        self.error
        raise

    return f
    
  def mag(self, spectrum, calibtype = 'AB'): #from erg/s/cm2/A to AB or Vega mag
 
    magout = -99.
    if (calibtype == 'Vega'):
      try:
        magout = -2.5 * np.log10(self.multintegrate(spectrum) / self.fluxVega / self.area) + 0.03
      except:                
        #self.error
        raise
    elif (calibtype == 'AB'):
      try:
        old_settings = np.seterr(divide='ignore') # if mag = infinity, deal with it....
        magout = -2.5 * np.log10(self.multintegrate(spectrum) / self.areanu ) - 48.60
        np.seterr(**old_settings)
      except:
        #self.error
        raise

    return magout

  def show(self):
    import matplotlib.pyplot as plt

    plt.ion()
    plt.figure()
    plt.plot(self.trans.w,self.trans.f)
    plt.show()

  def crop(self,wmin,wmax):

    iok = (self.trans.w <= wmax) & (self.trans.w >= wmin)

    if min(iok) == False:  # at least some bad wavelengths
      self.trans.w = self.trans.w[iok]
      self.trans.f = self.trans.f[iok]
      self.nw = len(self.trans.w)
      self.calibrate()



if __name__ == '__main__':
  pass