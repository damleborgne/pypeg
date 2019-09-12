 def multintegrate(transmission, spectrum,  weight = None, integ_type = 'double_interp'): 
    """ Assumes spectrum.w is sorted increasingly 

    Parameters
    ----------
    transmission : object 
      contains at least two attributes : w = wavelength, and """

    if weight is None:
      weight = np.ones(transmission.nw)

    # return nan if spectrum wavelengths have some nans....
    if np.isnan(np.sum(spectrum.w)):
      return np.nan

    is_sorted = np.all(np.diff(spectrum.w)>=0)
    if not(is_sorted):
      print("WARNING : wavelengths not sorted in call to multintegrate !!!!")

    if integ_type == 'interp_on_trans': # interpolate input spectrum on transmission curve

      fluxtot = 0.
      # spectrum flux interpolated on transmission wavelengths. Not good is there are emission lines
      fint = np.interp(transmission.w, spectrum.w, spectrum.f) # faster than interp1d. extrapolations set to limits
      fint[transmission.w < spectrum.w[0]] = 0.
      fint[transmission.w > spectrum.w[-1]] = 0.
      try:
        fluxtot = integrate.trapz(weight * fint * transmission.f, transmission.w) # check tofnu keyword ? is it different when we integrate over fnu ?
      except ValueError:
        print("Error in integrating spectrum on filter transmission")
        raise

    if (integ_type == 'double_interp'): # interpolate input spectrum on transmission curve
      fluxtot = 0.
      allw = np.sort(np.unique(np.hstack((transmission.w, spectrum.w))))
      f1 = np.interp(allw, spectrum.w, spectrum.f) # faster than interp1d. extrapolations set to limits
      f2 = np.interp(allw, transmission.w, weight * transmission.f) # faster than interp1d. extrapolations set to limits
      ioutcommon = (allw<np.min(transmission.w)) | (allw<np.min(spectrum.w)) | (allw>np.max(transmission.w)) | (allw>np.max(spectrum.w))
      f1[ioutcommon] = 0.
      f2[ioutcommon] = 0.
      try:
        fluxtot = integrate.trapz(f1*f2, allw) # check tofnu keyword ? is it different when we integrate over fnu ?
      except:
        print("Error in integrating spectrum on filter transmission")
        raise


    return fluxtot #erg/s/cm^2/A*A in filter

