#!/usr/bin/env python

###################################################################
# A simple module that fits a 2D PSF analytic profile to an image #
###################################################################

import numpy as np
import warnings
import emcee
from scipy import optimize
import astropy.modeling as modeling

class PSFfit(object):
    '''
    A class of object used to perform PSF fitting and astrometry

    Attributes:
        fitboxsize:
        data_stamp:
        ivar_stamp:
        fitx:
        fity:
        fitf:
        figbg:
        bestfitmodel:

    Methods:
        generate_data_stamp:
    '''

    @property
    def fitboxsize(self):
        return self._fitboxsize
    @fitboxsize.setter
    def fitboxsize(self, newval):
        self._fitboxsize = newval

    @property
    def data_stamp(self):
        return self._data_stamp
    @data_stamp.setter
    def data_stamp(self, newval):
        self._data_stamp = newval

    @property
    def ivar_stamp(self):
        return self._ivar_stamp
    @ivar_stamp.setter
    def ivar_stamp(self, newval):
        self._ivar_stamp = newval

    @property
    def fitx(self):
        return self._fitx
    @fitx.setter
    def fitx(self, newval):
        self._fitx = newval

    @property
    def fity(self):
        return self._fity
    @fity.setter
    def fity(self, newval):
        self._fity = newval

    @property
    def fitf(self):
        return self._fitf
    @fitf.setter
    def fitf(self, newval):
        self._fitf = newval

    @property
    def bestfitmodel(self):
        return self._bestfitmodel
    @bestfitmodel.setter
    def bestfitmodel(self, newval):
        self._bestfitmodel = newval

    def __init__(self, fitboxsize=None):

        self.fitboxsize = fitboxsize
        self.data_stamp = None
        self.ivar_stamp = None
        self.fitx = None
        self.fitxerr = None
        self.fity = None
        self.fityerr = None
        self.fitf = None
        self.fitferr = None
        self.fitfwhmx = None
        self.fitfwhmxerr = None
        self.fitfwhmy = None
        self.fitfwhmyerr = None
        self.fittheta = None
        self.fitthetaerr = None
        self.fitbeta = None
        self.fitbetaerr = None
        self.fitbg = 0.
        self.fitbgerr = None
        self.bestfitmodel = None
        self.lnlike = -np.inf

    def generate_data_stamp(self, frame, ivar=None, x0=None, y0=None, method='maxpix', r_tol=None):
        '''
        Crop a box of side length fitboxsize around (xc, yc) for psf fitting, where (xc, yc) will be selected using
        the specified method.

        Args:
            frame: a 2D image
            ivar: corresponding inverse variance of the image
            x0, y0: center coordinate around which to crop data (i.e. initial guess centroid of the target)
            method: 'input', crops around the input indices [int(x0), int(y0)]
                    'maxpix', crops around the maximum pixel. If (x0, y0) are given, then find the maximum pixel
                    within a fitbox centered at (x0, y0), otherwise, find the maximum pixel over the whole frame,
                    then, crop around this maximum pixel location (xc, yc).
                    'fullframe', use the given frame without cropping
            r_tol: tolerance for how far away (in pixels in either direction) the new cropping center can be from
                   the provided input

        Returns:
            initializes relevant class attributes
        '''

        if len(frame.shape) != 2:
            raise ValueError('image data must be a 2D array')
        if ivar is None:
            ivar = np.ones(frame.shape)
        if frame.shape != ivar.shape:
            print('shapes of data frames and inverse variance frames do not match, default to unit variances...')

        if self.fitboxsize is None:
            method = 'fullframe'
            self.fitboxsize = np.min(frame.shape)

        if method.lower() != 'input' and method.lower() != 'maxpix' and method.lower() != 'fullframe':
            raise ValueError('data stamp cropping method: {} is not supported'.format(method))

        if r_tol is None:
            r_tol = self.fitboxsize // 2

        mask = np.ones(frame.shape)
        if (x0 is not None) and (y0 is not None):
            xc = np.rint(x0).astype(int)
            yc = np.rint(y0).astype(int)

            if method.lower() == 'input':
                ymin, ymax, xmin, xmax = find_ind_bounds(yc, xc, self.fitboxsize)
                stamp = frame[ymin: ymax, xmin: xmax]
                ivar_stamp = ivar[ymin: ymax, xmin:xmax]
            else:
                # create mask that masks the pixels outside the fitbox
                ymin, ymax, xmin, xmax = find_ind_bounds(yc, xc, r_tol * 2 + 1)
                mask[ymin:ymax, xmin:xmax] = 0
                mask = 1 - mask

        if method.lower() == 'maxpix':
            # define maximum pixel location as new center around which to crop
            yc, xc = np.unravel_index(np.nanargmax(frame * mask), frame.shape)
            ymin, ymax, xmin, xmax = find_ind_bounds(yc, xc, self.fitboxsize)
            stamp = frame[ymin: ymax, xmin: xmax]
            ivar_stamp = ivar[ymin: ymax, xmin:xmax]
        elif method.lower() == 'fullframe':
            if frame.shape[0] != frame.shape[1]:
                warnings.warn('only square image fitting are supported, automatically cropping a square of the smaller '
                              'dimension...')
            self.fitboxsize = np.min(frame.shape)
            xc = int(self.fitboxsize // 2)
            yc = int(self.fitboxsize // 2)
            stamp = frame[:self.fitboxsize,:self.fitboxsize]
            ivar_stamp = ivar[:self.fitboxsize,:self.fitboxsize]

        self.fitx = xc
        self.fity = yc
        self.data_stamp = stamp
        self.ivar_stamp = ivar_stamp
        self.data_stamp_x_center = xc
        self.data_stamp_y_center = yc

def find_ind_bounds(yc, xc, boxsize, ylim=np.inf, xlim=np.inf, warn=True):
    '''
    Given coordinates [yc, xc], calculate the boundary indices of a square box centered at this coordinate
    'Right side' of the boundary is exclusive, 'left side' inclusive

    Args:
        xlim: int, the boundary of the original image/array, return indices will not exceed this boundary
        ylim: int, the boundary of the original image/array, return indices will not exceed this boundary

    Returns: ymin, ymax, xmin, xmax
    '''

    if not isinstance(yc, (int, np.integer)) or not isinstance(xc, (int, np.integer)):
        if warn:
            warnings.warn('in find_ind_bounds(): given coordinates (yc, xc) = ({}, {}) are not integers, will be rounded '
                          'to nearest int'.format(yc, xc))
        xc = np.rint(xc).astype(int)
        yc = np.rint(yc).astype(int)

    if boxsize < 1:
        raise ValueError('boxsize must be >= 1')

    if xc >= xlim or yc >= ylim or xc < 0 or yc < 0:
        raise ValueError('input indices exceed original array size')

    ymin = max(yc - boxsize // 2, 0)
    xmin = max(xc - boxsize // 2, 0)
    ymax = min(yc + boxsize // 2 + 1, ylim)
    xmax = min(xc + boxsize // 2 + 1, xlim)
    if boxsize % 2 == 0:
        # for even fitbox sizes, need to truncate ymax/xmax by 1
        ymax -= 1
        xmax -= 1

    return int(ymin), int(ymax), int(xmin), int(xmax)

def params_are_physical(p, boxsize):
    '''
    check physical constraints for the parameters, range for theta can be arbitrary as long as it spans
    a range of 2*pi

    Args:
        p: non linear model parameters
           [x0, y0, varx, vary, theta] if model is gaussian
           [x0, y0, fwhmx, fwhmy, theta, beta] if model is moffat
        boxsize: box within which x0, y0 must lie

    Returns:
        True if parameters are physical, False otherwise
    '''

    p = np.array(p)
    if np.any(p[:4] < 0):
        return False
    if np.any(p[:2] >= boxsize):
        return False
    if len(p) > 4 and (p[4] < -np.pi or p[4] > np.pi):
        return False
    if len(p) > 5 and p[5] < 0:
        return False

    return True

def generate_elliptical_gaussian_model(params, boxsize):
    '''
    Args:
        params: model parameters [x0, y0, varx, vary, theta], theta is the counter-clockwise rotation angle in radians
        boxsize:

    Returns:
        model image
    '''

    x = np.arange(boxsize)
    x, y = np.meshgrid(x, x)
    x0, y0 = (params[0], params[1])
    varx = params[2]
    vary = params[3]
    theta = params[4]
    a = np.cos(theta) ** 2 / (2 * varx) + np.sin(theta) ** 2 / (2 * vary)
    b = np.sin(2 * theta) / (4 * varx) - np.sin(2 * theta) / (4 * vary)
    c = np.sin(theta) ** 2 / (2 * varx) + np.cos(theta) ** 2 / (2 * vary)
    Delta = np.array([x - x0, y - y0])
    psf = np.exp(- (a * Delta[0] ** 2 + 2 * b * Delta[0] * Delta[1] + c * Delta[1] ** 2))

    return psf

def generate_elliptical_moffat_model(params, boxsize):
    '''
    Args:
        params: model parameters [x0, y0, fwhmx, fwhmy, theta, beta]
                theta is the counter-clockwise rotation angle in radians
        boxsize:

    Returns:
        model image
    '''

    x = np.arange(boxsize)
    x, y = np.meshgrid(x, x)
    x0, y0 = (params[0], params[1])
    beta = params[5]
    varx = (params[2] / (2. * np.sqrt(np.power(2., 1. / beta) - 1.))) ** 2
    vary = (params[3] / (2. * np.sqrt(np.power(2., 1. / beta) - 1.))) ** 2
    theta = params[4]
    a = np.cos(theta) ** 2 / varx + np.sin(theta) ** 2 / vary
    b = np.sin(2 * theta) / (2 * varx) - np.sin(2 * theta) / (2 * vary)
    c = np.sin(theta) ** 2 / varx + np.cos(theta) ** 2 / vary
    Delta = np.array([x - x0, y - y0])
    psf = np.power((1 + a * Delta[0] ** 2 + 2 * b * Delta[0] * Delta[1] + c * Delta[1] ** 2), -beta)

    return psf

def generate_airydisk_model(params, boxsize):
    '''
    Args:
        params: model parameters [x0, y0, fwhm]
        boxsize:

    Returns:
        model image
    '''

    x = np.arange(boxsize)
    x, y = np.meshgrid(x, x)
    x0, y0 = (params[0], params[1])
    fwhm = params[2]

    airy_psf = modeling.functional_models.AiryDisk2D()
    psf = airy_psf.evaluate(x, y, 1., x0, y0, fwhm/2.)

    return psf

def chi2_marginalized(p, fitobj, model='moffat', fitbg=True):

    lnlike = lnlike_marginalized(p, fitobj, model=model, fitbg=fitbg)

    return -2. * lnlike

def lnlike_marginalized(p, fitobj, model='moffat', fitbg=True, full_result=False):
    '''
    Args:
        p: non linear model parameters
           p=[x0, y0, varx, vary, theta] if model is gaussian
           p=[x0, y0, fwhmx, fwhmy, theta, beta] if model is moffat
        fitobj: a data class with the data_stamp for fitting and attributes for storing results
        model: string, type of analytical model used for fitting, accepts 'moffat', 'gaussian'
        fitbg: whether to fit for a constant background
        full_result: returns best fit linear parameters and best fit model in addition to log likelihood
                     this is needed when optimization/mcmc has converged and we need to retrieve the best fit results

    Returns:
        log likelihood marginalized over flux and background
    '''

    params = np.copy(np.array(p))
    x_offset = (fitobj.data_stamp_x_center - fitobj.fitboxsize // 2)
    y_offset = (fitobj.data_stamp_y_center - fitobj.fitboxsize // 2)
    params[0] -= x_offset
    params[1] -= y_offset

    if not params_are_physical(params, fitobj.fitboxsize):
        return -np.inf

    boxsize = fitobj.data_stamp.shape[0]  # data stamp is guaranteed to be square by the class method
    if model.lower() == 'gaussian':
        psf = generate_elliptical_gaussian_model(params, boxsize)
    elif model.lower() == 'moffat':
        psf = generate_elliptical_moffat_model(params, boxsize)
    elif model.lower() == 'airy':
        psf = generate_airydisk_model(params, boxsize)

    y = np.copy(fitobj.data_stamp.flatten())
    ierr = np.sqrt(fitobj.ivar_stamp.flatten())
    psf_good = np.copy(psf.flatten())

    # ignore bad pixels from the flattened array
    good_ind = np.isfinite(y)
    y = y[good_ind]
    ierr = ierr[good_ind]
    psf_good = psf_good[good_ind]
    npix = len(y)

    # establish the linear part of the model Ax = y
    if fitbg:
        # where A is the model matrix, x = [flux, B] and y is the data
        A = np.ones((npix, 2))
    else:
        A = np.ones((npix, 1))
    A[:, 0] = psf_good
    Ap = A * ierr[:, np.newaxis]
    yp = y * ierr
    try:
        x = np.linalg.lstsq(Ap, yp, rcond=None)[0]
    except:
        warnings.warn('linear algebra failed')
        return -np.inf
    if x[0] < 0:
        return -np.inf

    # calculate Cp, the covariance matrix of the linear model parameters
    Cp_inv = np.dot(Ap.T, Ap)
    det_Cp_inv = np.linalg.det(Cp_inv)
    if det_Cp_inv <= 0:
        warnings.warn('singular covariance matrix')
    # calculate chi squared and marginalize over f1, f2, and B by dividing by det(Cp_inv)
    chi2 = np.nansum((y - np.dot(A, x)) ** 2 * ierr ** 2)
    lnlike = -0.5 * chi2 - np.log(np.sqrt(det_Cp_inv))
    if full_result:
        try:
            return lnlike, x, psf
        except:
            return lnlike, 'NA'

    return lnlike

def non_lin_opt_fit(image, x0, y0, fwhmx, fwhmy, theta=0., beta=2., ivar=None, model='moffat', fitboxsize=15,
                    fitbg=True, crop_method='input', r_tol=2, opt_method='Nelder-Mead', enforce_chisqr=False):
    '''
    Wrapper to run a non-linear optimization PSF fit

    Args:
        image: 2D numpy array
        x0: scalar, initial guess for x pixel location, integer would suffice
        y0: scalar, initial guess for y pixel location, integer would suffice
        fwhmx: scalar, fwhm along x-axis (before rotation)
        fwhmy: scalar, fwhm along y-axis (before rotation)
        theta: scalar, angle of rotation in radians, counter-clockwise
        beta: scalar, exponent in the moffat profile formula
        ivar: 2D numpy array, inverse variance frame for the image, optional
        model: string, type of analytical profile to fit, accepts 'moffat' and 'gaussian'
        fitboxsize: integer, side length of the square box used for fitting
        fitbg: bool, whether to fit for a constant background
        crop_method: string, method for cropping the data stamp, see PSFfit.generate_data_stamp() docstring for details
        opt_method: string, method for non-linear optimization, see scipy documentation for details
        enforce_chisqr: bool, whether to re-adjust the inverse variances by a constant factor to enforce the chi sqaure
                              to be of order 1

    Returns:
        fit: an instance of PSFfit class, best fit parameters are saved to its attributes, see PSFfit data class
             for a list of attributes.
    '''

    if model.lower() != 'moffat' and model.lower() != 'gaussian':
        raise ValueError('Analytical model: {} is not supported'.format(model))

    # initialize a PSFfit object to store data stamp and best fit results
    fit = PSFfit(fitboxsize)
    fit.generate_data_stamp(image, ivar=ivar, x0=x0, y0=y0, method=crop_method, r_tol=r_tol)

    # set up initial guesses
    if model.lower() == 'moffat':
        p0 = np.array([x0, y0, fwhmx, fwhmy, theta, beta])
        dof = 7
    elif model.lower() == 'gaussian':
        varx = fwhmx ** 2 / (8. * np.log(2))
        vary = fwhmy ** 2 / (8. * np.log(2))
        p0 = np.array([x0, y0, varx, vary, theta])
        dof = 6

    result = optimize.minimize(chi2_marginalized, p0, (fit, model, fitbg), method=opt_method)

    if enforce_chisqr and result.success:
        if np.isfinite(result.fun):
            chisqr_dof = result.fun / (fit.fitboxsize ** 2 - dof)
            if chisqr_dof > 2. or chisqr_dof < 0.5:
                fit.ivar_stamp /= chisqr_dof
                result = optimize.minimize(chi2_marginalized, p0, (fit, model, fitbg), method=opt_method)

    if np.isfinite(result.fun):
        lnlike, linparam, psf = lnlike_marginalized(result.x, fit, model, fitbg, full_result=True)
        fit.lnlike = lnlike
        fit.fitx = result.x[0]
        fit.fity = result.x[1]
        fit.fitfwhmx = result.x[2]
        fit.fitfwhmy = result.x[3]
        fit.fittheta = result.x[4]
        fit.fitf = linparam[0]
        if model.lower() == 'moffat':
            fit.fitbeta = result.x[5]
        if fitbg:
            fit.fitbg = linparam[1]
        fit.bestfitmodel = psf * fit.fitf + fit.fitbg

    else:
        fit.bestfitmodel = np.zeros((fit.data_stamp.shape))

    return fit, result

def mcmc_fit(image, x0, y0, fwhmx, fwhmy, theta=0., beta=2., ivar=None, model='moffat', fitboxsize=15,
             fitbg=True, crop_method='input', nwalkers=20, nsteps=5000, discard=1000, progress=True):
    '''
    Wrapper to run an mcmc PSF fit

    Args:
        image: 2D numpy array
        x0: scalar, initial guess for x pixel location, integer would suffice
        y0: scalar, initial guess for y pixel location, integer would suffice
        fwhmx: scalar, fwhm along x-axis (before rotation)
        fwhmy: scalar, fwhm along y-axis (before rotation)
        theta: scalar, angle of rotation in radians, counter-clockwise
        beta: scalar, exponent in the moffat profile formula
        ivar: 2D numpy array, inverse variance frame for the image, optional
        model: string, type of analytical profile to fit, accepts 'moffat' and 'gaussian'
        fitboxsize: integer, side length of the square box used for fitting
        fitbg: bool, whether to fit for a constant background
        crop_method: string, method for cropping the data stamp, see PSFfit.generate_data_stamp() docstring for details
        nwalkers: integer, number of walkers
        nsteps: integer, mcmc steps
        discard: integer, burn-in to discard when calculating results
        progress: bool, show progress bar

    Returns:
        fit: an instance of PSFfit class, best fit parameters are saved to its attributes, see PSFfit data class
             for a list of attributes.
        sampler: the emcee sampler
    '''

    if model.lower() != 'moffat' and model.lower() != 'gaussian' and model.lower() != 'airy':
        raise ValueError('Analytical model: {} is not supported'.format(model))

    # initialize a PSFfit object to store data stamp and best fit results
    fit = PSFfit(fitboxsize)
    fit.generate_data_stamp(image, ivar=ivar, x0=x0, y0=y0, method=crop_method)

    # set up initial guesses
    if model.lower() == 'moffat':
        ndim = 6
        p0 = np.array([x0, y0, fwhmx, fwhmy, theta, beta])
    elif model.lower() == 'gaussian':
        ndim = 5
        varx = fwhmx ** 2 / (8. * np.log(2))
        vary = fwhmy ** 2 / (8. * np.log(2))
        p0 = np.array([x0, y0, varx, vary, theta])
    elif model.lower() == 'airy':
        if fwhmx != fwhmy:
            raise ValueError('airy models must be azimuthually symmetric, fwhmx and fwhmy must be the same')

        ndim = 3
        p0 = np.array([x0, y0, fwhmx])

    # create initial guesses for all walkers
    p0 = np.tile(p0, (nwalkers, 1))
    random_coeff = np.array([0.1, 0.1, 0.1, 0.1, 0.01, 0.1])
    random_coeff = np.tile(random_coeff[:ndim], (nwalkers, 1))
    p0 = np.random.randn(nwalkers, ndim) * random_coeff + p0

    # run mcmc
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnlike_marginalized, args=[fit, model, fitbg])
    sampler.run_mcmc(p0, nsteps, progress=progress)

    # save results
    flat_samples = sampler.get_chain(discard=discard, flat=True)
    mcmc = np.percentile(flat_samples, [16, 50, 84], axis=0).T
    errorbars = np.diff(mcmc)

    fit.fitx = mcmc[0, 1]
    fit.fitxerr = 0.5 * (np.abs(errorbars[0, 0]) + np.abs(errorbars[0, 1]))
    fit.fity = mcmc[1, 1]
    fit.fityerr = 0.5 * (np.abs(errorbars[1, 0]) + np.abs(errorbars[1, 1]))
    fit.fitfwhmx = mcmc[2, 1]
    fit.fitfwhmxerr = 0.5 * (np.abs(errorbars[2, 0]) + np.abs(errorbars[2, 1]))
    if model.lower() == 'airy':
        fit.fitfwhmy = fit.fitfwhmx
        fit.fitfwhmyerr = fit.fitfwhmxerr
    else:
        fit.fitfwhmy = mcmc[3, 1]
        fit.fitfwhmyerr = 0.5 * (np.abs(errorbars[3, 0]) + np.abs(errorbars[3, 1]))
        fit.fittheta = mcmc[4, 1]
        fit.fitthetaerr = 0.5 * (np.abs(errorbars[4, 0]) + np.abs(errorbars[4, 1]))
    if model.lower() == 'moffat':
        fit.fitbeta = mcmc[5, 1]
        fit.fitbetaerr = 0.5 * (np.abs(errorbars[5, 0]) + np.abs(errorbars[5, 1]))

    try:
        lnlike, linparam, psf = lnlike_marginalized(mcmc[:, 1], fit, model, fitbg, full_result=True)
    except:
        raise ValueError('mcmc failed, please plot chains to inspect manually')
    fit.lnlike = lnlike
    fit.fitf = linparam[0]
    if fitbg:
        fit.fitbg = linparam[1]
    fit.bestfitmodel = psf * fit.fitf + fit.fitbg

    return fit, sampler