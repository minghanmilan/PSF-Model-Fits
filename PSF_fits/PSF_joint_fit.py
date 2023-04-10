#!/usr/bin/env python

################################################################
# A program to jointly fit two blended PSFs with high accuracy #
################################################################

import numpy as np
import warnings
import sys
import emcee
from scipy import optimize
import PSF_model_fit as PSFfit

def _model_params_are_physical(p, boxsize):
    '''
    check physical constraints for the parameters, range for theta can be arbitrary as long as it spans
    a range of 2*pi

    Args:
        p: non linear model parameters
           [x1, y1, x2, y2, varx, vary, theta] if model is gaussian
           [x1, y1, x2, y2, fwhmx, fwhmy, theta, beta] if model is moffat
        boxsize: box within which x1, y1, x2, y2 must lie

    Returns:
        True if parameters are physical, False otherwise
    '''

    p = np.array(p)
    if np.any(p[:6] < 0):
        return False
    if np.any(p[:4] >= boxsize):
        return False
    if p[6] <= -np.pi or p[6] > np.pi:
        # theta chosen to have range (-pi, pi]
        return False
    if len(p) > 7 and p[7] < 0:
        return False

    return True

def _chisqr_marginalized(p, fitobj, twofitboxes=False, model='moffat', fitbg=False):

    lnlike = _lnlike_marginalized(p, fitobj, twofitboxes=twofitboxes, model=model, fitbg=fitbg)

    return -2. * lnlike

def _lnlike_marginalized(p, fitobj, twofitboxes=False, model='moffat', fitbg=False, full_result=False):
    '''
    Args:
        p: non linear model parameters
           p=[x1, y1, x2, y2, varx, vary, theta] if model is gaussian
           p=[x1, y1, x2, y2, fwhmx, fwhmy, theta, beta] if model is moffat
        fitobj: an instanace of PSFfit

    Returns:
        marginalized log likelihood
    '''

    p = np.array(p)

    boxsize = fitobj.fitboxsize
    if twofitboxes:
        p_boxes = np.copy(p) # shift positions of sources into their own corresponding fitbox frames
        # crop separate fit boxes for each source and adjust the initial guess positions for the offset
        data_stamp1 = fitobj.data_stamp[0]
        ivar_stamp1 = fitobj.ivar_stamp[0]
        p_box1 = np.copy(p) # shift position of sources into the frame of the first fitbox
        x_offset = fitobj.data_stamp_x_center[0] - boxsize // 2
        y_offset = fitobj.data_stamp_y_center[0] - boxsize // 2
        p_box1[:4] -= np.array([x_offset, y_offset, x_offset, y_offset])
        p_boxes[:2] -= np.array([x_offset, y_offset])
        data_stamp2 = fitobj.data_stamp[1]
        ivar_stamp2 = fitobj.ivar_stamp[1]
        p_box2 = np.copy(p) # shift position of sources into the frame of the first fitbox
        x_offset = fitobj.data_stamp_x_center[1] - boxsize // 2
        y_offset = fitobj.data_stamp_y_center[1] - boxsize // 2
        p_box2[:4] -= np.array([x_offset, y_offset, x_offset, y_offset])
        p_boxes[2:4] -= np.array([x_offset, y_offset])
        if not _model_params_are_physical(p_boxes, boxsize):
            return -np.inf
        if model.lower() == 'gaussian':
            psf1_box1 = PSFfit.generate_elliptical_gaussian_model(p_box1[[0, 1, 4, 5, 6]], boxsize)
            psf2_box1 = PSFfit.generate_elliptical_gaussian_model(p_box1[2:7], boxsize)
            psf1_box2 = PSFfit.generate_elliptical_gaussian_model(p_box2[[0, 1, 4, 5, 6]], boxsize)
            psf2_box2 = PSFfit.generate_elliptical_gaussian_model(p_box2[2:7], boxsize)
        elif model.lower() == 'moffat':
            psf1_box1 = PSFfit.generate_elliptical_moffat_model(p_box1[[0, 1, 4, 5, 6, 7]], boxsize)
            psf2_box1 = PSFfit.generate_elliptical_moffat_model(p_box1[2:8], boxsize)
            psf1_box2 = PSFfit.generate_elliptical_moffat_model(p_box2[[0, 1, 4, 5, 6, 7]], boxsize)
            psf2_box2 = PSFfit.generate_elliptical_moffat_model(p_box2[2:8], boxsize)
        # flatten and concatenate data and model
        # establish the linear part of the model Ax = y, where A is the model matrix, x the linear parameters matrix
        # and y is the data
        y = np.concatenate((data_stamp1.flatten(), data_stamp2.flatten()))
        ierr = np.sqrt(np.concatenate((ivar_stamp1.flatten(), ivar_stamp2.flatten())))
        psf1_good = np.concatenate((psf1_box1.flatten(), psf1_box2.flatten()))
        psf2_good = np.concatenate((psf2_box1.flatten(), psf2_box2.flatten()))
    else:
        params = np.copy(p)
        x_offset = fitobj.data_stamp_x_center - boxsize // 2
        y_offset = fitobj.data_stamp_y_center - boxsize // 2
        params[[0, 2]] -= x_offset
        params[[1, 3]] -= y_offset
        if model.lower() == 'gaussian':
            psf1 = PSFfit.generate_elliptical_gaussian_model(params[[0, 1, 4, 5, 6]], boxsize)
            psf2 = PSFfit.generate_elliptical_gaussian_model(params[2:7], boxsize)
        elif model.lower() == 'moffat':
            psf1 = PSFfit.generate_elliptical_moffat_model(params[[0, 1, 4, 5, 6, 7]], boxsize)
            psf2 = PSFfit.generate_elliptical_moffat_model(params[2:8], boxsize)
        # flatten data and model
        # establish the linear part of the model Ax = y, where A is the model matrix, x the linear parameters matrix
        # and y is the data
        y = fitobj.data_stamp.flatten()
        ierr = np.sqrt(fitobj.ivar_stamp.flatten())
        psf1_good = psf1.flatten()
        psf2_good = psf2.flatten()

    # ignore bad pixels from the flattened array
    good_ind = np.isfinite(y)
    y = y[good_ind]
    ierr = ierr[good_ind]
    psf1_good = psf1_good[good_ind]
    psf2_good = psf2_good[good_ind]

    npix = len(y) # length of flatten data array with bad pixels excluded
    if fitbg:
        # in this case x is [f1, f2, B]
        A = np.ones((npix, 3))
    else:
        # in this case x is [f1, f2]
        A = np.ones((npix, 2))
    A[:, 0] = psf1_good
    A[:, 1] = psf2_good
    Ap = A * ierr[:, np.newaxis]
    yp = y * ierr
    try:
        x = np.linalg.lstsq(Ap, yp, rcond=None)[0]
    except:
        warnings.warn('linear algebra failed')
        return -np.inf

    if np.any(x[:2] < 0):
        return -np.inf

    # calculate Cp, the covariance matrix of the model parameters
    Cp_inv = np.dot(Ap.T, Ap)
    det_Cp_inv = np.linalg.det(Cp_inv)
    if det_Cp_inv <= 0:
        warnings.warn('singular covariance matrix')
    # calculate chi squared and marginalize over linear paramters by dividing by det(Cp_inv)
    chi2 = np.nansum((y - np.dot(A, x)) ** 2 * ierr ** 2)
    lnlike = -0.5 * chi2 - np.log(np.sqrt(det_Cp_inv))

    if full_result:
        U, W, VT = np.linalg.svd(Ap)
        Winv = 1. / W
        Winv[W < np.amax(W) * 1e-14] = 0
        V = VT.T
        Cmodel = np.sum(V[np.newaxis, :, :] * V[:, np.newaxis, :] * Winv ** 2, axis=-1)

        if twofitboxes:
            return lnlike, x, psf1_box1, psf2_box1, psf1_box2, psf2_box2, Cmodel
        else:
            return lnlike, x, psf1, psf2, Cmodel

    return lnlike

def opt_fit(fit_obj, pri_loc, sec_loc, fwhmx, fwhmy, theta, beta, twofitboxes=False, model='moffat',
            method='Nelder-Mead', fitbg=False, enforce_chisqr=True, silent=True):
    '''
    Run a non-linear optimization PSF fit

    Args:
        fit_obj: an instance for the PSFfit class, used to generate data_stamp from img and stores some fitting results
        pri_loc: initial guess for the pixel location of the primary star in [x, y] format.
                 If twofitboxes is True, this will be the [x, y] pixel location in the frame of the original image.
                 If twofitboxes is False, this will be the [x, y] pixel location in the frame of the data_stamp.
        sec_loc: initial guess for the pixel location of the secondary star in [x, y] format
        fwhmx: scalar, fwhm along x-axis (before rotation)
        fwhmy: scalar, fwhm along y-axis (before rotation)
        theta: scalar, angle of rotation in radians, counter-clockwise
        beta: scalar, exponent in the moffat profile formula
        twofitboxes: bool, if False, there is only one data_stamp and one fitbox, which should contain both sources to
                                     be fitted jointly. (suitable for close separations)
                           if True, there should be two data stamps and two fitboxes available in fit_obj, each covering
                                    the core of one of the sources, where these two fitboxes will be fitted jointly.
                                    (suitable for large separations)
        model: string, type of analytical profile to fit, accepts 'moffat' and 'gaussian'
        method: scipy optimization method
        enforce_chisqr: bool, whether to re-adjust the inverse variances by a constant factor to enforce the chi sqaure
                              to be of order 1
        silent:

    Returns:

    '''

    if model.lower() != 'moffat' and model.lower() != 'gaussian':
        raise ValueError('Analytical model: {} is not supported'.format(model))

    # set up initial guesses
    dof = 0
    if model.lower() == 'moffat':
        dof = 10 # 8 parameters in p0 and 2 linear parameters: f1, f2
        p0 = np.array([pri_loc[0], pri_loc[1], sec_loc[0], sec_loc[1], fwhmx, fwhmy, theta, beta])
    elif model.lower() == 'gaussian':
        dof = 9 # 7 parameters in p0 and 2 linear parameters: f1, f2
        varx = fwhmx ** 2 / (8. * np.log(2))
        vary = fwhmy ** 2 / (8. * np.log(2))
        p0 = np.array([pri_loc[0], pri_loc[1], sec_loc[0], sec_loc[1], varx, vary, theta])

    result = optimize.minimize(_chisqr_marginalized, p0, (fit_obj, twofitboxes, model, fitbg), method=method)

    if enforce_chisqr and result.success:
        if np.isfinite(result.fun):
            chisqr_dof = result.fun / (fit_obj.fitboxsize ** 2 - dof)
            if chisqr_dof > 2. or chisqr_dof < 0.5:
                fit_obj.ivar_stamp /= chisqr_dof
                result = optimize.minimize(_chisqr_marginalized, p0, (fit_obj, twofitboxes, model, fitbg),
                                           method=method)

    if not result.success:
        if not silent:
            sys.stdout.write('\n')
            print('optimization did not converge')
        Cmodel = np.zeros((2, 2))
        fit_obj.fitf = np.array([0, 0])
        fit_obj.bestfitmodel = None

    else:
        if twofitboxes:
            lnlike, linparams, psf1_box1, psf2_box1, \
            psf1_box2, psf2_box2, Cmodel = _lnlike_marginalized(result.x, fit_obj, twofitboxes, model, fitbg,
                                                               full_result=True)
            fit_obj.fitx = result.x[[0, 2]]
            fit_obj.fity = result.x[[1, 3]]
            fit_obj.fitfwhmx = result.x[4]
            fit_obj.fitfwhmy = result.x[5]
            fit_obj.fittheta = result.x[6]
            if model.lower() == 'moffat':
                fit_obj.fitbeta = result.x[7]
            fit_obj.fitf = linparams
            fit_obj.bestfitmodel.append(linparams[0] * psf1_box1 + linparams[1] * psf2_box1)
            fit_obj.bestfitmodel.append(linparams[0] * psf1_box2 + linparams[1] * psf2_box2)
            fit_obj.bestfitmodel = np.array(fit_obj.bestfitmodel)
        else:
            lnlike, linparams, psf1, psf2, Cmodel = _lnlike_marginalized(result.x, fit_obj, twofitboxes, model, fitbg,
                                                                        full_result=True)
            fit_obj.fitx = result.x[[0, 2]]
            fit_obj.fity = result.x[[1, 3]]
            fit_obj.fitfwhmx = result.x[4]
            fit_obj.fitfwhmy = result.x[5]
            fit_obj.fittheta = result.x[6]
            if model.lower() == 'moffat':
                fit_obj.fitbeta = result.x[7]
            fit_obj.fitf = linparams
            fit_obj.bestfitmodel = linparams[0] * psf1 + linparams[1] * psf2

    return fit_obj, Cmodel

def mcmc_fit(fit_obj, pri_loc, sec_loc, fwhmx, fwhmy, theta, beta, twofitboxes=False, model='moffat',
             enforce_chisqr=False, silent=True):

    return