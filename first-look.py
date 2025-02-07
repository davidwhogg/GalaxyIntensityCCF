'''
This script takes an analysis started in the first-look.ipynb notebook and runs it in parallel
on a NERSC compute node.

The notebook generates FITS tables of the galaxy sample and stuck-fibers to use.  This code
goes to the DESI spectra files, opens them, and then for each nearby galaxy, pulls out a
subsection of the spectrum.

'''

import pylab as plt
import numpy as np
from astrometry.util.fits import *
from collections import Counter
from astrometry.libkd.spherematch import match_radec
import fitsio
from glob import glob
import sys
import os
from astrometry.util.plotutils import *
from astrometry.util.starutil import *
from astrometry.util.multiproc import multiproc
from scipy.ndimage import label, find_objects
from collections import Counter
from tractor import NanoMaggies
from astrometry.util.starutil_numpy import *
from astrometry.util.util import Sip
from tractor.brightness import NanoMaggies
import pickle

# GLOBALS
sample_targetids = None
targetid_map = None
Imatch = None
gals = None


def main():
    global sample_targetids
    global targetid_map
    global Imatch
    global gals

    # spec = fits_table('/pscratch/sd/d/dstn/stuck-sample.fits')
    # gals = fits_table('/pscratch/sd/d/dstn/galaxy-sample.fits')
    # 
    # spec_sub = fits_table()
    # spec_sub.target_ra = spec.target_ra
    # spec_sub.target_dec = spec.target_dec
    # spec_sub.targetid = spec.targetid
    # spec_sub.writeto('/pscratch/sd/d/dstn/stuck-sample-sub.fits')
    # 
    # gals_sub = fits_table()
    # gals_sub.target_ra = gals.target_ra
    # gals_sub.target_dec = gals.target_dec
    # gals_sub.targetid = gals.targetid
    # gals_sub.z = gals.z
    # gals_sub.writeto('/pscratch/sd/d/dstn/galaxy-sample-sub.fits')

    # Read in the samples
    spec = fits_table('/pscratch/sd/d/dstn/stuck-sample-sub.fits')
    gals = fits_table('/pscratch/sd/d/dstn/galaxy-sample-sub.fits')
    sky_sample = spec
    print(len(sky_sample), 'sky fibers', len(gals), 'galaxies')

    # Create the list of (all) the DESI spectro data products
    rr_bright_fns = glob('/global/cfs/cdirs/desi/spectro/redux/loa/healpix/main/bright/*/*/redrock-main-bright-*.fits')
    rr_bright_fns.sort()
    rr_dark_fns = [fn.replace('bright', 'dark') for fn in rr_bright_fns]
    co_dark_fns = [fn.replace('/redrock-', '/coadd-') for fn in rr_dark_fns]
    fns = co_dark_fns
    print(len(fns), 'files')

    # Cross-match the samples!
    Imatch = match_radec(sky_sample.target_ra, sky_sample.target_dec, gals.target_ra, gals.target_dec, 2./60., indexlist=True)
    print(sum([len(ii) for ii in Imatch if ii is not None]), 'matches')

    # These are the targetids we want to read (and a mapping back to the index in the sky_sample)
    targetid_map = dict([(t,i) for i,t in enumerate(sky_sample.targetid) if Imatch[i] is not None])
    sample_targetids = list(targetid_map.keys())

    # Read & process spectro files in parallel
    mp = multiproc(128)
    R = mp.map(one_file, fns)
    # Unpack the results into this list-of-lists
    rr = [[] for i in range(18)]
    for res in R:
        if res is None:
            continue
        for ri,resi in zip(rr, res):
            ri.extend(resi)
    # Name the unpacked list-of-lists elements!
    (o3_fluxes, o3_ivars, o3_dists, o3_gal_targetids, o3_stuck_targetids,
     o2_fluxes, o2_ivars, o2_dists, o2_gal_targetids, o2_stuck_targetids,
     ha_fluxes, ha_ivars, ha_dists, ha_gal_targetids, ha_stuck_targetids,
     b_wave, r_wave, z_wave) = rr

    # Wavelength grids are all the same
    b_wave = b_wave[0]
    r_wave = r_wave[0]
    z_wave = z_wave[0]

    sky_o3_fluxes = np.vstack(o3_fluxes)
    sky_o3_ivars  = np.vstack(o3_ivars)
    sky_o3_dists  = np.hstack(o3_dists)
    sky_o3_gal_targetids = np.hstack(o3_gal_targetids)
    sky_o3_stuck_targetids = np.hstack(o3_stuck_targetids)

    print('o3: flux', sky_o3_fluxes.shape)
    print('o3: dists', sky_o3_dists.shape)
    
    sky_o2_fluxes = np.vstack(o2_fluxes)
    sky_o2_ivars  = np.vstack(o2_ivars)
    sky_o2_dists  = np.hstack(o2_dists)
    sky_o2_gal_targetids = np.hstack(o2_gal_targetids)
    sky_o2_stuck_targetids = np.hstack(o2_stuck_targetids)
    
    sky_ha_fluxes = np.vstack(ha_fluxes)
    sky_ha_ivars  = np.vstack(ha_ivars)
    sky_ha_dists  = np.hstack(ha_dists)
    sky_ha_gal_targetids = np.hstack(ha_gal_targetids)
    sky_ha_stuck_targetids = np.hstack(ha_stuck_targetids)
    
    f = open('/pscratch/sd/d/dstn/lines.pickle', 'wb')
    pickle.dump(dict(
        sky_o3_fluxes=sky_o3_fluxes,
        sky_o3_ivars=sky_o3_ivars,
        sky_o3_dists=sky_o3_dists,
        sky_o3_gal_targetids=sky_o3_gal_targetids,
        sky_o3_stuck_targetids=sky_o3_stuck_targetids,
    
        sky_o2_fluxes=sky_o2_fluxes,
        sky_o2_ivars=sky_o2_ivars,
        sky_o2_dists=sky_o2_dists,
        sky_o2_gal_targetids=sky_o2_gal_targetids,
        sky_o2_stuck_targetids=sky_o2_stuck_targetids,
    
        sky_ha_fluxes=sky_ha_fluxes,
        sky_ha_ivars=sky_ha_ivars,
        sky_ha_dists=sky_ha_dists,
        sky_ha_gal_targetids=sky_ha_gal_targetids,
        sky_ha_stuck_targetids=sky_ha_stuck_targetids,
    
        b_wave = b_wave,
        r_wave = r_wave,
        z_wave = z_wave,
    ), f)
    f.close()


# For multiprocessing: process one DESI spectro file, pulling out our targetids of interest
# and extracting the relevant portion of the spectrum.
def one_file(fn):
    o3_fluxes = []
    o3_ivars = []
    o3_dists = []
    o3_gal_targetids = []
    o3_stuck_targetids = []

    o2_fluxes = []
    o2_ivars = []
    o2_dists = []
    o2_gal_targetids = []
    o2_stuck_targetids = []

    ha_fluxes = []
    ha_ivars = []
    ha_dists = []
    ha_gal_targetids = []
    ha_stuck_targetids = []

    if not os.path.exists(fn):
        return None
    fibermap = fits_table(fn)
    I = np.flatnonzero(np.isin(fibermap.targetid, sample_targetids))
    if len(I) == 0:
        return None
    F = fitsio.FITS(fn)
    targetids = fibermap.targetid[I]
    ras = fibermap.target_ra[I]
    decs = fibermap.target_dec[I]
    b_wave = F['B_WAVELENGTH'].read()
    b_flux = F['B_FLUX'].read()[I, :]
    b_ivar = F['B_IVAR'].read()[I, :]
    r_wave = F['R_WAVELENGTH'].read()
    r_flux = F['R_FLUX'].read()[I, :]
    r_ivar = F['R_IVAR'].read()[I, :]
    z_wave = F['Z_WAVELENGTH'].read()
    z_flux = F['Z_FLUX'].read()[I, :]
    z_ivar = F['Z_IVAR'].read()[I, :]
    for ispec,(targetid,ra,dec) in enumerate(zip(targetids, ras, decs)):
        isample = targetid_map[targetid]
        for igal in Imatch[isample]:
            z = gals.z[igal]
            # Book-keeping check: sky_sample & galaxy match; this fibermap row matches
            #print('Fibermap:', ra, 'Sky_sample:', sky_sample.target_ra[isample], 'gal:', gals.target_ra[igal])
            d = degrees_between(ra, dec, gals.target_ra[igal], gals.target_dec[igal])

            # [OII]
            target_restframe = 3726.
            target_wave = target_restframe * (1. + z)
            iwave = np.argmin(np.abs(b_wave - target_wave))
            S = 40
            #assert((iwave >= S) * (iwave < len(b_wave)-S))
            if (iwave >= S) and (iwave < len(b_wave)-S):
                f   = b_flux[ispec, iwave-S : iwave+S+1]
                fiv = b_ivar[ispec, iwave-S : iwave+S+1]
                o2_fluxes.append(f)
                o2_ivars.append(fiv)
                o2_dists.append(d)
                o2_gal_targetids.append(gals.targetid[igal])
                o2_stuck_targetids.append(targetid)

            # [OIII]
            target_restframe = 5007.
            target_wave = target_restframe * (1. + z)
            iwave = np.argmin(np.abs(r_wave - target_wave))
            S = 40
            #assert((iwave >= S) * (iwave < len(r_wave)-S))
            if (iwave >= S) and (iwave < len(r_wave)-S):
                f   = r_flux[ispec, iwave-S : iwave+S+1]
                fiv = r_ivar[ispec, iwave-S : iwave+S+1]
                o3_fluxes.append(f)
                o3_ivars.append(fiv)
                o3_dists.append(d)
                o3_gal_targetids.append(gals.targetid[igal])
                o3_stuck_targetids.append(targetid)

            # H-alpha
            target_restframe = 6562.
            target_wave = target_restframe * (1. + z)
            iwave = np.argmin(np.abs(z_wave - target_wave))
            S = 60
            #assert((iwave >= S) * (iwave < len(z_wave)-S))
            if (iwave >= S) and (iwave < len(r_wave)-S):
                f   = z_flux[ispec, iwave-S : iwave+S+1]
                fiv = z_ivar[ispec, iwave-S : iwave+S+1]
                ha_fluxes.append(f)
                ha_ivars.append(fiv)
                ha_dists.append(d)
                ha_gal_targetids.append(gals.targetid[igal])
                ha_stuck_targetids.append(targetid)

    print('.', end='')
    sys.stdout.flush()
    return (o3_fluxes, o3_ivars, o3_dists, o3_gal_targetids, o3_stuck_targetids,
            o2_fluxes, o2_ivars, o2_dists, o2_gal_targetids, o2_stuck_targetids,
            ha_fluxes, ha_ivars, ha_dists, ha_gal_targetids, ha_stuck_targetids,
            [b_wave], [r_wave], [z_wave])
    
if __name__ == '__main__':
    main()
