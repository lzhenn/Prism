#/usr/bin/env python
"""Preprocessing the WRF input file"""

import datetime
import numpy as np
import xarray as xr
import pandas as pd
import gc
import netCDF4 as nc4
import wrf  
from copy import copy
from scipy import interpolate
import sys
sys.path.append('../')
from utils import utils

print_prefix='lib.preprocess_wrfinp>>'

class wrf_mesh:

    '''
    Construct grid info and UVW mesh template
    
    Attributes
    -----------
    dx:         float, scalar
        discritized spacing in x-direction (m)

    dy:         float, scalar
        discritized spacing in y-direction (m)

    U:          float, 3d
        zonal wind (m/s)

    V:          float, 3d
        meridional wind (m/s)

    W:          float, 3d
        vertical wind (m/s)

    ter:        float, 2d
        terrain height

    z:          float, 1d, bottom_top
        model layer height above terrain

    ztop:       float, scalar
        model top layer elevation above sea level

    dnw:        float, 1d, bottom_top
        delta eta values on vertical velocity levels 

    geo_z_idx:  int, scalar
        z index where geostraphic wind prevails (init height of free atm)

    near_surf_z_idx: int, scalar
        z index of init height of Ekman layer

    Methods
    '''
    
    def __init__(self, cfg):
        """ construct input wrf file names """
        
        utils.write_log(print_prefix+'Init wrf_mesh obj...')
        utils.write_log(print_prefix+'Read input files...')
        
        # collect global attr
        self.dateseries=pd.date_range(start=cfg['TRAINING']['training_start'], end=cfg['TRAINING']['training_end'])
        self.ncfiles=[] 
        for datestamp in self.dateseries:
            
            yyyy=datestamp.strftime('%Y')
            mm=datestamp.strftime('%m')
            dd=datestamp.strftime('%d')
            nc_fn='./input/training/wrfout_d01_'+yyyy+'-'+mm+'-'+dd+'_12:00:00'
            utils.write_log(print_prefix+'Read '+nc_fn)
            self.ncfiles.append(nc4.Dataset(nc_fn))
        
        self.slp = wrf.getvar(self.ncfiles, 'slp', timeidx=wrf.ALL_TIMES, method='cat')
        # lats lons on mass and staggered grids
        self.XLAT=wrf.getvar(self.ncfiles[0],'XLAT')
        self.XLONG=wrf.getvar(self.ncfiles[0],'XLONG')

        # shape
        shp=self.slp.shape
        self.nrec=shp[0]
        self.nrow=shp[1]
        self.ncol=shp[2]
if __name__ == "__main__":
    pass
