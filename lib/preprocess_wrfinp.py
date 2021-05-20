#/usr/bin/env python
"""Preprocessing the WRF input file"""

import datetime
import numpy as np
import xarray as xr
import pandas as pd
import netCDF4 as nc4
import wrf  
from copy import copy
import sys, os, subprocess

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
    
    def __init__(self, cfg, call_from='training'):
        """ construct input wrf file names """
        
        utils.write_log(print_prefix+'Init wrf_mesh obj...')
        utils.write_log(print_prefix+'Read input files...')
        
        # collect global attr
        nc_fn_base='./input/'+call_from+'/'
        if call_from=='training':
            timestamp_start=datetime.datetime.strptime(cfg['TRAINING']['training_start']+'12','%Y%m%d%H')
            timestamp_end=datetime.datetime.strptime(cfg['TRAINING']['training_end']+'12','%Y%m%d%H')
            self.dateseries=pd.date_range(start=timestamp_start, end=timestamp_end, freq='D')
        elif call_from=='inference':
            fn_stream=subprocess.check_output('ls '+nc_fn_base+'wrfout*', shell=True).decode('utf-8')
            fn_list=fn_stream.split()
            start_basename=fn_list[0].split('/')[3]
            end_basename=fn_list[-1].split('/')[3]
            timestamp_start=datetime.datetime.strptime(start_basename[11:],'%Y-%m-%d_%H:%M:%S')
            timestamp_end=datetime.datetime.strptime(end_basename[11:],'%Y-%m-%d_%H:%M:%S')
            self.dateseries=pd.date_range(start=timestamp_start, end=timestamp_end, freq='H')
        
        for idx, datestamp in enumerate(self.dateseries):
            nc_fn=nc_fn_base+'wrfout_d01_'+datestamp.strftime('%Y-%m-%d_%H:%M:%S')
            utils.write_log(print_prefix+'Read '+nc_fn)
            #self.ncfiles.append(nc4.Dataset(nc_fn))
            ncfile=nc4.Dataset(nc_fn)
            if idx ==0:
                da=wrf.getvar(ncfile, 'slp')
                # lats lons on mass and staggered grids
                self.xlat=wrf.getvar(ncfile,'XLAT')
                self.xlong=wrf.getvar(ncfile,'XLONG')
            else:
                da=xr.concat([da,wrf.getvar(ncfile, 'slp')],dim='time')
            ncfile.close()
        
        self.data = da 
        # shape
        shp=self.data.shape
        self.nrec=shp[0]
        self.nrow=shp[1]
        self.ncol=shp[2]
if __name__ == "__main__":
    pass
