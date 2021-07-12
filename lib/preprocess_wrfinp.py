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

import lib
from utils import utils

print_prefix='lib.preprocess_wrfinp>>'

class WrfMesh:

    '''
    Construct grid info and UVW mesh template
    
    Attributes
    -----------
    
    Methods
    -----------
    
    '''
    
    def __init__(self, cfg, call_from='training'):
        """ construct input wrf file names """
        
        utils.write_log(print_prefix+'Init wrf_mesh obj...')
        utils.write_log(print_prefix+'Read input files...')
        
        varlist=lib.cfgparser.cfg_get_varlist(cfg,'SHARE','var')
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
            if cfg['SHARE'].getboolean('debug_mode'):
                utils.write_log(print_prefix+'Debug mode turns on!')
                end_basename=fn_list[2].split('/')[3]
            else:
                end_basename=fn_list[-1].split('/')[3]
            timestamp_start=datetime.datetime.strptime(start_basename[11:],'%Y-%m-%d_%H:%M:%S')
            timestamp_end=datetime.datetime.strptime(end_basename[11:],'%Y-%m-%d_%H:%M:%S')
            self.dateseries=pd.date_range(start=timestamp_start, end=timestamp_end, freq='H')
        
        da_dic={}
        for idx, datestamp in enumerate(self.dateseries):
            nc_fn=nc_fn_base+'wrfout_d01_'+datestamp.strftime('%Y-%m-%d_%H:%M:%S')
            utils.write_log(print_prefix+'Read '+nc_fn)
            
            ncfile=nc4.Dataset(nc_fn)
            if 'h500' in varlist or 'h200' in varlist:
                z=wrf.getvar(ncfile,'z')
                pres=wrf.getvar(ncfile,'pressure')


            if idx ==0:
                for var in varlist:
                    if var == 'h500':
                        da_dic[var]=wrf.interplevel(z, pres, 500).interpolate_na(dim='south_north',fill_value='extrapolate')
                    elif var == 'h200':
                        da_dic[var]=wrf.interplevel(z, pres, 200).interpolate_na(dim='south_north',fill_value='extrapolate')
                    else:
                        da_dic[var]=wrf.getvar(ncfile, var)

                # lats lons on mass and staggered grids
                self.xlat=wrf.getvar(ncfile,'XLAT')
                self.xlong=wrf.getvar(ncfile,'XLONG')
            else:
                for var in varlist:
                    if var == 'h500':
                        da_dic[var]=xr.concat([da_dic[var],wrf.interplevel(z, pres, 500).interpolate_na(dim='south_north',fill_value='extrapolate')],dim='time')
                    elif var == 'h200':
                        da_dic[var]=xr.concat([da_dic[var],wrf.interplevel(z, pres, 200).interpolate_na(dim='south_north',fill_value='extrapolate')],dim='time')
                    else:
                        da_dic[var]=xr.concat([da_dic[var],wrf.getvar(ncfile, var)],dim='time')
            ncfile.close()
        self.data_dic = da_dic 
        self.varlist=varlist
        # shape
        shp=self.data_dic[varlist[0]].shape
        self.nrec=shp[0]
        self.nrow=shp[1]
        self.ncol=shp[2]

def get_varlist(cfg):
    varlist=cfg['SHARE']['var'].split(',')
    varlist=[ele.strip() for ele in varlist]
    return varlist


if __name__ == "__main__":
    pass
