#/usr/bin/env python
"""Preprocessing the ERA5 input file"""

import datetime
import numpy as np
import xarray as xr
import pandas as pd
import os, subprocess, sys
from multiprocessing import Pool

import lib
from utils import utils

print_prefix='lib.preprocess_era5inp>>'
CWD=sys.path[0]

class ERAMesh:

    '''
    Construct grid info and UVW mesh template
    
    Attributes
    -----------
    
    Methods
    -----------
    
    '''
    
    def __init__(self, cfg, call_from='training'):
        """ construct input wrf file names """
        
        utils.write_log(print_prefix+'Init era5_mesh obj...')
        utils.write_log(print_prefix+'Read input files...')
        
        # collect global attr
        self.era_src=cfg['TRAINING']['era5_src']
        self.ntasks=int(cfg['SHARE']['ntasks'])
        self.varlist=['u10','v10','msl', 'z']
        self.dsmp_interval=int(cfg['SHARE']['dsmp_interval'])

        self.s_sn, self.e_sn = int(cfg['SHARE']['s_sn']),int(cfg['SHARE']['e_sn'])
        self.s_we, self.e_we = int(cfg['SHARE']['s_we']),int(cfg['SHARE']['e_we'])

        if call_from=='training':
            
            timestamp_start=datetime.datetime.strptime(
                    cfg['TRAINING']['training_start']+'00','%Y%m%d%H')
            timestamp_end=datetime.datetime.strptime(
                    cfg['TRAINING']['training_end']+'23','%Y%m%d%H')
            all_dateseries=pd.date_range(
                    start=timestamp_start, end=timestamp_end, freq='6H')

            self.dateseries=self._pick_date_frame(cfg, all_dateseries)
        
        elif call_from=='inference':
            fn_stream=subprocess.check_output(
                    'ls '+self.era_src+'wrfout*', shell=True).decode('utf-8')
            fn_list=fn_stream.split()
            start_basename=fn_list[0].split('/')[-1]
            if cfg['INFERENCE'].getboolean('debug_mode'):
                utils.write_log(print_prefix+'Debug mode turns on!')
                end_basename=fn_list[self.ntasks-1].split('/')[-1]
            else:
                end_basename=fn_list[-1].split('/')[-1]
            timestamp_start=datetime.datetime.strptime(start_basename[11:],'%Y-%m-%d_%H:%M:%S')
            timestamp_end=datetime.datetime.strptime(end_basename[11:],'%Y-%m-%d_%H:%M:%S')
            self.dateseries=pd.date_range(start=timestamp_start, end=timestamp_end, freq='H')
    
        self.load_data()
    
    def _pick_date_frame(self, cfg, all_dates):
        
        ''' 
            pick date series list according to 
            sub month and sub hrs
        '''
        subhr_list=lib.cfgparser.cfg_get_varlist(cfg,'TRAINING','sub_hrs')
        submon_list=lib.cfgparser.cfg_get_varlist(cfg,'TRAINING','sub_mons')

        if subhr_list[0] == '-1':
            subhr_list=[0, 6, 12, 18]
        if submon_list[0]== '-1':
            submon_list=range(1,13)

        sel_dates=all_dates[all_dates.hour.isin([int(subhr) for subhr in subhr_list])]
        sel_dates=sel_dates[sel_dates.month.isin([int(submon) for submon in submon_list])]

        return sel_dates


    def load_data(self):
        ''' load datasets '''
        era_src=self.era_src
        init_ts=self.dateseries[0] 
        
        varlist=self.varlist
        ntasks=self.ntasks
        da_dic={}
       
        # let's do the multiprocessing magic!
        utils.write_log(print_prefix+'Multiprocessing initiated. Master process %s.' % os.getpid())

        # get monthly frq list (align with file convention)
        curr_yyyymm=init_ts.strftime('%Y%m')
        file_yyyymm =[init_ts]
        for itime in self.dateseries:
            if itime.strftime('%Y%m') != curr_yyyymm:
                curr_yyyymm=itime.strftime('%Y%m')
                file_yyyymm.append(itime)
        
        len_file=len(file_yyyymm)
        len_per_task=len_file//ntasks
        results=[]
        
        # start process pool
        process_pool = Pool(processes=ntasks)
        
        # open tasks ID 0 to ntasks-2
        for itsk in range(ntasks-1):  
            
            ifile_yyyymm=file_yyyymm[itsk*len_per_task:(itsk+1)*len_per_task]
            
            result=process_pool.apply_async(
                run_mtsk, 
                args=(itsk, ifile_yyyymm, da_dic, self, ))
            results.append(result)

        # open ID ntasks-1 in case of residual
        ifile_yyyymm=file_yyyymm[(ntasks-1)*len_per_task:]

        result=process_pool.apply_async(
            run_mtsk, 
            args=(ntasks-1, ifile_yyyymm, da_dic, self, ))

        results.append(result)
        utils.write_log(print_prefix+'Waiting for all subprocesses done...')
        
        process_pool.close()
        process_pool.join()
        
        # reorg da_dict

        for idx, res in enumerate(results):
            if idx==0:
                da_dic=res.get()
            else:
                for var in varlist:
                    da_dic[var]=xr.concat(
                        [da_dic[var], res.get()[var]], dim='time') 
        
       
        
        self.data_dic = da_dic 
        self.varlist=varlist
        
        # shape
        temp_var=da_dic[varlist[0]]
        shp=temp_var.shape
        self.nrec=shp[0]
        self.nrow=shp[1]
        self.ncol=shp[2]
        self.lon=temp_var['longitude'].values
        self.lat=temp_var['latitude'].values

def run_mtsk(itsk, file_yyyymm, da_dic, era_hdl):
    """
    multitask read file
    """
    era_src=era_hdl.era_src
    varlist=era_hdl.varlist

    all_ts=era_hdl.dateseries

    len_files=len(file_yyyymm)
    da_dic={}
    
    
    # read the first file in the list
    its_yyyymm=file_yyyymm[0]
    sub_ts=get_sub_ts(all_ts, its_yyyymm)
    
    utils.write_log('%sTASK[%02d]: Read %04d of %04d --- %s' % (
        print_prefix, itsk, 0,(len_files-1), its_yyyymm.strftime('%Y-%m')))
    
    for var in varlist:
        var_temp=get_var_xr(era_src,its_yyyymm,var)
        
        da_dic[var]=var_temp.sel(
                time=sub_ts,
                latitude=slice(era_hdl.e_sn,era_hdl.s_sn),
                longitude=slice(era_hdl.s_we, era_hdl.e_we))
    
    # read the rest files in the list
    for idx, its_yyyymm in enumerate(file_yyyymm[1:]):
        
        utils.write_log('%sTASK[%02d]: Read %04d of %04d --- %s' % (
            print_prefix, itsk, (idx+1), (len_files-1),its_yyyymm.strftime('%Y-%m')))
        
        sub_ts=get_sub_ts(all_ts, its_yyyymm)
        
        for var in varlist:
            var_temp=get_var_xr(era_src,its_yyyymm,var)
            var_temp=var_temp.sel(
                time=sub_ts,
                latitude=slice(era_hdl.e_sn,era_hdl.s_sn),
                longitude=slice(era_hdl.s_we, era_hdl.e_we))
            
            da_dic[var]=xr.concat([da_dic[var],var_temp], dim='time')
    
    utils.write_log('%sTASK[%02d]: All files loaded.' % (print_prefix, itsk))
    
    return da_dic

def get_var_xr(src, ts, var):
    ''' retrun var xr obj according to var name'''
    
    if var=='z':
        nc_fn=src+'/'+ts.strftime('%Y%m')+'-h500.nc'
    else:
        nc_fn=src+'/'+ts.strftime('%Y%m')+'-surf.nc'
    
    da=xr.load_dataset(nc_fn)
    var_xr=da[var]
   
    return var_xr


def get_varlist(cfg):
    ''' seperate vars in cfg varlist csv format '''
    varlist=cfg['SHARE']['var'].split(',')
    varlist=[ele.strip() for ele in varlist]
    return varlist


def get_sub_ts(all_ts, yyyymm):
    ''' get sub ts according to '''
    sub_ts=all_ts[all_ts.year==yyyymm.year]
    sub_ts=sub_ts[sub_ts.month==yyyymm.month]
    return sub_ts

if __name__ == "__main__":
    '''
    Code for unit test
    '''
    utils.write_log('Read Config...')

    
    # init wrf handler and read training data
    era_hdl=lib.preprocess_wrfinp.WrfMesh(cfg_hdl)
 
