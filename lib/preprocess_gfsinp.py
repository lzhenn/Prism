#/usr/bin/env python
"""Preprocessing the GFS input file"""

import datetime
import numpy as np
import xarray as xr
import pandas as pd
import os, subprocess, sys
from multiprocessing import Pool

import lib
from utils import utils

print_prefix='lib.preprocess_gfsinp>>'
CWD=sys.path[0]
G=9.81
class GFSMesh:

    '''
    Construct grid info and UVW mesh template
    
    Attributes
    -----------
    
    Methods
    -----------
    
    '''
    
    def __init__(self, cfg, call_from='training'):
        """ construct input wrf file names """
        utils.write_log(print_prefix+'Init gfs_mesh obj...')
        utils.write_log(print_prefix+'Read input files...')
        
        # ---deal with cfg file        
        # collect global attr
        self.gfs_src=cfg['INFERENCE']['gfs_src']
        self.gfs_frq=cfg['INFERENCE']['gfs_frq']
        self.gfs_days=cfg['INFERENCE']['gfs_days']

        self.ntasks=int(cfg['SHARE']['ntasks'])
        self.varlist=[
                'UGRD_P0_L103_GLL0','VGRD_P0_L103_GLL0',
                'PRMSL_P0_L101_GLL0', 'HGT_P0_L100_GLL0']

        self.dsmp_interval=int(cfg['SHARE']['dsmp_interval'])

        self.s_sn, self.e_sn = int(cfg['SHARE']['s_sn']),int(cfg['SHARE']['e_sn'])
        self.s_we, self.e_we = int(cfg['SHARE']['s_we']),int(cfg['SHARE']['e_we'])
        
        # ---read timestamp file
        with open(self.gfs_src+'init_time','r') as f:
            self.fc_init_ts=datetime.datetime.strptime(f.readline(),'%Y%m%d%H')

        # ---loop fcst files
        fn_stream=subprocess.check_output(
                'ls '+self.gfs_src+'gfs*nc', shell=True).decode('utf-8')
        fn_list=fn_stream.split()
       
        self.fn_list=fn_list
        
        self.dateseries=pd.date_range(
                start=self.fc_init_ts, 
                periods=int(self.gfs_days)*24/int(self.gfs_frq)+1, 
                freq=self.gfs_frq+'H')
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
        fn_list=self.fn_list
        init_ts=self.dateseries[0] 
        
        varlist=self.varlist
        ntasks=self.ntasks
        da_dic={}
       
        # let's do the multiprocessing magic!
        utils.write_log(print_prefix+'Multiprocessing initiated. Master process %s.' % os.getpid())

       
        len_file=len(fn_list)
        len_per_task=len_file//ntasks
        
        results=[]
        
        # start process pool
        process_pool = Pool(processes=ntasks)
        
        # open tasks ID 0 to ntasks-2
        for itsk in range(ntasks-1):  
            
            sub_list=fn_list[itsk*len_per_task:(itsk+1)*len_per_task]
            sub_ts=self.dateseries[
                    itsk*len_per_task:(itsk+1)*len_per_task]
            
            result=process_pool.apply_async(
                run_mtsk, 
                args=(itsk, sub_list, sub_ts, da_dic, self, ))
            results.append(result)

        # open ID ntasks-1 in case of residual
        sub_list=fn_list[(ntasks-1)*len_per_task:]
        sub_ts=self.dateseries[(ntasks-1)*len_per_task:]

        result=process_pool.apply_async(
            run_mtsk, 
            args=(ntasks-1, sub_list, sub_ts, da_dic, self, ))

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
        self.lon=temp_var['lon_0'].values
        self.lat=temp_var['lat_0'].values

def run_mtsk(itsk, sub_list, sub_ts, da_dic, gfs_hdl):
    """
    multitask read file
    """

    varlist=gfs_hdl.varlist


    len_files=len(sub_list)
    da_dic={}
    

    # get fn: gfs.t00z.pgrb2.0p25.f000.nc
    fn=sub_list[0].split('/')[-1]
    
    utils.write_log('%sTASK[%02d]: Read %04d of %04d --- %s' % (
        print_prefix, itsk, 0,(len_files-1), fn))

    for var in varlist:
        var_temp=get_var_xr(sub_list[0], sub_ts[0], var)
        
        da_dic[var]=var_temp.sel(
                lat_0=slice(gfs_hdl.s_sn,gfs_hdl.e_sn),
                lon_0=slice(gfs_hdl.s_we, gfs_hdl.e_we))
    # read the rest files in the list
    for idx, full_fn in enumerate(sub_list[1:]):
        
        fn=full_fn.split('/')[-1]
        
        utils.write_log('%sTASK[%02d]: Read %04d of %04d --- %s' % (
            print_prefix, itsk, (idx+1), (len_files-1), fn))
        
        for var in varlist:
            var_temp=get_var_xr(full_fn, sub_ts[idx+1],var)
            var_temp=var_temp.sel(
                lat_0=slice(gfs_hdl.s_sn,gfs_hdl.e_sn),
                lon_0=slice(gfs_hdl.s_we, gfs_hdl.e_we))
            
            da_dic[var]=xr.concat([da_dic[var],var_temp], dim='time')
    
    utils.write_log('%sTASK[%02d]: All files loaded.' % (print_prefix, itsk))
    
    # convert gpm to m^2/s^2
    da_dic['HGT_P0_L100_GLL0']=G*da_dic['HGT_P0_L100_GLL0']
    return da_dic

def get_var_xr(src, ts, var):
    ''' retrun var xr obj according to var name'''
    
    da=xr.load_dataset(src)
    var_xr=da[var]
    var_xr['time']=ts
    return var_xr


def get_varlist(cfg):
    ''' sepgfste vars in cfg varlist csv format '''
    varlist=cfg['SHARE']['var'].split(',')
    varlist=[ele.strip() for ele in varlist]
    return varlist


if __name__ == "__main__":
    '''
    Code for unit test
    '''
    pass
    
