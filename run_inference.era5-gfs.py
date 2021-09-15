#!/home/metctm1/array/soft/anaconda3/bin/python
'''
Date: Sep 12, 2021
Prism is a SOM-based classifier to classify weather types 
according to regional large-scale weather charts obtained
from ERA5 and GFS.

This is the main script to drive the model

Revision:
May 05, 2021 --- Architecture Design 
May 19, 2021 --- Implementation
Sep 12, 2021 --- Fit ERA5-GFS pipeline

Zhenning LI
'''

import numpy as np
import pandas as pd
import os, logging

import lib 
import core
from utils import utils
from multiprocessing import Pool, sharedctypes

def main_run():
    
    print('*************************PRISM START*************************')
       
    # wall-clock ticks
    time_mgr=lib.time_manager.TimeManager()
    
    # logging manager
    logging.config.fileConfig('./conf/logging_config.ini')
    
    
    utils.write_log('Read Config...')
    cfg_hdl=lib.cfgparser.read_cfg('./conf/config.era5-gfs.ini')

    if cfg_hdl['INFERENCE'].getboolean('down_realtime_gfs'):
        utils.write_log('Download realtime GFS...')
        utils.down_gfs(cfg_hdl)
    utils.write_log('Preprocess GFS...')
    gfs_hdl=lib.preprocess_gfsinp.GFSMesh(cfg_hdl, 'inference') 
    utils.write_log('Construct Prism...')
    prism=core.prism_era5_gfs.Prism(gfs_hdl,cfg_hdl, 'inference')
    utils.write_log('Prism Cast...')
    prism.cast() 
    print('*********************PRISM ACCOMPLISHED*********************')


if __name__=='__main__':
    main_run()
