#!/home/metctm1/array/soft/anaconda3/bin/python
'''
Date: May 05, 2021
Prism is a SOM-based classifier to classify weather types 
according to regional large-scale weather charts obtained
from WRFOUT.

This is the main script to drive the model

Revision:
May 05, 2021 --- Architecture Design 
May 17, 2021 --- MVP: training pipeline
Zhenning LI
'''

import numpy as np
import pandas as pd
import os, logging.config

import lib 
import core
from utils import utils
from multiprocessing import Pool, sharedctypes

def main_run():
    
    print('*************************PRISM START*************************')
       
    # wall-clock ticks
    time_mgr=lib.time_manager.time_manager()
    
    # logging manager
    logging.config.fileConfig('./conf/logging_config.ini')
    
    utils.write_log('Read Config...')
    cfg_hdl=lib.cfgparser.read_cfg('./conf/config.ini')

    if cfg_hdl['OTHER'].getboolean('relink_pathwrf'):
        utils.write_log('Relink training pathwrf...')
        utils.link_path(cfg_hdl)
    
    wrf_hdl=lib.preprocess_wrfinp.wrf_mesh(cfg_hdl)
    prism=core.prism.prism(wrf_hdl,cfg_hdl)
    prism.train()

    print('*********************PRISM ACCOMPLISHED*********************')

if __name__=='__main__':
    main_run()
