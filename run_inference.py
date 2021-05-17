#!/home/metctm1/array/soft/anaconda3/bin/python
'''
Date: May 05, 2021
Prism is a SOM-based classifier to classify weather types 
according to regional large-scale weather charts obtained
from WRFOUT.

This is the main script to drive the model

Revision:
May 05, 2021 --- Architecture Design 

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
    time_mgr=lib.time_manager.time_manager()
    
    # logging manager
    logging.config.fileConfig('./conf/logging_config.ini')
    
    
    utils.write_log('Read Config...')
    cfg_hdl=lib.cfgparser.read_cfg('./conf/config.ini')

    utils.write_log('Read Input Observations...')

    if cfg_hdl['CORE'].getboolean('model_rebuild_flag'):
        pass
    
    if cfg_hdl['CORE'].getboolean('model_infer_flag'):
        pass
    
    print('*********************PRISM ACCOMPLISHED*********************')



def run_mtsk(itsk, obv_lst, clock, estimator, fields_hdl, cfg_hdl):
    """
    Aeolus cast function for multiple processors
    """
    utils.write_log('TASK[%02d]: Aeolus Interpolating Estimator Casting...' % itsk)
    while not(clock.done):
           
        estimator.cast(obv_lst, fields_hdl, clock)

        utils.write_log('TASK[%02d]: Output Diagnostic UVW Fields...' % itsk )
        core.aeolus.output_fields(cfg_hdl, estimator, clock)
        
        clock.advance()
    utils.write_log('TASK[%02d]: Aeolus Subprocessing Finished!' % itsk)
    
    return 0

if __name__=='__main__':
    main_run()
