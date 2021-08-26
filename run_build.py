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
May 25, 2021 --- v0.10: Multivariable concatenation
Jun  2, 2021 --- v0.90: Major pipelines 
Jul  2, 2021 --- v0.95: Naming, evaluation, and 2-D topology 
Jul 12, 2021 --- v0.96: neighbourhood function and grid search module
Jul 20, 2021 --- v0.97: Multiprocessing in IO and grid search training
Jul 30, 2021 --- v0.98: Major debug for eliminating mexican_hat neighbour func
Aug 11, 2021 --- v0.99: Major features addition
Aug 26, 2021 --- v1.00: Major features addition: closest Euclidean Distance Match

Zhenning LI
'''

import numpy as np
import pandas as pd
import os, sys, logging.config

import lib 
import core
from utils import utils

CWD=sys.path[0]
def main_run():
    
    print('*************************PRISM START*************************')
       
    # wall-clock ticks
    time_mgr=lib.time_manager.TimeManager()
    
    # logging manager
    logging.config.fileConfig(CWD+'/conf/logging_config.ini')
    
    utils.write_log('Read Config...')
    cfg_hdl=lib.cfgparser.read_cfg(CWD+'/conf/config.ini')
    
    if cfg_hdl['OTHER'].getboolean('relink_pathwrf'):
        utils.write_log('Relink training pathwrf...')
        utils.link_path(cfg_hdl)
    
    
    # init grid searcher for hyper-parameter optimazation
    grid_searcher=lib.grid_searcher.GridSearcher(cfg_hdl)
    
    # init wrf handler and read training data
    wrf_hdl=lib.preprocess_wrfinp.WrfMesh(cfg_hdl)
    # initiate clusterer
    prism=core.prism.Prism(wrf_hdl,cfg_hdl)

    grid_searcher.search(cfg_hdl, prism)


    print('*********************PRISM ACCOMPLISHED*********************')

if __name__=='__main__':
    main_run()
