#/usr/bin/env python
"""Commonly used utilities

    Function    
    ---------------
    obv_examiner(obv_df):
        Examine the input observational data
    
    throw_error(source, msg):
        Throw error with call source and error message

"""
import datetime
import os
import numpy as np
import pandas as pd
import logging

DEG2RAD=np.pi/180.0

def throw_error(source, msg):
    '''
    throw error and exit
    '''
    logging.error(source+msg)
    exit()

def write_log(msg, lvl=20):
    '''
    write logging log to log file
    level code:
        CRITICAL    50
        ERROR   40
        WARNING 30
        INFO    20
        DEBUG   10
        NOTSET  0
    '''

    logging.log(lvl, msg)

def link_path(cfg):
    """ link path wrfout to input dir """
    dateseries=pd.date_range(start=cfg['TRAINING']['training_start'], end=cfg['TRAINING']['training_end'])
    for datestamp in dateseries:
        yyyy=datestamp.strftime('%Y')
        mm=datestamp.strftime('%m')
        dd=datestamp.strftime('%d')

        src_wrfpath=cfg['OTHER']['src_wrf']+yyyy
        src_wrfpath=src_wrfpath+'/'+yyyy+mm+'/'+yyyy+mm+dd+'12'
        try:
            os.system('ln -sf '+src_wrfpath+'/wrfout_d01_'+yyyy+'-'+mm+'-'+dd+'_12:00:00 ./input/training/')
        except:
            write_log('wrfout_d01_'+yyyy+'-'+mm+'-'+dd+'_12:00:00 not found',30)

def link_realtime(cfg):
    """ link realtime wrfout to input dir """
    today=datetime.datetime.now()
    while 1:
        yyyy=today.strftime('%Y')
        mm=today.strftime('%m')
        dd=today.strftime('%d')
        src_wrfpath=cfg['OTHER']['src_wrf']+yyyy
        src_wrfpath=src_wrfpath+'/'+yyyy+mm+'/'+yyyy+mm+dd+'12'
        # test if file exist
        if os.path.exists(src_wrfpath):
            os.system('rm -f  ./input/inference/*')
            os.system('ln -sf '+src_wrfpath+'/wrfout_d01* ./input/inference/')
            write_log(src_wrfpath+' has been successfully linked!')
            break
        else:
            write_log(src_wrfpath+' not found, try 1 day before',30)
            today=today-datetime.timedelta(days=1)

def get_std_dim0(data):
    """ standardize the series on the 0 dim """
    data_mean=data.mean(axis=0)
    data_std=data.std(axis=0)
    data=(data-data_mean)/data_std
    return data, data_mean, data_std

