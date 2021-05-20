#/usr/bin/env python
"""
Core Component: Prism Classifier 
    Classes: 
    -----------
        prism: core class, weather classifier 

    Functions:
    -----------
"""
import xarray as xr
import numpy as np
import copy
import sys, os
from utils import utils
import minisom
import pickle

print_prefix='core.aeolus>>'

class prism:

    '''
    Aeolus interpolator, interpolate in-situ obvs onto wrf mesh 
    
    Attributes
    -----------
    dis_mtx_u(n_sn, n_we_stag, n_obv), float, distance between obv and grid point, matrix on staggered u grid

    Methods
    -----------
    train(), train the model by WRF data
    cast(), cast on WRF data 

    '''
    
    def __init__(self, wrf_hdl, cfg_hdl, call_from='trainning'):
        """ construct prism classifier """
        self.nrec=wrf_hdl.nrec
        nrow=self.nrow=wrf_hdl.nrow
        ncol=self.ncol=wrf_hdl.ncol

        self.nfea=nrow*ncol 
       
        self.dateseries=wrf_hdl.dateseries

        self.xlat, self.xlong=wrf_hdl.xlat, wrf_hdl.xlong
        if call_from=='trainning':
            raw_data=wrf_hdl.data.values.reshape((self.nrec,-1))
            self.preprocess=cfg_hdl['TRAINING']['preprocess_method']
            if self.preprocess == 'temporal_norm':
                self.data, self.mean, self.std=utils.get_std_dim0(copy.copy(raw_data))
            elif self.preprocess=='original':
                self.data=raw_data

            self.n_types=int(cfg_hdl['TRAINING']['n_types'])
            self.sigma=float(cfg_hdl['TRAINING']['sigma'])
            self.lrate=float(cfg_hdl['TRAINING']['learning_rate'])
            self.iterations=int(cfg_hdl['TRAINING']['iterations'])
        elif call_from=='inference':
            db_in=xr.load_dataset('./db/som_cluster.nc')            
            self.preprocess=db_in.attrs['preprocess_method']
            if self.preprocess == 'temporal_norm':
                mean, std = db_in['mean'], db_in['std']
                self.data=wrf_hdl.data.values
                for ii in range(0, self.nrec):
                    self.data[ii,:,:]=(self.data[ii,:,:]-mean)/std
                self.data=self.data.reshape((self.nrec,-1))

    def train(self):
        """ train the prism classifier """
        utils.write_log(print_prefix+'trainning...')

        som = minisom.MiniSom(1, self.n_types, self.nfea, 
                sigma=self.sigma, learning_rate=self.lrate) 
        
        som.train(self.data, self.iterations, verbose=True) 
       
        self.som=som

        # archive the model and clustered nodes
        self.archive()

    def cast(self):
        """ cast the prism on new synoptic maps """
        self.load()
        
        # archive classification result in csv
        winners=[self.som.winner(x) for x in self.data]
        with open('./output/inference_cluster.csv', 'w') as f:
            for datestamp, winner in zip(self.dateseries, winners):
                f.write(datestamp.strftime('%Y-%m-%d_%H:%M:%S,')+str(winner[1])+'\n')


    def archive(self):
        """ archive the prism classifier in database """

        # archive model
        with open('./db/som.archive', 'wb') as outfile:
            pickle.dump(self.som, outfile)

        # archive classification result in csv
        winners=[self.som.winner(x) for x in self.data]
        with open('./db/train_cluster.csv', 'w') as f:
            for datestamp, winner in zip(self.dateseries, winners):
                f.write(datestamp.strftime('%Y-%m-%d_12:00:00,')+str(winner[1])+'\n')

        # archive classification result in netcdf
        centroid=self.som.get_weights()[0]
        centroid=centroid.reshape(self.n_types, self.nrow, self.ncol)
        
        ds_out=self.org_output_nc(centroid)
  
        out_fn='./db/som_cluster.nc'
        ds_out.to_netcdf(out_fn)



    def org_output_nc(self, centroid):
        """ organize output file """
        if self.preprocess == 'temporal_norm':
            self.mean=self.mean.reshape(self.nrow, self.ncol)
            self.std=self.std.reshape(self.nrow, self.ncol)
            ds_out= xr.Dataset(
                data_vars={   
                    'som_cluster':(['ntype','nrow', 'ncol'], centroid),
                    'mean':(['nrow', 'ncol'], self.mean),
                    'std':(['nrow', 'ncol'], self.std),
                    'xlat':(['nrow', 'ncol'], self.xlat),
                    'xlong':(['nrow', 'ncol'], self.xlong),
                },  
                coords={
                    },
                attrs={
                    'preprocess_method':self.preprocess
                    }
            )  
        elif self.preprocess=='original':
            ds_out= xr.Dataset(
                data_vars={   
                    'som_cluster':(['ntype','nrow', 'ncol'], centroid),
                    'xlat':(['nrow', 'ncol'], self.xlat),
                    'xlong':(['nrow', 'ncol'], self.xlong),
                },  
                coords={
                    },
                attrs={
                    'preprocess_method':self.preprocess
                    }
            )
        return ds_out


    def load(self):
        """ load the archived prism classifier in database """
        with open('./db/som.archive', 'rb') as infile:
            self.som = pickle.load(infile)

if __name__ == "__main__":
    pass
