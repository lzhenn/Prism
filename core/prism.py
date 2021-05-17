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

print_prefix='core.aeolus>>'

class prism:

    '''
    Aeolus interpolator, interpolate in-situ obvs onto wrf mesh 
    
    Attributes
    -----------
    dis_mtx_u(n_sn, n_we_stag, n_obv), float, distance between obv and grid point, matrix on staggered u grid

    Methods
    -----------
    train(), cast interpolation on WRF data
    cast(), cast interpolation on WRF data 

    '''
    
    def __init__(self, fields_hdl, cfg_hdl):
        """ construct aeolus interpolator """
        self.nrec=fields_hdl.nrec
        nrow=fields_hdl.nrow
        ncol=fields_hdl.ncol

        self.data=fields_hdl.slp.values.reshape((self.nrec,-1))
        self.data=(self.data-self.data.mean(axis=0))/self.data.std(axis=0)
        self.nfea=nrow*ncol 
    def train(self):
        mapsize=4
        som = minisom.MiniSom(1, mapsize, self.nfea, sigma=0.3, learning_rate=0.5) # initialization of 6x6 SOM
        som.train(self.data, 100, verbose=True) # trains the SOM with 100 iterations
        print([som.winner(x) for x in self.data])
        '''
        som = sompy.SOMFactory.build(self.data, mapsize, mask=None, mapshape='planar', 
                lattice='rect', normalization='var', initialization='pca', 
                neighborhood='gaussian', training='batch', name='sompy')
        som.train(n_job=1, verbose='info')  # verbose='debug' will print more, and verbose=None wont print anything
        '''
    def cast(self, obv_lst, fields_hdl, clock):
        """ cast interpolation on WRF mesh """
        # e-folding rate in cross-layer interpolation 
        efold_r=fields_hdl.efold_r
        # convective propagation distance
        conv_t=fields_hdl.conv_t
        

if __name__ == "__main__":
    pass
