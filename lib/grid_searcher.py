#/usr/bin/env python3
"""configuration funcs to get parameters from user"""
from utils import utils
import lib

import numpy as np
import itertools, os, time
from multiprocessing import Pool, sharedctypes

print_prefix='lib.grid_searcher>>'

class GridSearcher:
    '''
    GridSearcher class for searching best combinations of hyper-parameter 
    for prism clusterer

    Attributes
    -----------

    Methods
    -----------
 

    '''

    def __init__(self, cfg_hdl):
        """ construct gridsearcher to control prism """

        self.gs_flag=cfg_hdl['TRAINING'].getboolean('grid_search_opt')
        
        if self.gs_flag:
            utils.write_log(print_prefix+'Init grid searcher...')
            self.nworkers=int(cfg_hdl['GRID_SEARCH']['gs_nworkers'])
            self.gs_sigma=lib.cfgparser.cfg_get_varlist(cfg_hdl,'GRID_SEARCH','gs_sigma')
            self.gs_lr=lib.cfgparser.cfg_get_varlist(cfg_hdl,'GRID_SEARCH','gs_learning_rate')
            self.gs_nodexy=lib.cfgparser.cfg_get_varlist(cfg_hdl,'GRID_SEARCH','gs_nodexy')
            self.gs_nb_func=lib.cfgparser.cfg_get_varlist(cfg_hdl,'GRID_SEARCH','gs_nb_func')
            self.gs_iter=lib.cfgparser.cfg_get_varlist(cfg_hdl,'GRID_SEARCH','gs_iterations')
        else:
            utils.write_log(print_prefix+'Single hyper-para comb, no need grid search...')

    
    def search(self, cfg, prism):
        """ search best hyper-parameter combination in space """
        
        if self.gs_flag:

            best_score=-1
            # all possible combs            
            comb=list(itertools.product(
                self.gs_sigma,self.gs_lr,
                self.gs_nodexy, self.gs_nb_func,
                self.gs_iter))
            num_comb=len(comb)

            utils.write_log(print_prefix+'Grid Search through '+str(num_comb)+' possible combinations...')
            
            # ------Below for multitaks grid search with shared memory on training data----------
            utils.write_log(print_prefix+'Multitask Grid Search with master process %s.' % os.getpid())
            ntasks=self.nworkers
            
            # Initial shared data and delete prism attr refrence
            # this is used for lowering copy overhead 
            # in multiprocessing forking
            train_data=prism.data
            shared_data = create_share_type(train_data)
            delattr(prism,'data')
            
            len_per_task=num_comb//ntasks
            results=[]
            
            # start process pool
            process_pool = Pool(processes=ntasks, 
                    initializer=_init, initargs=(shared_data,))

            # open tasks ID 0 to ntasks-2
            for itsk in range(ntasks-1): 
                icomb_lst=comb[itsk*len_per_task:(itsk+1)*len_per_task]
                results.append(process_pool.apply_async(run_mtsk,
                        args=(itsk, icomb_lst, prism, cfg, )))

            # open ID ntasks-1 in case of residual
            icomb_lst=comb[(ntasks-1)*len_per_task:]
            results.append(process_pool.apply_async(run_mtsk, 
                    args=(ntasks-1, icomb_lst, prism, cfg, )))

            utils.write_log('Waiting for all subprocesses done...')
            
            # wait childs come
            process_pool.close()
            process_pool.join()

           
            # ------Upper for multitaks grid search with shared memory on training data----------
            best_score=-1
            best_idx=-1
            
            for idx, res in enumerate(results):
                edic=res.get()
                if edic['silhouette_score'] > best_score:
                    best_score=edic['silhouette_score']
                    best_idx=idx

            best_edic=results[best_idx].get()
            utils.write_log(print_prefix+'''All search done, best silhouette_score:'''+str(best_score)+', archiving model...')
            prism.sigma, prism.lrate=best_edic['best_sigma'], best_edic['best_lrate']
            prism.n_nodey, prism.nb_func=best_edic['best_1dnodey'], best_edic['best_nb_func']
            prism.data=train_data             
        
        
        # execute for single run or for best grid search
        prism.train()
        prism.evaluate(cfg)

        if self.gs_flag:
            prism.edic.update({
                    'best_sigma':prism.sigma,
                    'best_lrate':prism.lrate,
                    'best_1dnodey':prism.n_nodey,
                    'best_nb_func':prism.nb_func
                    }) 
        # model archive
        prism.archive()

def run_mtsk(itsk, comb, prism, cfg):
    """
    run grid search in multitasks!
    """
     
    start = time.time()
    train_data = np.ctypeslib.as_array(s_data)
    num_comb=len(comb)
    best_score=-1

    for idx, itms in enumerate(comb):
        
        # assignment
        prism.sigma, prism.lrate=float(itms[0]), float(itms[1])
        prism.nb_func=itms[3]
        prism.n_nodex= int(itms[2].split('x')[0])
        prism.n_nodey= int(itms[2].split('x')[1])
        prism.iterations=int(itms[4])

        # debug output
        utils.write_log('%sTASK[%02d]: Grid Search round: %04d/%04d' % (
            print_prefix, itsk, idx+1, num_comb))
        utils.write_log('''%sTASK[%02d]: Grid Search para combinations:
                 sigma=%s, lrate=%s, nodexy=%s, 
                 nb_func=%s, iterations=%s''' %(
                    print_prefix, itsk, itms[0],itms[1], itms[2], 
                    itms[3], itms[4]))
        
        # execute
        prism.train(train_data=train_data, verbose=False)
        prism.evaluate(cfg, train_data=train_data, verbose=False)
        
        # store best model in this child process
        curr_score=prism.edic['silhouette_score']
        if curr_score > best_score:
            utils.write_log('%sTASK[%02d]: Grid Search Found best silhouette_score: %5.3f' % (
                print_prefix, itsk, curr_score))
            
            best_score=curr_score
            best_edic=prism.edic
            best_edic.update({
                    'best_sigma':prism.sigma,
                    'best_lrate':prism.lrate,
                    'best_1dnodey':prism.n_nodey,
                    'best_nb_func':prism.nb_func
                    })  

    end = time.time()
    utils.write_log('%sTASK[%02d]: Grid Search Run completed with %0.3f seconds elapsed.' % (
                print_prefix, itsk, (end - start)))
    
    return best_edic 

def create_share_type(np_array):
    ''' create shared memory among processors with training data '''
    
    np_carr = np.ctypeslib.as_ctypes(np_array)
    shared_array = sharedctypes.Array(np_carr._type_, np_carr, lock=True) 
    return shared_array

def _init(shared_data):
    """ 
        Each pool process calls this initializer. Load the array
        to be populated into that process's global namespace 
    """
    global s_data
    s_data=shared_data



