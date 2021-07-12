#/usr/bin/env python3
"""configuration funcs to get parameters from user"""
from utils import utils
import lib
import itertools

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
            self.gs_sigma=lib.cfgparser.cfg_get_varlist(cfg_hdl,'GRID_SEARCH','gs_sigma')
            self.gs_lr=lib.cfgparser.cfg_get_varlist(cfg_hdl,'GRID_SEARCH','gs_learning_rate')
            self.gs_nodey=lib.cfgparser.cfg_get_varlist(cfg_hdl,'GRID_SEARCH','gs_1dnodey')
            self.gs_nb_func=lib.cfgparser.cfg_get_varlist(cfg_hdl,'GRID_SEARCH','gs_nb_func')
            self.gs_respl_int=lib.cfgparser.cfg_get_varlist(cfg_hdl,'GRID_SEARCH','gs_respl_int')
        else:
            utils.write_log(print_prefix+'Single hyper-para comb, no need grid search...')

    
    def search(self, cfg, prism):
        """ search best hyper-parameter combination in space """
        if self.gs_flag:

            best_score=-1

            # all possible combs            
            comb=list(itertools.product(
                self.gs_sigma,self.gs_lr,
                self.gs_nodey, self.gs_nb_func))
            num_comb=len(comb)
            utils.write_log(print_prefix+'grid search through '+str(num_comb)+' possible combinations...')
            
            prism.n_nodex=1
            for idx, itms in enumerate(comb):
                # assignment
                prism.sigma=float(itms[0])
                prism.lrate=float(itms[1])
                prism.n_nodey=int(itms[2])
                prism.nb_func=itms[3]
                
                # debug output
                utils.write_log(print_prefix+'grid search round:'+str(idx+1)+'/'+str(num_comb))
                utils.write_log(print_prefix+'grid search sigma='+itms[0])
                utils.write_log(print_prefix+'grid search lrate='+itms[1])
                utils.write_log(print_prefix+'grid search nodey='+itms[2])
                utils.write_log(print_prefix+'grid search nb_func='+itms[3])

                # execute
                prism.train()
                prism.evaluate(cfg)
                
                # examine score
                curr_score=prism.edic['silhouette_score']
                if curr_score > best_score:
                    utils.write_log(print_prefix+'found current best score:'+str(curr_score)+', archive model...')
                    best_score=curr_score
                    prism.edic.update({
                            'gs_sigma':prism.sigma,
                            'gs_lrate':prism.lrate,
                            'gs_1dnodey':prism.n_nodey,
                            'gs_nb_func':prism.nb_func
                            })
                    prism.archive()
                
            utils.write_log(print_prefix+'All search done, best silhouette_score:'+str(best_score)+', model archived...')
        
        else:
            # execute
            prism.train()
            prism.evaluate(cfg)
            prism.archive()

