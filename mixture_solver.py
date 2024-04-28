import numpy as np
import itertools as it
from sklearn.mixture import GaussianMixture
from pprint import pprint
import pdb


class GaussianMixtureSolver():
    '''
    '''

    def __init__(self):
        '''
        '''

    def get_best_estimated_matching_error(self,intv_args_dict,gm,debug=False):
        '''
        '''
        mean_list=gm.means_
        cov_list = gm.covariances_

        comp_perm_list = it.permutations(intv_args_dict.keys())
        min_err = float("inf")
        min_perm=None
        for comp_perm in comp_perm_list:
            err = 0.0
            for cidx,comp in enumerate(comp_perm):
                mean_err = np.sum(
                    np.abs(mean_list[cidx]-intv_args_dict[comp]["true_params"]["mui"])
                )/np.sum(np.abs(intv_args_dict[comp]["true_params"]["mui"])+1e-7)
                cov_error = np.sum(
                    np.abs(cov_list[cidx]-intv_args_dict[comp]["true_params"]["Si"])
                )/np.sum(np.abs(intv_args_dict[comp]["true_params"]["Si"])+1e-7)
                
                
                err+=mean_err+cov_error
            #Now checking if this perm/matching gives minimum error
            if err<min_err:
                min_err=err 
                min_perm=comp_perm
        
        if debug:
            print("error:",min_err)
            print("min_perm:",min_perm)
        return min_err,min_perm


    def mixture_disentangler(self,intv_args_dict,mixture_samples,debug=False):
        #Now we are ready run the mini disentanglement algos
        gm = GaussianMixture(n_components=len(intv_args_dict),random_state=0).fit(
                                                    mixture_samples
        )#None number of component not allowed!
        
        if debug:
            print("==================================")
            print("Estimated Means (unmatched):")
            pprint(gm.means_*(gm.means_>1e-5))
            print("==================================")
            pprint("Estimated Covarainces (unmatched):")
            pprint(gm.covariances_*(gm.covariances_>1e-5))
        min_err,min_perm = self.get_best_estimated_matching_error(intv_args_dict,gm)
        return min_err,min_perm