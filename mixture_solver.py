import numpy as np
import itertools as it
from sklearn.mixture import GaussianMixture
from pprint import pprint
import pdb
import pandas as pd


from pgmpy.estimators import PC
from causaldag import unknown_target_igsp
import causaldag as cd
from causaldag.utils.ci_tests import gauss_ci_suffstat, gauss_ci_test, MemoizedCI_Tester
from causaldag.utils.invariance_tests import gauss_invariance_suffstat, gauss_invariance_test, MemoizedInvarianceTester


class GaussianMixtureSolver():
    '''
    '''
    def __init__(self,true_gSCM):
        '''
        '''
        self.true_gSCM = true_gSCM

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
        
        #Updating the interv dict with appropriate perm
        for cidx,comp in enumerate(min_perm):
            intv_args_dict[comp]["est_params"]={}
            intv_args_dict[comp]["est_params"]["mui"] = mean_list[cidx]
            intv_args_dict[comp]["est_params"]["Si"] = cov_list[cidx]

        return min_err,min_perm,intv_args_dict

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
        min_err,min_perm,intv_args_dict = self.get_best_estimated_matching_error(
                                                                intv_args_dict,gm)

        return min_err,intv_args_dict
    
    def run_pc_over_each_component(self,intv_args_dict,num_samples):
        '''
        '''
        sample_per_comp = num_samples//len(intv_args_dict.keys())
        #Iterating over all the estimated component and 
        for comp in intv_args_dict.keys():
            est_mui = intv_args_dict[comp]["est_params"]["mui"]
            est_Si = intv_args_dict[comp]["est_params"]["Si"]

            #Now we will generate samples from this distribution
            samples = pd.DataFrame(
                        np.random.multivariate_normal(est_mui,
                                                    est_Si,
                                                    size=sample_per_comp),
                        columns=[str(idx) for idx in range(est_mui.shape[0])]
            )

            #Now we will run the CI test
            pc = PC(samples)
            pdag = pc.skeleton_to_pdag(*pc.build_skeleton(ci_test='pearsonr'))
            intv_args_dict[comp]["est_pdag"] = pdag
            # pdb.set_trace()
    
        return intv_args_dict
    
    def identify_intervention_utigsp(self,intv_args_dict,num_samples):
        '''
        They assume we have access to the observational data
        code taken from UTGSP tutorial
        '''
        sample_per_comp = num_samples//len(intv_args_dict.keys())
        num_nodes = intv_args_dict[comp]["true_params"]["mui"].shape[0]

        #Generating the observational samples
        obs_true_mui = intv_args_dict["obs"]["true_params"]["mui"]
        obs_true_Si = intv_args_dict["obs"]["true_params"]["Si"]
        obs_samples = np.random.multivariate_normal(est_mui,
                                                est_Si,
                                                size=sample_per_comp)


        #Iterating over all the estimated component and
        actual_target_list = []
        utarget_sample_list  = []
        for comp in intv_args_dict.keys():
            #Skipping the obs
            if comp=="obs":
                continue
            actual_target_list.append(comp)

            est_mui = intv_args_dict[comp]["est_params"]["mui"]
            est_Si = intv_args_dict[comp]["est_params"]["Si"]

            #Now we will generate samples from this distribution
            intv_samples = np.random.multivariate_normal(est_mui,
                                                    est_Si,
                                                    size=sample_per_comp)
            utarget_sample_list.append(intv_samples)
        
        #Creating the suddicient statistics
        obs_suffstat = gauss_ci_suffstat(obs_samples)
        invariance_suffstat = gauss_invariance_suffstat(obs_samples, 
                                                        utarget_sample_list)
        #CI tester and invariance tester
        alpha = 1e-3
        alpha_inv = 1e-3
        ci_tester = MemoizedCI_Tester(gauss_ci_test, 
                                        obs_suffstat, alpha=alpha)
        invariance_tester = MemoizedInvarianceTester(
                                        gauss_invariance_test,
                                        invariance_suffstat, 
                                        alpha=alpha_inv)
        
        #Runnng UGSP
        setting_list = [dict(known_interventions=[]) for _ in len(actual_target_list)]
        est_dag, est_targets_list = unknown_target_igsp(setting_list, 
                                                num_nodes, 
                                                ci_tester, 
                                                invariance_tester)
        print(est_targets_list)

        return est_dag,est_targets_list