import numpy as np
import itertools as it
from sklearn.mixture import GaussianMixture
from pprint import pprint
import pdb
import pandas as pd
import json
import pathlib
import multiprocessing as mp


from pgmpy.estimators import PC
from causaldag import unknown_target_igsp,igsp
import causaldag as cd
from causaldag import partial_correlation_suffstat, partial_correlation_test, MemoizedCI_Tester
from causaldag import gauss_invariance_suffstat, gauss_invariance_test, MemoizedInvarianceTester


from scm_module import *

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
                )#/np.sum(np.abs(intv_args_dict[comp]["true_params"]["mui"])+1e-7)
                cov_error = np.sum(
                    np.abs(cov_list[cidx]-intv_args_dict[comp]["true_params"]["Si"])
                )#/np.sum(np.abs(intv_args_dict[comp]["true_params"]["Si"])+1e-7)
                
                
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

    def mixture_disentangler(self,intv_args_dict,mixture_samples,tol,debug=False):
        #Now we are ready run the mini disentanglement algos
        gm = GaussianMixture(n_components=len(intv_args_dict),
                                    tol=tol,
                                    random_state=0,

        ).fit(mixture_samples)
        #None number of component not allowed!
        
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
            #Check significance level (fisher transform CI test)
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
        num_nodes = intv_args_dict["obs"]["true_params"]["mui"].shape[0]

        #Generating the observational samples
        obs_true_mui = intv_args_dict["obs"]["true_params"]["mui"]
        obs_true_Si = intv_args_dict["obs"]["true_params"]["Si"]
        obs_samples = np.random.multivariate_normal(obs_true_mui,
                                                obs_true_Si,
                                                size=sample_per_comp)
        #Generating the samples from the estimated params
        obs_est_mui = intv_args_dict["obs"]["est_params"]["mui"]
        obs_est_Si = intv_args_dict["obs"]["est_params"]["Si"]
        obs_est_samples = np.random.multivariate_normal(obs_est_mui,
                                                obs_est_Si,
                                                size=sample_per_comp)
        
        


        #Iterating over all the estimated component but first adding the obs first
        #This is required by the UGSP
        actual_target_list = ["obs",]
        utarget_sample_list  = [obs_est_samples,]
        utarget_oracle_sample_list = [obs_samples,]
        igsp_setting_list = [dict(interventions=[]),]

        for comp in intv_args_dict.keys():
            #Skipping the obs cuz already added above
            if comp=="obs":
                continue
            actual_target_list.append(comp)
            igsp_setting_list.append(dict(
                                        interventions=[int(comp)]
            ))

            #Generating the samples from the estimated parameters
            est_mui = intv_args_dict[comp]["est_params"]["mui"]
            est_Si = intv_args_dict[comp]["est_params"]["Si"]
            #Now we will generate samples from this distribution
            intv_samples = np.random.multivariate_normal(est_mui,
                                                    est_Si,
                                                    size=sample_per_comp)
            utarget_sample_list.append(intv_samples)



            #Genearting the samples for the oracle using the exact parameters
            oracle_mui = intv_args_dict[comp]["true_params"]["mui"]
            oracle_Si = intv_args_dict[comp]["true_params"]["Si"]
            oracle_intv_samples = np.random.multivariate_normal(oracle_mui,
                                                    oracle_Si,
                                                    size=sample_per_comp)
            utarget_oracle_sample_list.append(oracle_intv_samples)
        
        #Creating the suddicient statistics
        obs_suffstat = partial_correlation_suffstat(obs_samples)
        invariance_suffstat = gauss_invariance_suffstat(obs_samples, 
                                                utarget_sample_list)
        oracle_invariance_suffstat = gauss_invariance_suffstat(obs_samples, 
                                                utarget_oracle_sample_list)

        #CI tester and invariance tester
        alpha = 1e-3
        alpha_inv = 1e-3
        ci_tester = MemoizedCI_Tester(partial_correlation_test, 
                                        obs_suffstat, alpha=alpha)
        invariance_tester = MemoizedInvarianceTester(
                                        gauss_invariance_test,
                                        invariance_suffstat, 
                                        alpha=alpha_inv)
        oracle_invariance_tester = MemoizedInvarianceTester(
                                        gauss_invariance_test,
                                        oracle_invariance_suffstat, 
                                        alpha=alpha_inv)
        
        #Runnng UTGSP
        setting_list = [dict(known_interventions=[]) for _ in range(len(actual_target_list))]
        print("Running UGSP")
        est_dag, est_targets_list = unknown_target_igsp(setting_list, 
                                                set(list(range(num_nodes))), 
                                                ci_tester, 
                                                invariance_tester)
        
        #Running the Oracle-UTGSP (i.e with sample from correct params)
        setting_list = [dict(known_interventions=[]) for _ in range(len(actual_target_list))]
        print("Running Oracle-UTGSP")
        oracle_est_dag, oracle_est_targets_list = unknown_target_igsp(setting_list, 
                                                set(list(range(num_nodes))), 
                                                ci_tester, 
                                                oracle_invariance_tester)

        #Running the IGSP to see the upper bound of the estimation
        #Question: Should we put extra obs data similar to UTGSP here?
        print("Running the IGSP")
        igsp_est_dag = igsp(igsp_setting_list,
                            set(list(range(num_nodes))),
                            ci_tester, 
                            invariance_tester,
        )

        #Here we are not matching the target using the similarty but 
        #rather than the parameter which is already done in the step1
        #unlike our last project where we matched target using JS.
        #Adding the estimated targets to the acutal targets dct
        for act_tgt,est_tgt in zip(actual_target_list,est_targets_list):
            intv_args_dict[act_tgt]["est_tgt"]=list(est_tgt)
        
        for act_tgt,oracle_est_tgt in zip(actual_target_list,oracle_est_targets_list):
            intv_args_dict[act_tgt]["oracle_est_tgt"]=list(oracle_est_tgt)

        
        return est_dag,intv_args_dict,oracle_est_dag,igsp_est_dag


#SINGLE EXPERIMENT KERNEL
def run_mixture_disentangle(args):
    '''
    '''
    #Collecting the metrics to evaluate the results later
    metric_dict={}

    #Creating the SCM
    gargs={}
    gargs["noise_mean_list"]=[args["obs_noise_mean"],]*args["num_nodes"]
    gargs["noise_sigma_list"]=[args["obs_noise_var"],]*args["num_nodes"]
    scmGen = RandomSCMGenerator(num_nodes=args["num_nodes"],
                                  max_strength=args["max_edge_strength"],
                                  num_parents=args["num_parents"],
                                  args=args,
    )
    gSCM = scmGen.generate_gaussian_scm(scm_args=gargs)
    
    
    #Step 0: Generating the samples and interventions configs
    print("Generating mixture samples!")
    intv_args_dict,mixture_samples = gSCM.generate_gaussian_mixture(
                                                        args["intv_type"],
                                                        args["intv_targets"],
                                                        args["new_noise_mean"],
                                                        args["new_noise_sigma"],
                                                        args["mix_samples"],
    )

    #Step 1: Running the disentanglement
    print("Step 1: Disentangling Mixture")
    gSolver = GaussianMixtureSolver(gSCM)
    err,intv_args_dict = gSolver.mixture_disentangler(intv_args_dict,
                                                        mixture_samples,
                                                        args["gmm_tol"])
    metric_dict["param_est_rel_err"]=err
    print("error:",err)
    





    #Step 2: Finding the graph for each component
    print("Stage 2: Estimating individual graph using PC")
    # gSolver.run_pc_over_each_component(intv_args_dict,args["stage2_samples"])
    est_dag,intv_args_dict,oracle_est_dag,igsp_est_dag = gSolver.identify_intervention_utigsp(
                                        intv_args_dict,args["stage2_samples"])
    metric_dict["est_dag"]=est_dag
    metric_dict["oracle_est_dag"]=oracle_est_dag
    metric_dict["igsp_dag"]=igsp_est_dag


    #Evaluation: 
    print("==========================")
    print("Estimated Target List")
    for comp in intv_args_dict.keys():
        if comp=="obs":
            continue 
        
        print("actual_tgt:{}\testimated_tgt:{}".format(
                                        int(comp),
                                        intv_args_dict[comp]["est_tgt"])
        )
    
    #Computing the SHD
    print("==========================")
    metric_dict=compute_shd(intv_args_dict,metric_dict)
    print("SHD:",metric_dict["shd"])
    print("Oracle-SHD:",metric_dict["oracle_shd"])
    print("actgraph:\n",metric_dict["act_dag"])
    print("est graph:\n",metric_dict["est_dag"])
    print("oracle est graph:\n",metric_dict["oracle_est_dag"])

    #Computing the js of target
    print("==========================")
    metric_dict=compute_target_jaccard_sim(intv_args_dict,metric_dict)
    print("Avg JS:",metric_dict["avg_js"])
    print("Avg Oracle JS:",metric_dict["avg_oracle_js"])
    
    
    #Dumping the experiment
    pickle_experiment_result_json(args,intv_args_dict,metric_dict)
    return intv_args_dict,metric_dict

def compute_shd(intv_args_dict,metric_dict):
    '''
    Assumption: the graph returned by utgsp is a dag (I think it is true
    from the code)
    '''
    #First of all we have to create a DAG object as per the causaldag
    obs_A = intv_args_dict["obs"]["true_params"]["Ai"]
    # print(obs_A)
    num_nodes = obs_A.shape[0]
    #Creating the actual DAG 
    act_dag = cd.DAG()
    act_dag.add_nodes_from([idx for idx in range(num_nodes)])
    #Adding the edges
    for tidx in range(num_nodes):
        for fidx in range(0,tidx):
            # print("tidx:{}\tfidx:{}\tval:{}".format(tidx,fidx,obs_A[fidx][tidx]))
            if abs(obs_A[tidx][fidx])>0:
                # print("Adding the edge:{}-->{}",fidx,tidx)
                act_dag.add_arc(fidx,tidx)
    metric_dict["act_dag"]=act_dag

    #Computing the shd with est dag
    est_dag = metric_dict["est_dag"]
    shd = est_dag.shd(act_dag)
    metric_dict["shd"]=shd

    #Computing the SHD for the oracle est dag
    oracle_est_dag = metric_dict["oracle_est_dag"]
    oracle_shd = oracle_est_dag.shd(act_dag)
    metric_dict["oracle_shd"]=oracle_shd

    #Computing the shd for the oracle igsp dag
    if "igsp_dag" in metric_dict:
        igsp_est_dag = metric_dict["igsp_dag"]
        igsp_shd = igsp_est_dag.shd(act_dag)
        metric_dict["igsp_shd"]=igsp_shd
    
    return metric_dict

def compute_target_jaccard_sim(intv_args_dict,metric_dict):
    '''
    #Assumption: This assumes that the component is atomic
    right now.
    '''
    similarity_list = []
    oracle_similarity_list = []
    for comp in intv_args_dict.keys():    
        #Computing the similarity for each component
        if comp=="obs":
            actual_tgt=set([])
        else:
            actual_tgt = set([int(comp)])
        est_tgt = set(intv_args_dict[comp]["est_tgt"])
        oracle_est_tgt = set(intv_args_dict[comp]["oracle_est_tgt"])

        if comp=="obs" and len(est_tgt.union(actual_tgt))==0:
            js=1.0
            oracle_js = 1.0
        else:
            js = len(est_tgt.intersection(actual_tgt))\
                        /len(est_tgt.union(actual_tgt))
            oracle_js = len(oracle_est_tgt.intersection(actual_tgt))\
                        /len(oracle_est_tgt.union(actual_tgt))
        similarity_list.append(js)
        oracle_similarity_list.append(oracle_js)
    
    avg_js = np.mean(similarity_list)
    avg_oracle_js = np.mean(oracle_similarity_list)
    metric_dict["avg_js"]=avg_js
    metric_dict["avg_oracle_js"]=avg_oracle_js

    return metric_dict

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj,cd.DAG):
            #Converting the DAG objects to adjmatrix 
            #BEWARE, this adj matrix is in different form 
            #(so use DAG.from_amat to recover graph)
            return dict(
                        amat=obj.to_amat()[0].tolist(),
                        nodes=obj.to_amat()[1],
            )
        return super(NpEncoder, self).default(obj)

def pickle_experiment_result_json(expt_args,intv_args_dict,metric_dict):
    '''
    '''
    #Now we are ready to save the results
    experiment_dict = dict(
                        expt_args=expt_args,
                        intv_args_dict=intv_args_dict,
                        metric_dict=metric_dict
    )

    write_fname = "{}/exp_{}.json".format(
                                    expt_args["save_dir"],
                                    expt_args["exp_name"]
    )
    print("Writing the results to: ",write_fname)
    with open(write_fname,"w") as whandle:
        json.dump(experiment_dict,whandle,cls=NpEncoder,indent=4)


#PARLLEL EXPERIMENT RUNNER
def jobber(all_expt_config,save_dir,num_parallel_calls):
    '''
    '''
    #First of all we have to generate all possible experiments
    flatten_args_key = []
    flatten_args_val = []
    for key,val in all_expt_config.items():
        flatten_args_key.append(key)
        flatten_args_val.append(val)
    
    #Getting all the porblem configs
    problem_configs = list(it.product(*flatten_args_val))
    #Now generating all the experimetns arg
    all_expt_args = []

    expt_args_list = []
    for cidx,config in enumerate(problem_configs):
        config_dict = {
            key:val for key,val in zip(flatten_args_key,config)
        }

        args = dict(
                save_dir=save_dir,
                exp_name="{}".format(cidx),
                num_nodes = config_dict["num_nodes"],
                obs_noise_mean = config_dict["obs_noise_mean"],
                obs_noise_var = config_dict["obs_noise_var"],
                max_edge_strength = config_dict["max_edge_strength"],
                graph_sparsity_method = config_dict["graph_sparsity_method"],
                adj_dense_prop = config_dict["adj_dense_prop"],
                num_parents = config_dict["num_parents"],
                new_noise_mean = config_dict["new_noise_mean"],
                mix_samples = config_dict["sample_size"],
                stage2_samples = config_dict["sample_size"],
                gmm_tol=config_dict["gmm_tol"],
                intv_type=config_dict["intv_type"],
                new_noise_sigma=config_dict["new_noise_sigma"],
        )
        if config_dict["intv_targets"]=="all":
            args["intv_targets"]=list(range(config_dict["num_nodes"]))
        else:
            raise NotImplementedError
        expt_args_list.append(args)
        
        
        #If we want to run sequentially    
        # intv_args_dict,metric_dict = run_mixture_disentangle(args)
        # print("=================================================")
        # print("\n\n\n\n\n\n")
    
    #Running the experiment parallely
    with mp.Pool(num_parallel_calls) as p:
        p.map(run_mixture_disentangle,expt_args_list)
    print("Completed the whole experiment!")

    

if __name__=="__main__":
    # Graphs Related Parameters
    all_expt_config = dict(
        #Graph related parameters
        run_list = list(range(10)), #for random runs with same config, needed?
        num_nodes = [6,],
        max_edge_strength = [1.0,],
        graph_sparsity_method=["adj_dense_prop",],#[adj_dense_prop, use num_parents]
        num_parents = [None],
        adj_dense_prop = [0.1,0.2,0.4,0.6,0.8,0.9,0.95,1.0],
        obs_noise_mean = [0.0],
        obs_noise_var = [1.0],
        #Intervnetion related related parameretrs
        new_noise_mean= [1.0],
        intv_targets = ["all"],
        intv_type = ["do"], #hard,do,soft
        new_noise_sigma = [0.0],#[0.1,1.0,2.0,8.0],
        #Sample and other statistical parameters
        sample_size = [2**idx for idx in range(10,18)],
        gmm_tol = [1e-3], #1e-3 default #10000,5000,1000 for large nodes
    )


    save_dir="all_expt_logs/expt_logs_11.05.24-sparsity_n6"
    pathlib.Path(save_dir).mkdir(parents=True,exist_ok=True)
    jobber(all_expt_config,save_dir,num_parallel_calls=64)
    
    