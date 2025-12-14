## Learning Mixtures of Unknown Causal Interventions

This repository contains the code and experiments accompanying the NeurIPS 2024 paper. This work studies causal discovery when interventional data are noisy and arise from mixtures of intended and unintended interventions. Focusing on linear Gaussian structural equation models with unknown causal graphs, it shows that both do- and soft-interventions generate sufficient distributional diversity to disentangle mixed data efficiently. The required sample complexity decreases as interventions induce larger changes in the affected variables. Despite intervention noise, the causal graph remains identifiable up to its interventional Markov equivalence class, matching guarantees from ideal interventions.


[Arxiv Link](https://arxiv.org/abs/2411.00213) [NeurIPS Link](https://proceedings.neurips.cc/paper_files/paper/2024/hash/1dcee1cd6890ab7fcdf173ec10526da9-Abstract-Conference.html) [Poster](https://github.com/BigBang0072/mixture_mec/blob/comp_sel/assets/NeurIPS2024Poster.pdf)
  

### Running the Code
1. Checkout the correct branch with ```` git checkout comp_sel ````
2. First install all the dependencies using ````pip install -r requirements.txt````
3. Run the experiment job:
    1. ````python mixture_solver.py --simulation ```` : This will run the experiments on the simulation dataset.
    2. ````python mixture_solver.py --sachs ````: This will run the experiments on the simulated mixture of intervention on the real-world SACHS dataset. 

This will run the experiments in the default configuration. To change different parameters of the experiments, here is the complete list:


**Simulation Experiment**: In the ``run_simulation_experiments`` function one can tune the following parameters:
1. ``run_list``: list of random experiment names. Each name will spawn a random experiment based on the other parameters, which can be used to estimate the algorithm's variance. 
2.  ``num_nodes``: Number of nodes in the random graph. We can give this as a list.
3.  ``max_edge_strength``: The maximum edge strength in the Linear SCM of the underlying causal model. We can give this as a list.
4. ``graph_sparsity_method``: The default implemented method is adj_dense_prop, which makes the adjacency matrix of the graph sparse based on the adj_dense_prop parameter given below.
5. ``adj_dense_prop``: The fraction of the adjacency matrix that is non-zero. The value can be any real number between 0 and 1. This can be a list as well.
6.   ``noise_type``: The default is "gaussian" exogenous noise used in the SCM of the causal graph
7. ``obs_noise_mean``: The mean of exogenous noise. This can be a list.
8. `obs_noise_var``: The variance of exogenous noise. This can be a list.
9. ``new_noise_mean``: The mean of the new noise distribution after intervention. This can be a list.
10. ``intv_targets``: The number of nodes to intervene. Currently, we provide two functionalities, "half" and "full", that intervene randomly on half of the nodes and all the nodes, respectively. This can be a list. 
11. ``intv_type``: The type of intervention we want to perform from the set (hard,do,soft)
12. ``new_noise_var``: The new variance of the noise distribution. This is used for soft and hard interventions. This can be a list
13. ``sample_size``: The number of samples to use in the experiment. This can be a list.
14. ``gmm_tol``: The tolerance to use for the Gaussian mixture model. This can be a list. 
15. ``cutoff_drop_ratio``: The cutoff percentage to use for selecting the number of components as mentioned in the paper. This can be a list

For all the parameters that take a list asan  argument, we first do a cartesian product of all the parameters and spawn a random graph according to the parameters given, and run the experiment.



### Reference

If you use this code, please cite:

```bibtex
@inproceedings{kumar_mixture_neurips_2024,
 author = {Kumar, Abhinav and Shiragur, Kirankumar and Uhler, Caroline},
 booktitle = {Advances in Neural Information Processing Systems},
 doi = {10.52202/079017-0527},
 editor = {A. Globerson and L. Mackey and D. Belgrave and A. Fan and U. Paquet and J. Tomczak and C. Zhang},
 pages = {16538--16568},
 publisher = {Curran Associates, Inc.},
 title = {Learning Mixtures of Unknown Causal Interventions},
 url = {https://proceedings.neurips.cc/paper_files/paper/2024/file/1dcee1cd6890ab7fcdf173ec10526da9-Paper-Conference.pdf},
 volume = {37},
 year = {2024}
}
