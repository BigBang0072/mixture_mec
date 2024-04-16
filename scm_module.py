import numpy as np
import pdb
from pprint import pprint
import itertools as it
import matplotlib.pyplot as plt

class GaussianSCM:
    '''
    '''
    def __init__(self,args):
        '''
        '''
        self.debug=args["debug_mode"] if "debug" in args["debug_mode"] else False
        if self.debug:
            print("==================================")
            print("Generating SCM")
        #Paramters for true underlying SCM
        self.noise_mu = np.array(args["noise_mean_list"],dtype=np.float32)
        self.noise_D = np.diag(args["noise_sigma_list"])
        self.noise_D_list= np.diag(self.noise_D)
        self.dim = self.noise_D.shape[0]
        #This adjacency matrix is lower traingular
        self.A = np.array(args["adj_mat"])
        assert np.allclose(self.A, np.tril(self.A)),"adj not lower triangluar"
        #Creating the B matrix from this A
        self.B = np.linalg.inv(np.eye(self.dim)-self.A)
        #TODO: Later we can allow for the random permutation and keep the P for mapping
         
    def _generate_sample(self,num_samples,Ai,noise_Di,noise_mui):
        '''
        Here we will do ancestral sampling using the diagnoal adjacency matrix
        Should we do it ourselves in brute force manner? Use some library or 
        atleast parallelize later
        '''
        #Generating the independent noise for each var and each sample
        standard_noise = np.random.randn(num_samples,self.dim)
        noise = standard_noise*np.diag(noise_Di) + noise_mui
        #Now we are ready to generate all the samples
        Bi = np.linalg.inv(np.eye(self.dim)-Ai)
        X = np.matmul(Bi,noise.T).T #to keep the sample x dim shape

        #Generating the covariance matrix to compare later
        Si = np.matmul(np.matmul(Bi,noise_Di),Bi.T)
        x_mui = np.matmul(Bi,noise_mui)
        return X,Si,x_mui
    
    def generate_sample_with_atomic_intervention(self,num_samples,intv_args):
        '''
        interv_args: 
            inode : intervened node
            soft_vec : the vector that will signify the soft internvetion
        '''
        if intv_args==None or intv_args["intv_type"]=="obs":
            Ai = self.A.copy()
            noise_Di = self.noise_D.copy()
            noise_mui = self.noise_mu.copy()
        elif intv_args["intv_type"]=="do":
            Ai = self.A.copy()
            Ai[intv_args["inode"],:]=0
            #Updating the noise variance for this node
            noise_Di = self.noise_D.copy()
            noise_Di[intv_args["inode"],intv_args["inode"]]=0
            #Updating the mean of this node too
            noise_mui = self.noise_mu.copy()
            noise_mui[intv_args["inode"]]=intv_args["new_mui"] #the constant it sets to
        elif intv_args["intv_type"]=="hard":
            Ai = self.A.copy()
            Ai[intv_args["inode"],:]=0
            #Updating the noise variance for this node
            noise_Di = self.noise_D.copy()
            noise_Di[intv_args["inode"],intv_args["inode"]]=intv_args["new_sigmai"]
            #Updating the mean of this node too
            noise_mui = self.noise_mu.copy()
            noise_mui[intv_args["inode"]]=intv_args["new_mui"]
        elif intv_args["intv_type"]=="soft":
            raise NotImplementedError()
            #Otherwise we will perform a soft internvetion
        else:
            raise NotImplementedError()

        #Now finally generating the sample
        X,Si,x_mui = self._generate_sample(num_samples=num_samples,
                                  Ai=Ai,
                                  noise_Di=noise_Di,
                                  noise_mui=noise_mui)
        #Also generating the covariance for the safe keeping
        true_params=dict(
                        Si = Si,
                        mui = x_mui
        )
        if self.debug:
            print("==================================")
            pprint("Generating samples for:")
            pprint(intv_args)
            pprint("True params:")
            pprint(true_params)

        return X,true_params


if __name__=="__main__":
    args={}
    num_nodes=2
    args["noise_mean_list"]=[0.0,0.0]
    args["noise_sigma_list"]=[1.0,1.0]
    args["adj_mat"]=np.array([
                    [0,0],
                    [1,0]
    ])
    #Creating the SCM 
    gSCM = GaussianSCM(args)
        



    