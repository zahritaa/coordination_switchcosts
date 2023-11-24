import pandas as pd
#import matplotlib.pyplot as plt
import SwitchingCoordination as sc
import numpy as np
from tqdm import tqdm
import pickle
import sys

import multiprocessing 
num_processors = multiprocessing.cpu_count()
print("Number of processors: ", num_processors)

from joblib import Parallel, delayed

# Define the method to run multi_processing on MonteCarlo Rep.s
multi_processing_method = "joblib" # "single_process" # "joblib" # "multiprocessing" # 

from importlib import reload
reload(sc)
# %load_ext autoreload
# %autoreload 2

import time

# change line 107 to  data['pos']=(params['L'])*np.random.random((params['N'],2))

# simulation parameters -- 'x' for the scanning parameters
N = 100
ref_time = 1.0
switching_rate_vector = np.logspace(-4.0, 1.0, 21)
switchingrates = np.concatenate((np.logspace(-4,1,21),np.logspace(-4,1,21),np.linspace(0.01,0.01,11),np.linspace(0.1,0.1,11)),axis=0)
densities = np.concatenate((np.linspace(16,16,21),np.linspace(32,32,21),np.linspace(5.3,32,11),np.linspace(5.3,32,11)),axis=0)
scanvals = switchingrates
sim_list = list(zip(switchingrates, [0.0]*len(scanvals), [0.0]*len(scanvals), densities, ['spatial_metric']*len(scanvals)))+list(zip(switchingrates, [5.0]*len(scanvals), [0.75]*len(scanvals), densities, ['spatial_angular']*len(scanvals)))+list(zip(switchingrates, [-5.0]*len(scanvals), [0.25]*len(scanvals), densities, ['spatial_angular']*len(scanvals)))

switching_rate = sim_list[int(sys.argv[1])][0]
noise_std = 0.5
sim_time = 150
avg_frequency = 0.0
std_frequency= 0.0
spatial_movement = True
write_file = False # True
coupling_strength = 2.0
network_type = sim_list[int(sys.argv[1])][4] # 'spatial_metric', # spatial_angular, angular_random, angular_proportion, angular_disproportion
self_link = 'off' # change before running!!!
b = sim_list[int(sys.argv[1])][1]
c = sim_list[int(sys.argv[1])][2]
density = sim_list[int(sys.argv[1])][3]
L = np.sqrt(N/density)
RWS = f'b{b}_c{c}' # Roulette Wheel Selection

# monte-carlo parameters
n_mc_reps = 320

folderName = f'sim'
# file name
if(network_type.find('spatial')!=-1):
    fileName = 'scan_'+str(network_type)+'_nMC-'+str(n_mc_reps)+'_N-'+str(N)+'_d'+str(density)+'_sr-'+str(switching_rate)+'_noise-'+str(noise_std)+'_K-'+str(coupling_strength)+'_wa-'+str(avg_frequency)+'_ws-'+str(std_frequency)+'_sl-'+str(self_link)+'_RWS-'+str(RWS)+'full-area'

# make an empty out_data to fill in later
out_data = {} 
out_data_list = []
param_scan_dict = {"switchingRate": {"range": switching_rate_vector, "log": True},#np.logspace(-4.0, 1.0, 20)
                "N": {"range": np.linspace(100,100,1), "log": False}} #np.linspace(3,30,10)
#param_scan_dict = {"couplingStrength": {"range": np.linspace(0.0, 1.0, 10), "log": False},
#                   "N": {"range": np.linspace(5,5,1), "log": False}}
default_N = -1
default_switchingRate = 1.0
default_couplingStrength = -1.0
# initialize a parameter dictionary
params = sc.InitParams(N=default_N,switchingRate=default_switchingRate,couplingStrength=coupling_strength,
                            refTime=ref_time, simTime=sim_time, noiseStd=noise_std,
                            avgFrequency=avg_frequency, stdFrequency=std_frequency, spatialMovement=spatial_movement, L=L, networkType=network_type, fSteepness=b,fTransition=c, writeFile=write_file,showAnimation=False)
# save the params merged with param_scan_dict to a pickle file for later use
params_to_save = params | param_scan_dict
with open(f'{folderName}/{fileName}_params.pkl', 'wb') as handle:
    pickle.dump(params_to_save, handle, protocol=pickle.HIGHEST_PROTOCOL)
start = time.perf_counter()
# first parameters loop
#for i_switching_rate, switching_rate in enumerate(tqdm(param_scan_dict['switchingRate']['range'])):

params['switchingRate'] = switching_rate

# second parameter loop
for i_N, N in enumerate(param_scan_dict['N']['range']):
    params['N'] = int(N)
    ## # Normal for loop with single processor
    if multi_processing_method=="single_process":
        for mc_iter in np.arange(n_mc_reps):
        
    #       # perform a single simulation
            out_data_tmp = sc.SingleSimulation(params)
            # make an empty (temporary) dict to put all the data (params + output) into
            tmp_dict = {}
            # put the params into the dict
            for key, val in dict.items(params):
                tmp_dict[key] = val
            # put the time and order arrays into the dict
            tmp_dict['t'] = np.array(out_data_tmp[0]['t'])
            tmp_dict['order'] = np.array(out_data_tmp[0]['order'])
            # tmp_dict["mc_iter"] = mc_iter
            if(params['spatialMovement']):
                tmp_dict['clusternumber'] = np.array(out_data_tmp[0]['clusternumber'])
            # append it to the list 
            out_data_list.append(tmp_dict)
    # # parallel processing using "multiprocessing" 
    elif multi_processing_method=="multiprocessing":
        with multiprocessing.Pool(processes=num_processors) as pool:
            results = pool.map(sc.SingleSimulation, [params] * n_mc_reps)
            for out_data_tmp in results:
            # out_data_tmp = results
                tmp_dict = {}
        #         # put the params into the dict
                for key, val in dict.items(params):
                    tmp_dict[key] = val
                # put the time and order arrays into the dict
                tmp_dict['t'] = np.array(out_data_tmp[0]['t'])
                tmp_dict['order'] = np.array(out_data_tmp[0]['order'])
                # tmp_dict["mc_iter"] = mc_iter
                if(params['spatialMovement']):
                    tmp_dict['clusternumber'] = np.array(out_data_tmp[0]['clusternumber'])
                # append it to the list 
                out_data_list.append(tmp_dict)
    ## # parallel processing using "joblib"
    elif multi_processing_method=="joblib":
        results = Parallel(n_jobs=num_processors)(delayed(sc.SingleSimulation)(params) for _ in range(n_mc_reps))
        for out_data_tmp in results:
            tmp_dict = {}
    #         # put the params into the dict
            for key, val in dict.items(params):
                tmp_dict[key] = val
            # put the time and order arrays into the dict
            tmp_dict['t'] = np.array(out_data_tmp[0]['t'])
            tmp_dict['order'] = np.array(out_data_tmp[0]['order'])
                #tmp_dict["mc_iter"] = mc_iter
            if(params['spatialMovement']):
                tmp_dict['clusternumber'] = np.array(out_data_tmp[0]['clusternumber'])
            # append it to the list 
            out_data_list.append(tmp_dict)
finish = time.perf_counter()
print(f'Finished in {round(finish-start, 2)} second(s)')
# convert it to a pd.df
out_data_df = pd.DataFrame(out_data_list)

# add the relaxation time, mean for polarization and clusternumber for each single simulation
out_data_df['tFinalOrder'] = [sc.reachingfinalorder(params, x) for x in out_data_df['order']]
out_data_df['meanOrder'] = [np.mean(x[-30:]) for x in out_data_df['order'][:]]
if(params['spatialMovement']):
    out_data_df['MeanClusternumber'] = [np.mean(x[-30:]) for x in out_data_df['clusternumber'][:]]

#save DataFrame to pickle file
out_data_df.to_pickle(f'{folderName}/{fileName}_data.pkl')