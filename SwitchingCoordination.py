import numpy as np
#import pylab as pl
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx

# for colormaps of the nodes in the network
from palettable.cmocean.diverging import Balance_20 as CMap # Curl_20 # Delta_20 # Balance_20
# from palettable.colorbrewer.diverging import RdBu_10 as CMap # RdBu_10 # RdYlGn_10
CMap_MPL = CMap.get_mpl_colormap()

# for saving animation
import imageio
from matplotlib.animation import PillowWriter
from matplotlib.animation import FFMpegWriter

def InitParams(N: int = 3,
               couplingStrength = 1.0, noiseStd = 0.1, switchingRate = 1.0,
               refTime = 1.0, dt = 0.1, simTime = 100.0, outTime = 1.0,
               avgFrequency = 0.0, stdFrequency = 0.0,
               networkShowType = 'movingRing',
               spatialMovement = False, speed = 0.01,
               linkThreshold = 1.0, L = 5.0, boundary_condition = 'periodic', # 'periodic', 'reflective', 'none'
               networkType = 'spatial_metric', # 'spatial_metric', # spatial_angular, angular_random, angular_proportion, angular_disproportion
               fSteepness = 2.0, fTransition = 0.2,
               writeFile = False,
               nTraceHistory : int = 10,
               showAnimation = False, saveAnimation = False, animationFileName = 'animation.mp4'):

    ''' Initialize parameter dictionary'''

    params=dict()
    params['N']=N # number of agents
    params['couplingStrength']=couplingStrength # alignment strength
    params['noiseStd']=noiseStd
    params['noiseAmplitude']=noiseStd*np.sqrt(dt) # noise strength 
    params['switchingRate']=switchingRate # rate of randomly switching neighbors
    params['refTime']=refTime # refractory time -> "blind time" after switching
    params['avgFrequency']=avgFrequency # for Kuramoto model -> average eigenfrequency of agents
    params['stdFrequency']=stdFrequency # for Kuramoto model -> std. deviation of eigenfrequencies
    params['dt']=dt # numerical time step
    params['simTime']=simTime # simulation time in time units
    params['simSteps']=int(simTime/dt) # number simulation steps
    params['outTime']=outTime; # time intervall between outputs saved to outData
    params['outStep']=int(outTime/dt) # time steps between outputs to outData
    params['writeFile']=writeFile # write results to file
    params['showAnimation']=showAnimation # bool: show animation of the graph + phase of the agents
    params['saveAnimation']=saveAnimation # bool: save animation as animationFileName 
    params['animationFileName']=animationFileName # string: name of the animation file
    params['networkShowType']=networkShowType # string: 'movingRing' or 'clock' 
    params['spatialMovement']=spatialMovement # bool: if True, agents move on a 2D plane
    params['speed']=speed # speed of agents if spatialMovement=True
    params['networkType']=networkType # string: 'spatial_metric', 'angular_random', 'angular_proportion', 'angular_disproportion'
    params['linkThreshold']=linkThreshold # threshold for spatial metric network
    params['nTraceHist'] = nTraceHistory # number of history steps to show in the animation
    params['fSteepness'] = fSteepness # steepness for logistic probability of switching (if negative: preference to aligned neighbors)
    params['fTransition'] = fTransition # transition point factor for logistic probability of switching (gets multiplied by pi)
    params['L'] = L # environment size
    params['boundary_condition'] = boundary_condition # Boundary condition of the environment 'periodic', 'reflective', 'none'
    return params


def InitData(params):
    ''' Initialize dictionaries keeping the simulation data'''
    
    data=dict()


    # phase/polar angle or heading of the agent
    data['phi']=2*np.pi*np.random.random(params['N'])
    # set Kuramoto frequencies -> if avg and std set to 0 then all omega=0 -> XY model, simple directional alignment
    if(params['avgFrequency']!=0 or params['stdFrequency']!=0):
        data['omega']=np.random.normal(loc=params['avgFrequency'],scale=params['stdFrequency'],size=params['N']) 
        if(data['omega'].min()<0):
            print('Warning: negative omega values!!! Check if this is intended.')
    else:
        data['omega']=np.zeros(params['N'])

    # index of the neighbor which the agent pays attention to,  each agent has only one neighbor 
    data['neighbor']=np.int32(np.zeros(params['N']))
    # initial random assignment of a neighbor
    nArray=np.arange(params['N'])
    for i in range(params['N']):
        data['neighbor'][i]=np.random.choice(np.delete(nArray,i))
    
    # timer keeping track of refractory time. If timer>0 refractory phase -> no coupling.
    data['timer']=-1*np.ones(params['N'])
    # array keeping track of coupling strength: If timer>0 coupling=0, else coupling=params['couplingStrength']
    data['coupling']=np.ones(params['N'])


    # initialize the history of the positions for showing the trace in the animation
    data['traceHist']=np.zeros((params['nTraceHist'],params['N'],2))*np.nan


    # initialize outData dict for saving only every n-th time step 
    outData=dict()
    outData['t']=[0]
    outData['phi']=[2*np.pi*np.random.random(params['N'])]
    outData['neighbor']=[data['neighbor']]
    outData['timer']=[-params['dt']*np.ones(params['N'])]
    outData['order']=[np.abs(np.mean(np.exp(1j * data['phi'])))]
    

    # if spatial movement for collective motion is True add pos to agents
    if(params['spatialMovement']):
        data['pos']=(params['L']/4)*np.random.random((params['N'],2)) + np.ones((params['N'],2))*params['L']*3/8

        # data['pos']=(params['linkThreshold'])*np.random.random((params['N'],2)) + np.ones((params['N'],2))*(params['L'] - params['linkThreshold'])/2


        #data['pos']=params['L']*np.random.random((params['N'],2))
        outData['pos']=[data['pos']]

        # chose initial neighbors under linkThreshold constraint using spatial network G (but random in terms of heading direction)
        G = make_spatial_metric_network(data=data,params=params)
        for i in range(params['N']):
            nArray = np.array(list(G.neighbors(i)))

            # if nArray is empty add the agent itself to the neighbor list
            if(len(nArray)==0):
                nArray = np.array([i])
            
            data['neighbor'][i]=np.random.choice(nArray)

        outData['neighbor']=[data['neighbor']]
        
        undirected_interaction_G = make_network_from_neighbors(neighbor_list=data['neighbor'], N=params['N'], convert_to_undirected=True)
        outData['clusternumber'] = [calculate_cluster_order_parameter(undirected_interaction_G)]

    return data,outData

def UpdatePhiTimer(phi,omega,timer,neighbor,coupling,N,dt,noiseAmplitude):
    ''' update the phi according to the Euler scheme, update timer'''
    noise = np.random.normal(loc=0,scale=noiseAmplitude,size=N)
    dphi  = (omega+coupling*np.sin(phi[neighbor] - phi))*dt + noise

    phi+= dphi
    #print('--------------')
    #print(timer)
    timer-=dt
    #print(timer)
    #print('--------------')
    phi=np.mod(phi,2*np.pi) 

    return phi,timer

def RouletteWheel(agent,phi,N,fSteepness,fTransition):
    nArray=np.arange(N)
    neighborArray=np.delete(nArray,agent)
    probabilities=np.zeros(N)

    # calculate delta phi for each neighbor of current agent
    for i,val in enumerate(neighborArray):
        # calculate probability with logistic function
        transitionpoint=fTransition*np.pi # factor for moving the transition point of the logistic function
        deltaphi=np.abs(phi[agent]-phi[val]) % np.pi
        probabilities[val] = 1/(1+np.exp(-fSteepness*(deltaphi-transitionpoint)))
   
    # normalize probabilities to sum up to one
    probabilities = probabilities / np.sum(probabilities)    

    epsilon = 1e-14
    condition_on_norm = np.abs(1.0-np.sum(probabilities)) < epsilon
    assert condition_on_norm, "Probabilities do not sum up to ~1" 

    # create the vektor for the Roulette Wheel Selection
    edges = np.cumsum(probabilities)
    rand_num = np.random.rand()
    neighbor = np.argmax(edges > rand_num)

    return neighbor


def RouletteWheel_from_potential_network(agent,phi,neighborArray,N,fSteepness,fTransition):
    ''' Returns only ONE neighbor based on the roulette wheel selection among the neighbors defined by the Spatial graph (excluding itself) -> NO self-loop'''
    # make a warning if neighborArray is empty
    if(len(neighborArray)==0):
        # print("Warning: neighborArray is empty for agent ", agent, flush=True)
        neighborArray = [agent]

    probabilities=np.zeros(N)

    # calculate delta phi for each neighbor of current agent
    for i,val in enumerate(neighborArray):
        # calculate probability with logistic function
        transitionpoint=fTransition*np.pi # factor for moving the transition point of the logistic function
        deltaphi=np.abs(phi[agent]-(phi[val]))
        #if(deltaphi-deltaphi%np.pi):
        #    deltaphi = np.pi-deltaphi%np.pi
        deltaphi=np.abs((deltaphi-deltaphi%np.pi!=0)*np.pi-deltaphi%np.pi) # equivalent to the 2 lines above
        #deltaphi=np.abs(phi[agent]-phi[val]) % np.pi ---> This was not correct :/
        probabilities[val] = 1/(1+np.exp(-fSteepness*(deltaphi-transitionpoint)))
   
    # normalize probabilities to sum up to one
    probabilities = probabilities / np.sum(probabilities)    

    epsilon = 1e-10
    condition_on_norm = np.abs(1.0-np.sum(probabilities)) < epsilon
    assert condition_on_norm, "Probabilities do not sum up to ~1"

    # create the vektor for the Roulette Wheel Selection
    edges = np.cumsum(probabilities)
    rand_num = np.random.rand()
    neighbor = np.argmax(edges > rand_num)

    return neighbor

def UpdateNetwork(phi,neighbor,timer,coupling,switchingRate,dt,N,refTime,couplingStrength,fSteepness,fTransition):
    ''' update network, rewire with probability switchingRate*dt per time step per agent'''
    rnd=np.random.random(size=N)
    switchArray=rnd<(switchingRate*dt)
    
    nArray=np.arange(N)
    for idx in np.where(switchArray)[0]:
        # pick ONE neighbor excluding itself -> NO self-loop
        #neighbor[idx]=np.random.choice(np.delete(nArray,np.int32([neighbor[idx],idx])))
        # pick ONE neighbor including itself -> self-loop
        #neighbor[idx]=np.random.choice(nArray)
        # pick ONE neighbor based on roulette wheel selection (excluding itself) -> NO self-loop
        neighbor[idx]=RouletteWheel(idx,phi,N,fSteepness,fTransition)

    timer[switchArray]=refTime
    coupling[:]=couplingStrength
    coupling[timer>0]=0.0

    return neighbor,timer,coupling

def UpdateNetwork_spatial(phi,neighbor,timer,coupling,switchingRate,dt,N,refTime,couplingStrength,need_a_switch,spatial_graph,fSteepness,fTransition,network_type='spatial_metric'):
    ''' update network, based on the spatial network (as the potential interaction network), rewire with probability switchingRate*dt per time step per agent'''
    rnd=np.random.random(size=N)
    switchArray=rnd<(switchingRate*dt)

    # check if agents are alone: their neighbors is themselves, then they need a switch (if possible)
    self_looped_agents = neighbor==np.arange(N)
    need_a_switch = np.logical_or(need_a_switch, self_looped_agents)
    # agents who passed the BC need a switch regardless of the random number
    switchArray = np.logical_or(switchArray, need_a_switch) 

    nArray=np.arange(N)
    for idx in np.where(switchArray)[0]:
        
        if(network_type=='spatial_metric'):
            # pick ONE neighbor excluding itself -> NO self-loop
            nArray = np.array(list(spatial_graph.neighbors(idx)))

            # if nArray is empty add the agent itself to the neighbor list
            if(len(nArray)==0):
                nArray = np.array([idx])
            
            # print("networks for agent ", idx, " : ", nArray, flush=True)
            neighbor[idx]=np.random.choice(nArray)
            # neighbor[idx]=np.random.choice(np.delete(nArray,np.int32([neighbor[idx],idx])))

        elif(network_type=='spatial_angular'):
            nArray = np.array(list(spatial_graph.neighbors(idx)))
            # print("networks for agent ", idx, " : ", nArray, flush=True)
            # pick ONE neighbor based on roulette wheel selection among the neighbors defined by the Spatial graph (excluding itself) -> NO self-loop
            neighbor[idx]=RouletteWheel_from_potential_network(idx,phi,nArray,N,fSteepness,fTransition)

        elif(network_type=='angular_proportion'):
            # pick ONE neighbor based on roulette wheel selection (excluding itself) -> NO self-loop
            neighbor[idx]=RouletteWheel(idx,phi,N,fSteepness,fTransition)

    timer[switchArray]=refTime
    coupling[:]=couplingStrength
    coupling[timer>0]=0.0

    return neighbor,timer,coupling


def UpdateOutData(params,data,outData,t):
    ''' append current time step results to outData '''

    outData['t'].append(t)
    outData['phi'].append(np.copy(data['phi']))
    outData['neighbor'].append(np.copy(data['neighbor'][:]))
    outData['timer'].append(np.copy(data['timer'][:]))
    outData['order'].append(np.abs(np.mean(np.exp(1j * data['phi']))))

    if(params['spatialMovement']):
        undirected_interaction_G = make_network_from_neighbors(neighbor_list=data['neighbor'], N=params['N'], convert_to_undirected=True)
        outData['clusternumber'].append(calculate_cluster_order_parameter(undirected_interaction_G))

    return 

def GenerateOutputString(params):
    ''' generate output string for file name'''
    nameString='N'+str(params['N'])+'-K'+str(params['couplingStrength'])+'-R'+str(params['switchingRate'])+'-sigma'+str(params['noiseStd'])
    nameString=nameString.replace('.','_')

    return nameString

def SaveResultsToFile(params,outData):
    ''' save order parameter to file '''
    saveDict={k:v for k,v in outData.items() if k in ['t','order'] }
    df=pd.DataFrame.from_dict(saveDict)
    nameString=GenerateOutputString(params)
    df.to_csv('results/order-'+nameString+'.csv',index=False) 
    
    return

def make_network(params, data):
    ''' return the network given the list of neighbors '''
    # make an empty directed graph with N nodes
    G = nx.empty_graph(n=params['N'],create_using=nx.DiGraph())

    # go through the neighbor list and add the corresponding directed edge
    for node_i in range(G.number_of_nodes()):
        G.add_edge(*(node_i, data['neighbor'][node_i]))
    
    return G

def draw_animation_frame(params, data, outData, time, fig, ax, node_pos, save_animation, 
                         moviewriter, animation_frame_list=[], with_clocks=False): #
    ''' return a figure showing the agents on a graph + their states as color '''
    
    # TODO : we don't need to make the network from null everytime, but rather update the links. check how this speed up the code execution
    G = make_network(params=params, data=data)


    node_colors = []
    font_colors = []
    node_pos_ring = []
    for tmp_data in data['phi']:
        # node_pos_ring.append()
        node_colors.append(CMap_MPL(tmp_data/np.pi/2))
        font_colors.append(CMap_MPL(1-(tmp_data/np.pi/2)))

    ax.clear()
    
    if(with_clocks):
        nx.draw_networkx(G, with_labels=True, pos=node_pos, node_color=node_colors, 
                font_color="mintcream", font_weight="bold", verticalalignment='center_baseline',
                node_size=1000,
                connectionstyle='arc3, rad=0.2')
        node_pos_np = np.array(list(node_pos.values()))
        arm_length = 0.1
        arm_delta_pos = arm_length*np.vstack((np.cos(data['phi']), np.sin(data['phi']))).T
        clock_tip_points = node_pos_np + arm_delta_pos

        # Connect the points with lines
        for i in range(0, len(clock_tip_points)):
            ax.plot([node_pos_np[i,0], clock_tip_points[i,0]], [node_pos_np[i,1], clock_tip_points[i,1]], color='r')

        plt.plot()
    else:
        nx.draw_networkx(G, with_labels=True, pos=node_pos, node_color=node_colors, 
                font_color="mintcream", font_weight="bold", verticalalignment='center_baseline')

    ax.set_title("Frame %d:  order: %f"%((time+1),(outData['order'][-1])))
    plt.pause(0.0001)


    ## # when using ImageIO library
    # if(save_animation):
    #     image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    #     image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    #     animation_frame_list.append(image)
    # else:
    #     animation_frame_list = []
    
    if(save_animation):
        moviewriter.grab_frame()

    # return animation_frame_list


def draw_animation_frame_ring_phase(params, data, outData, time, fig, ax, save_animation, 
                         moviewriter): #
    ''' return a figure showing the agents' phase on a ring + graph '''
    G = make_network(params=params, data=data)

    node_colors = []

    # color nodes based on their phase
    for tmp_data in data['phi']:
        node_colors.append(CMap_MPL(tmp_data/np.pi/2))
    
    # color nodes based on their frequency
    # range_omega = np.max(data['omega']) - np.min(data['omega'])
    # # range_omega = params['stdFrequency']
    # min_omega = np.min(data['omega'])
    # avg_omega = np.median(data['omega'])
    # for tmp_data in data['omega']:
    #     # node_colors.append(CMap_MPL((tmp_data-min_omega)/(range_omega)))
    #     node_colors.append(CMap_MPL(2*(tmp_data-avg_omega)/(range_omega)))

    ax.clear()
    
    # make the position of the nodes on the ring based on their phase
    ring_rad = 1
    node_pos_list = []
    for tmp_data in data['phi']:
        node_pos_list.append(np.array(ring_rad*[np.cos(tmp_data), np.sin(tmp_data)]))
    

    # draw the network
    nx.draw_networkx(G, with_labels=False, pos=node_pos_list, 
            connectionstyle='arc3, rad=0.5', node_size=0, alpha=0.25, arrowstyle='-|>', arrowsize=5)
    
    # add the node outward arms on the ring
    node_pos_np = np.array(node_pos_list)
    arm_length = 0.1
    arm_delta_pos = arm_length*np.vstack((np.cos(data['phi']), np.sin(data['phi']))).T
    clock_tip_points = node_pos_np + arm_delta_pos

     # Connect the points with lines
    for i in range(0, len(clock_tip_points)):
        ax.plot([node_pos_np[i,0], clock_tip_points[i,0]], [node_pos_np[i,1], clock_tip_points[i,1]], color=node_colors[i])

    phase_ring = plt.Circle((0., 0.), ring_rad, edgecolor='black', facecolor='none', linewidth=0.2)
    ax.add_artist(phase_ring)

    plt.xlim([-1.2*ring_rad, 1.2*ring_rad])
    plt.ylim([-1.2*ring_rad, 1.2*ring_rad])
    

    ax.set_title("Frame %d:  order: %f"%((time+1),(outData['order'][-1])))
    plt.axis("off")

    plt.plot()
    plt.pause(0.0001)

    ## # when using ImageIO library
    # if(save_animation):
    #     image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    #     image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    #     animation_frame_list.append(image)
    # else:
    #     animation_frame_list = []
    
    if(save_animation):
        moviewriter.grab_frame()

    # return animation_frame_list

# draw animation of spatial collective motion
def draw_animation_frame_spatial(params, data, outData, time, fig, ax, save_animation, G, interaction_G,
                                    moviewriter,title_vals,link_over_boundary=np.nan):
    ''' return a figure showing the agents' movin in 2D space following their heading bias and the network of their interactions '''

    node_colors = []
    font_colors = []
    node_pos_ring = []
    for tmp_data in data['phi']:
        # node_pos_ring.append()
        node_colors.append(CMap_MPL(tmp_data/np.pi/2))
        font_colors.append(CMap_MPL(1-(tmp_data/np.pi/2)))

    ax.clear()

    # ## Alternative way of drawing the network using NetworkX
    # nx.draw_networkx(G, with_labels=True, pos=data['pos'], node_color=node_colors, 
    #             font_color="mintcream", font_weight="bold", verticalalignment='center_baseline')

    # nx.draw_networkx(G, with_labels=True, pos=data['pos'], node_color=node_colors, 
    #         font_color="mintcream", font_weight="bold", verticalalignment='center_baseline',
    #         node_size=1000)
    

    # draw the points using scatter plot of the traceHistory for the length of the trace history
    for i in range(0, params['nTraceHist']):
        node_size_tmp = 5.0*(1-np.double(i/params['nTraceHist']))
        ax.scatter(data['traceHist'][i,:,0], data['traceHist'][i,:,1], c='chocolate', s=node_size_tmp, alpha=1.0)
            

    # draw the nodes using data['pos']
    ax.scatter(data['pos'][:,0], data['pos'][:,1], c=node_colors, s=50)

    
    # draw the edges using the G network: this is the potential network defined by the distance in space
    # use matplotlib plot function to draw the edges
    # for i in range(0, params['N']):
    #    for j in range(i+1, params['N']):
    #        if(G.has_edge(i,j)):
    #            ax.plot([data['pos'][i,0], data['pos'][j,0]], [data['pos'][i,1], data['pos'][j,1]], 
    #                    color='slategrey', linewidth=0.5, alpha=0.25)
    
    # use matplotlib plot function to draw the edges, trying to consider the periodic links 
    for i in range(0, params['N']):
        for j in range(i+1, params['N']):
            dist_vec = data['pos'][j]-data['pos'][i]

            if(G.has_edge(i,j) and (np.linalg.norm(dist_vec) < params['linkThreshold'])):
                ax.plot([data['pos'][i,0], data['pos'][j,0]], [data['pos'][i,1], data['pos'][j,1]], 
                        color='slategrey', linewidth=0.5, alpha=0.25)
                
            if(G.has_edge(i,j) and (np.linalg.norm(dist_vec) > params['linkThreshold'])):
                # we assume that we already have checked the periodic distance, and that the link is definitely over the boundary
                ind_xy_flip = np.abs(dist_vec)>params['L']/2

                dist_vec[ind_xy_flip] = -np.sign(dist_vec[ind_xy_flip])*(params['L'] - np.abs(dist_vec[ind_xy_flip])) # equivalent!
                # dist_vec[ind_xy_flip] = -np.sign(dist_vec[ind_xy_flip])*(params['L']) + dist_vec[ind_xy_flip] # 

                virtual_neighbor = data['pos'][i]+dist_vec
                virtual_focal = data['pos'][j]-dist_vec

                # draw a line between the virual neighbor and actual focal
                ax.plot([data['pos'][i,0], virtual_neighbor[0]], [data['pos'][i,1], virtual_neighbor[1]], 
                        color='slategrey', linewidth=0.5, alpha=0.25)
                # draw a line between the actual neighbor and virtual focal
                ax.plot([data['pos'][j,0], virtual_focal[0]], [data['pos'][j,1], virtual_focal[1]],
                        color='slategrey', linewidth=0.5, alpha=0.25)

                
                # ax.arrow(data['pos'][j,0]-dist_vec[0], data['pos'][j,1]-dist_vec[1], dist_vec[0], dist_vec[1],
                #         head_width=0.0, head_length=0.15, fc='slategrey',linewidth=0.5, length_includes_head=True, alpha=0.25)

                # ax.arrow(data['pos'][i,0], data['pos'][i,1], dist_vec[0], dist_vec[1],
                #         head_width=0.1, head_length=0.15, fc='k', ec='k', length_includes_head=True, alpha=0.25)

    # nx.draw_networkx_edges(G, pos=data['pos'], ax=ax, edge_color='slategrey', width=0.5, alpha=0.25)

    # draw the edges using the data['neighbor']: this the actual network of interaction between agents
    for i in range(0, params['N']):
        # draw lines for the links
        # ax.plot([data['pos'][i,0], data['pos'][data['neighbor'][i],0]], [data['pos'][i,1], data['pos'][data['neighbor'][i],1]], 
        #         color='k', linewidth=1.0, alpha=0.5)

        # draw arrow instead of line for actual interaction links
        dist_vec = data['pos'][data['neighbor'][i]]-data['pos'][i]

        if (np.linalg.norm(dist_vec) < params['linkThreshold']):
            # first (only) draw the link for those within linkThreshold
            ax.arrow(data['pos'][i,0], data['pos'][i,1], dist_vec[0], dist_vec[1],
                        head_width=0.1, head_length=0.15, fc='k', ec='k', length_includes_head=True, alpha=0.3)

        # if the link is over the boundary, plot a vector to the outside of the boundary
        elif(link_over_boundary[i]): # and (np.linalg.norm(dist_vec) > params['linkThreshold'])
            ind_xy_flip = np.abs(dist_vec)>params['L']/2

            # |    o--------x1-------->o<=======x2==|=====o  : (vec) x1 - (vec) x2 = (vec) L
            # |    o--------x2-------->o<=======x1==|=====o  : (vec) - x1 + (vec) x2 = (vec) L
            # x2 = x1 +- L = -sign(x1)*L + x1

            # dist_vec[ind_xy_flip] = -np.sign(dist_vec[ind_xy_flip])*(params['L'] - np.abs(dist_vec[ind_xy_flip])) # equivalent!
            dist_vec[ind_xy_flip] = -np.sign(dist_vec[ind_xy_flip])*(params['L']) + dist_vec[ind_xy_flip] # 

            # now draw the arrow TO the focal agent from outside of the arena (periodic virtual neighbor)
            ax.arrow(data['pos'][data['neighbor'][i],0]-dist_vec[0], data['pos'][data['neighbor'][i],1]-dist_vec[1], dist_vec[0], dist_vec[1],
                        head_width=0.1, head_length=0.15, fc='k', ec='k', length_includes_head=True, alpha=0.3)
            # draw the arrow from the neighbor to the virtual focal agent
            ax.arrow(data['pos'][i,0], data['pos'][i,1], dist_vec[0], dist_vec[1],
                        head_width=0.1, head_length=0.15, fc='k', ec='k', length_includes_head=True, alpha=0.3)
    
    # # Draw a tail for each node
    # node_pos_np = np.array(data['pos'])
    # tail_length = 0.02
    # # make the tail point
    # tail_delta_pos = tail_length*np.vstack((np.cos(data['phi']), np.sin(data['phi']))).T
    # tail_tip_points = node_pos_np - tail_delta_pos

    # # Connect the points with lines
    # for i in range(0, len(tail_tip_points)):
    #     ax.plot([node_pos_np[i,0], tail_tip_points[i,0]], [node_pos_np[i,1], tail_tip_points[i,1]], color='r')

    ax.set_title("Frame %d: order: %1.3f, clust. G: %1.2f, (%1.2f)" %((time+1),title_vals[0],title_vals[1],title_vals[2]))

    ax.axis('on')
    plt.axis("on")
    if(params['boundary_condition']!= 'none'):
        ax.set_xlim([0, params['L']])
        ax.set_ylim([0, params['L']])
    plt.pause(0.0001)
    
    if(save_animation):
        moviewriter.grab_frame()

    return fig, ax

def reachingfinalorder(params, orderlist=[], min_finalorder=30):
    zaehler = 0
    t_finalorder = params['simTime']
    final_order = np.mean(orderlist[-20:])
    for i_zahl, zahl in enumerate(orderlist[:]):
            if zahl >= (final_order-np.var(orderlist[:])):
                zaehler += 1
                if zaehler == min_finalorder:
                    t_finalorder = i_zahl-min_finalorder
                    break
            elif (zahl < final_order, zaehler > 0):
                zaehler = 0

    return t_finalorder

# take a step in the direction of phi
def take_step(pos, phi, dt, speed,L,boundary_condition='none'):
    ''' take a step in the direction of phi using speed and dt'''
    pos += speed*dt*np.vstack((np.cos(phi), np.sin(phi))).T

    # if(boundary_condition=='none'):
    #     # need_a_switch = phi>np.inf # False condition for all agents

    # el
    if(boundary_condition=='periodic'):
        ## Periodic boundary condition
        # use the remnant of the position divided by L to get the position inside the environment
        pos = np.mod(pos, L)

        

    elif(boundary_condition=='reflective'):
        ## Reflective boundary condition
        print("Warning: reflective boundary condition is not implemented yet!")
        # if the position is out of the unit square, reflect it back
        # pos[pos>1.0] = 2.0 - pos[pos>1.0]
        # pos[pos<0.0] = -pos[pos<0.0]


    return pos

def make_spatial_metric_network(data,params):
    ''' update the network of agents based on their position, if it is below a link_threshold'''
    # make an empty graph
    G = nx.Graph()

    # assign position to the nodes of the graph based on data pos
    G.add_nodes_from(range(params['N']), pos=data['pos'])

    # add edges between nodes if they are below the link_threshold
    for node_i in range(params['N']):
        for node_j in range(node_i+1, params['N']):
            if(np.linalg.norm(data['pos'][node_i] - data['pos'][node_j]) < params['linkThreshold']):
                G.add_edge(*(node_i, node_j))

            # add edges for neighbors below the linkThreshold considering the periodic boundary condition
            if(params['boundary_condition']=='periodic'):
                dist_vec = np.abs(data['pos'][node_i]-data['pos'][node_j])
                dist_vec[dist_vec>(params['L']/2)] = params['L'] - dist_vec[dist_vec>(params['L']/2)] # this is always a positive number

                # TO SARA: this was unnecessary. We used it for the visuallizaiton, where the direction was important. Here only the abs. value is relevant.
                # ind_xy_flip = np.abs(dist_vec)>params['L']/2
                # dist_vec[ind_xy_flip] = np.mod(-(params['L'] - dist_vec[ind_xy_flip]),params['L'])

                if (np.linalg.norm(dist_vec) < params['linkThreshold']):
                    G.add_edge(*(node_i, node_j))

    return G

def calculate_cluster_order_parameter(G):
    ''' calculate the cluster order parameter of the graph G'''
    # get the connected components of the graph G
    connected_components = list(nx.connected_components(G))
    #calculate the sum of squared number of particles within each component
    sum_squared = 0
    for component in connected_components:
        sum_squared += len(component)**2

    # divide the sum by the total number of nodes
    cluster_order_parameter = np.sqrt(sum_squared)/G.number_of_nodes()
    return cluster_order_parameter


def make_network_from_neighbors(neighbor_list, N, convert_to_undirected=False):
    ''' make a network from the neighbor list'''

    # make an empty directed graph with N nodes
    if(convert_to_undirected):
        G = nx.empty_graph(n=N)
    else:
        G = nx.empty_graph(n=N,create_using=nx.DiGraph())

    # go through the neighbor list and add the corresponding directed edge
    for node_i in range(G.number_of_nodes()):
        G.add_edge(*(node_i, neighbor_list[node_i]))
        if(convert_to_undirected):
            G.add_edge(*(neighbor_list[node_i], node_i))
    
    return G

def SingleSimulation(params,data=[]):
    ''' perform a single run'''
    
    #initialize data if not passed to function
    if(len(data)==0):
        data,outData=InitData(params)

    # make fig and ax for animation visualization
    if(params['showAnimation']):
        fig, ax = plt.subplots(figsize=(5,5));
        G_null = nx.empty_graph(n=params['N'],create_using=nx.DiGraph())
        node_pos = nx.circular_layout(G_null)

        moviewriter = np.nan # just a dummy variable for initialization
        if(params['saveAnimation']):
            # if animationFileName contains .gif, use ImageIO library to save gif
            if(params['animationFileName'].find('.gif')!=-1):
                animation_frame_list = []
                moviewriter = PillowWriter(fps=30) # for GIFs
                moviewriter.setup(fig, 'animations/' + params['animationFileName'], dpi=100)
            elif(params['animationFileName'].find('.mp4')!=-1):
                moviewriter = FFMpegWriter(fps=15)  # for MP4
                moviewriter.setup(fig, f'animations/' + params['animationFileName'], dpi=100)


    # perform time loop for simple Euler scheme integration
    for t in range(1,params['simSteps']):

        data['phi'], data['timer'] = UpdatePhiTimer(data['phi'],data['omega'],data['timer'],data['neighbor'],data['coupling'],
                                                    params['N'],params['dt'],params['noiseAmplitude'])

        # if spatial movement update the position of agents using their phi as heading direction
        # check if the agent is out of the square and need to switch their neighbor
        if(params['spatialMovement']):
            need_a_switch = np.zeros(params['N'], dtype=bool)           # Reset the need_a_switch array every time step
            link_over_boundary = np.zeros(params['N'], dtype=bool)      # Reset the link_over_boundary array every time step: This is only for visualization purposes

            data['pos'] = take_step(pos = data['pos'], phi = data['phi'], dt = params['dt'], 
                                                   speed = params['speed'], L = params['L'], boundary_condition=params['boundary_condition'])
            
            # if two agents that are neighbors of the interaction network are more distant than the link threshold, they need a switch
            for i in range(params['N']):
                dist_vec = np.abs(data['pos'][i] - data['pos'][data['neighbor'][i]])

                # check if components of dist_vec are greater than L/2, make them periodic
                if(params['boundary_condition']=='periodic'):
                    dist_vec[dist_vec>(params['L']/2)] = params['L'] - dist_vec[dist_vec>(params['L']/2)] # this is always a positive number
                    # Now the dist_vec is periodic! 
                    link_over_boundary[i] = True

                if(np.linalg.norm(dist_vec) > params['linkThreshold']):
                    need_a_switch[i] = True
            
            # print("need a switch: ", need_a_switch, flush=True)
            if(params['showAnimation']):
                # add the current pos to the trace history and delete the oldest trace history 
                data['traceHist'][-1,:,:] = data['pos']
                data['traceHist'] = np.roll(data['traceHist'], 1, axis=0)


        # if the network type is spatial_metric update the network based on the position of agents
        if(params['networkType'].find('spatial')!=-1):
            G = make_spatial_metric_network(data=data,params=params)

            # calculate the cluster order parameter of the graph G using Eq. 12 of Ref [Zhao, et al 2021]
            component_val_G = calculate_cluster_order_parameter(G)
            data['neighbor'],data['timer'],data['coupling'] = UpdateNetwork_spatial(data['phi'],data['neighbor'],data['timer'],data['coupling'],
                                                                        params['switchingRate'],params['dt'],params['N'],
                                                                        params['refTime'],params['couplingStrength'], need_a_switch=need_a_switch,
                                                                        spatial_graph=G, fSteepness=params['fSteepness'],fTransition=params['fTransition'],network_type=params['networkType'])

            # make a graph from the neighbor list (interaction graph)
            # CHECK! connected component does not work for directed graphs, so we convert it to undirected graph
            interaction_G = make_network_from_neighbors(neighbor_list=data['neighbor'], N=params['N'], convert_to_undirected=False)
            undirected_interaction_G = make_network_from_neighbors(neighbor_list=data['neighbor'], N=params['N'], convert_to_undirected=True)


            # calculate the cluster order parameter of the interaction graph  using Eq. 12 of Ref [Zhao, et al 2021]
            component_val_interaction_G = calculate_cluster_order_parameter(undirected_interaction_G)


        elif(params['networkType']=='angular_proportion'): 
            data['neighbor'],data['timer'],data['coupling'] = UpdateNetwork(data['phi'],data['neighbor'],data['timer'],data['coupling'],
                                                                        params['switchingRate'],params['dt'],params['N'],
                                                                        params['refTime'],params['couplingStrength'], params['fSteepness'],params['fTransition'])
        
        #write outData 
        if(t % params['outStep']==0):
            UpdateOutData(params,data,outData,t*params['dt'])

            

        if(params['showAnimation']):
            # for the collective motion with spatial movement
            if(params['networkType'].find('spatial')!=-1):
                order = np.abs(np.mean(np.exp(1j * data['phi'])))
                draw_animation_frame_spatial(params=params, data=data, outData=outData, time=t,
                                            fig=fig, ax=ax, save_animation=params['saveAnimation'],  G=G, interaction_G=interaction_G,
                                            moviewriter=moviewriter, title_vals=[order, component_val_G, component_val_interaction_G],link_over_boundary=link_over_boundary)
            
            # elif params network type contains 'angular' then draw the animation of the network with the agents' phase
            elif(params['networkType'].find('angular')!=-1):
                if(params['networkShowType']=='movingRing'):
                    draw_animation_frame_ring_phase(params=params, data=data, outData=outData, time=t,
                                                fig=fig, ax=ax, save_animation=params['saveAnimation'], 
                                                moviewriter=moviewriter)
                    
                elif(params['networkShowType']=='clock'):   
                    draw_animation_frame(params=params, data=data, outData=outData, time=t, 
                                                                ax=ax, fig=fig, node_pos=node_pos, 
                                                                save_animation=params['saveAnimation'],
                                                                moviewriter=moviewriter,
                                                                with_clocks=True)
            
            

    # save results to file
    if(params['writeFile']):
        SaveResultsToFile(params,outData)
        
    if(params['saveAnimation']):
        # # save to a GIF file using ImageIO library
        # imageio.mimsave("animation.gif", animation_frame_list, duration=0.5)

        # # save to a GIF file using ImageIO library
        # imageio.mimsave("animation.mp4", animation_frame_list, fps=25, codec="libx264")

        # # save the video Writer using matplotlib.animation
        moviewriter.finish()

    return outData, data



