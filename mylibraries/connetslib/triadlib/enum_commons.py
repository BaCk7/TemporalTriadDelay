# from . import directed_census_utils as directed_census_utils

def make_folder(dirname):
    
    import os, errno
    
    print("Creating:", dirname)

    try:
        os.mkdir(dirname)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        print("Folder:",dirname,"already exists, we are going on")
        print()
        pass

    


def parallelize_task(params, init_worker, runnable_function, processes = 2, max_subprocesses = 2):
    
    import os
    import pickle
    import math
    
    graph_name = params["base_prefix_for_snapshot_data"]
    processes = processes # Numero di processi worker. sono come nuovi notebook o esecuzioni python - multiprocessing in python fa schifo
    #limitv = max_subprocesses # Si pone come limita a quanti thread fare da suddividere tra i processi 
    
    params["processes"] = processes
    params["max_subprocesses"] = max_subprocesses
    
    directory_path =  graph_name+"_census_list"# + str(datetime.now())
    #print("directory_path:", directory_path)#,"rs",str(random_size))
    params["directory_path"] = directory_path
    make_folder(directory_path)
    
    #print(params)
    
    if "STORE_CLOSED" in params and params["STORE_CLOSED"] == True:
        storage_path = graph_name+"_triads"# + str(datetime.now())
        print("storage_path:", storage_path)#,"rs",str(random_size))
        params["storage_path"] = storage_path
        make_folder(storage_path)
    
    log_folder_path = graph_name+"_logs"# + str(datetime.now())
    print("log_folder_path:", log_folder_path)#,"rs",str(random_size))
    params["log_folder_path"] = log_folder_path
    make_folder(log_folder_path)

    
    len_selected_nodes = params["number_of_nodes"]

    if processes:
        processes = int(processes)
    else:
        #processes is the number of worker processes to use. If processes is None then the number returned by os.cpu_count() is used.
        processes = os.cpu_count()

    print("processes", processes)

    chunk_size = math.ceil(len_selected_nodes / max_subprocesses) #int( len_selected_nodes / limitv)

    print("Chunk size", chunk_size)

    num_chunks =  math.ceil(len_selected_nodes/ chunk_size)

    #num_chunks = limitv
    print("TOTAL CHUNKS ADJUSTED:",num_chunks)
    
    params["num_chunks"] = num_chunks
    params["chunk_size"] = chunk_size
    
    # Store
    path = params["base_prefix_for_snapshot_data"]
    
    params["result_path"]  = path + "_MY_SNAP_SETTINGS.json"
    
    import json, os
    
    with open(os.path.join(path + "_MY_SNAP_SETTINGS.json") , "w+") as f:
        json.dump(params,f, indent=4, default=lambda o: '<not serializable>') 
    
    # Parallelize
    from multiprocessing import Pool
    import time
    from datetime import datetime,timedelta
    
    
    p = Pool(processes=processes, initializer=init_worker, initargs=(params,"test"))#,PR,PR_shape ))

    #num_chunks = limitv # max_subprocesses

    print(datetime.now()," Starting with #Chunks:", max_subprocesses)


    
    #map_results = p.map(runnable_function, zip(range(max_subprocesses)) ) # ##[chunk_size] * num_chunks,

    try:
        # from tqdm import tqdm
        # from tqdm.notebook import tqdm as tqdm
        from tqdm.auto import tqdm

    except ImportError:
        def tqdm(iterator, *args, **kwargs):
            return iterator

    
    map_results = []
    
    with tqdm(total=max_subprocesses) as pbar:
        for i, res in enumerate( p.imap_unordered( runnable_function, zip(range(max_subprocesses)) ) ):
            map_results.append(res)
            pbar.update()

                
    #zip iteratore che date delle liste,
    #crea delle tuple prendendo da ogni lista.
    print(datetime.now(),"MAIN: AGGREGATING RESULTS", datetime.now())
    aggregated_results = map_results[0]

    
    
    ## Reduce the partial solutions
    for r in map_results[1:]:
        for k,obj in r.items():
            for key in obj:
                aggregated_results[k][key] += obj[key]

    return aggregated_results
    

def dummy_init_worker(params,dummy_variable): # dummy variable necessaria altrimenti tratta params come se fossero 18 argomenti diversi
    print("Init worker start")
    print(params)
    
    global var_dict # https://stackoverflow.com/questions/38795826/optimizing-multiprocessing-pool-with-expensive-initialization
    var_dict = params
    
    print("Init worker FINISH")
    


# In[66]:


def dummy_runnable(chunk_index): # ex _metric_censusmap_bigdata(G_selected_nodes) + enumerate_triadic_census(chunk_index)
    print(chunk_index)
    print(var_dict["base_prefix_for_snapshot_data"])
    return {"metrica0":{"my_index":chunk_index[0]}}
    

# aggregated_results = parallelize_task(params, init_worker=dummy_init_worker, runnable_function=dummy_runnable, processes=2, max_subprocesses=4)

# aggregated_results
# # Coppia initworker runnable per census_list
# 
# Initworker deve lavorare con params, dovrebbe esserci tutto ciò che gli serve altrimenti va aggiunto nel main
# 
# Invece runnable ha le funzioni in directed_census_utils da chiamare

   

from datetime import datetime,timedelta

import time

def monitor_elapsed_time(func):
    def wrapper(*args, **kwargs):
        print(datetime.now(), 'Starting', func.__name__)
        t_start = time.time()
        ret = func(*args, **kwargs)
        print(datetime.now(),'Completed',func.__name__ ,'in ', time.time()-t_start)
        return ret
    return wrapper  
    
    

def iterator_union_sorted_array(A,B,skip_to = None):

    i=0
    j=0

    if skip_to:
        #trova primo el > di skip_to
        #print("SKIPTO")
        while skip_to > A[i]:
            i+=1
        while skip_to > B[j]:
            j+=1

    while i < len(A) and j < len(B):
        last = min(A[i],B[j])

        if A[i] == last:
            i += 1
        if B[j] == last:
            j += 1

        yield(last)

    if i < len(A):
        while i < len(A):
            yield A[i]
            i += 1

    if j < len(B):
        while j < len(B):
            yield (B[j])
            j+=1


def iterator_union_sorted_array_helper(A,B,skip_to = None):
    if len(A) == 0:
        #print("A")
        if skip_to:
            return ( n for n in B if n > skip_to)
        else:
            return iter(list(B))

    if len(B) == 0:
        #print("B")
        if skip_to:
            return ( n for n in A if n > skip_to)
        else:
            return iter(list(A))

    return iterator_union_sorted_array(A,B,skip_to)

import scipy.sparse
import networkx as nx

def get_subgraph_from_sparse(sparse_matrix,node_in_matrix,v,u,w):
    
    G = nx.DiGraph()

    pairs = [ (v,u), (v,w), (u,v), (u,w), (w,v), (w,u)]


    for src,target in pairs:
        t = sparse_matrix[node_in_matrix[src],node_in_matrix[target]]
        if t:
            G.add_edge(src,target, t = t)

    return G    
    
    
    

## Storage of closed triads

def load_json(PATH):
    import json 
    with open(PATH,"rb") as f:
        jj = json.load(f) 
    return jj

def load_census_results(SNAP_SETTINGS_PATH):
    
    import connetslib.triadlib.aggregated_functions as agg_funcs
    
    SNAP_SETTINGS = load_json(SNAP_SETTINGS_PATH)
    
    DIR = SNAP_SETTINGS["directory_path"]
    
    aggregated = agg_funcs.aggregate_results(DIR)
    
    return aggregated


def store_triad(vals):
    v,u,w = vals
    return  f'{",".join( [str(el) for el in vals ] )}\n'.encode() # ",".join( [str(el) for el in [v,(u-v),(w-v)] ] ) + "\n" 

def parse_stored_triads(SNAP_SETTINGS_PATH, covert_to_username = True):
    
    import json, os, gzip
    
    SNAP_SETTINGS = load_json(SNAP_SETTINGS_PATH)
    
    user_mapping_path = load_json(SNAP_SETTINGS["path_mapping_id_to_user"])
    
    folder = SNAP_SETTINGS["storage_path"]

    fnames = sorted([ os.path.join(folder,fname)  for fname in os.listdir(folder)  ])#if fname.endswith("csv.gz") 
    
    #print(fnames)
    
    for fname in fnames:
        with gzip.open(fname,"rb") as f:
            for line in f:
                v,u,w = line.rstrip().decode().split(",")
                #print(v,u,w)
                if covert_to_username:
                    v = user_mapping_path[v]
                    u = user_mapping_path[u]
                    w = user_mapping_path[w]
                else:
                    v = int(v)
                    u = int(u) 
                    w = int(w) 
                    
                yield(v,u,w)
                
def get_stored_triads(SNAP_SETTINGS_PATH):
    import networkx as nx
    
    SNAP_SETTINGS = load_json(SNAP_SETTINGS_PATH)

    G = nx.read_gpickle(SNAP_SETTINGS["path_original_snapshot"]) #("karaatetestgraph.gpickle") # nx.read_gpickle(SNAP_SETTINGS["path_mapped"])
    G.remove_edges_from(nx.selfloop_edges(G))
    
    for triad in parse_stored_triads(SNAP_SETTINGS_PATH, covert_to_username=True):
        #v,u,w = triad
        triangle = G.subgraph(triad)
        # print(v,u,w)
       
        yield triangle
                
# def store_triad_gap_encoding(vals):
#     v,u,w = vals
#     return  f'{",".join( [str(el) for el in [v, (u-v), (w-v)] ] )}\n'.encode() 


def get_original_snapshot(SNAP_SETTINGS_PATH):

    import networkx as nx
    
    SNAP_SETTINGS = load_json(SNAP_SETTINGS_PATH)

    G = nx.read_gpickle(SNAP_SETTINGS["path_original_snapshot"]) #("karaatetestgraph.gpickle") # nx.read_gpickle(SNAP_SETTINGS["path_mapped"])
    G.remove_edges_from(nx.selfloop_edges(G))    
    
    return G