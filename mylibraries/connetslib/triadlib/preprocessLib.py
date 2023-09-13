# Al momento per evolving graph, directed

import networkx as nx
from collections import defaultdict
import pickle
import argparse
import json
import numpy as np
import os

from datetime import datetime
import time
def monitor_elapsed_time(func):
    def wrapper(*args, **kwargs):
        print(datetime.now(), 'Starting', func.__name__)
        t_start = time.time()
        ret = func(*args, **kwargs)
        print(datetime.now(),'Completed',func.__name__ ,'in ', time.time()-t_start)
        return ret
    return wrapper


@monitor_elapsed_time
def make_mapping_user_id(original_digraph,path):
    
    sorted_nodes = [n[0] for n in sorted(original_digraph.degree, key = lambda x: (x[1], x[0]))]

    #Creo il mapping da utente a id numerico
    #print(datetime.now(),"Creating: mappings")
    m = {v: i for i, v in enumerate(sorted_nodes)} #questo è il mapping_user_to_id

    if path:
        with open(path + "_mapping_user_to_id.json", "w") as f:
            f.write(json.dumps(m , indent=4))

    #E il suo viceversa, da indice a utente
    m2 = {i:v for v,i in m.items()}

    if path:
        with open(path + "_mapping_id_to_user.json", "w") as f:
            f.write(json.dumps(m2, indent=4))
    
    return m,m2


@monitor_elapsed_time    
def make_timestamp_map(original_digraph,path, timestamp_key = "timestamp"):
     #Possiamo risparmiare spazio e rendere più veloci i confronti se usiamo dei numeri al posto dei datetime
    timestamps = set()

    for e in original_digraph.edges(data=True):
        #print(e)
        timestamps.add(e[2][timestamp_key])
        #break

    print("NOTE: timestamps will start from 1")
    timestamp_to_index = { t: k+1 for k, t in enumerate(list(sorted(timestamps)))}
    
    if path:
        with open(path + "_timestamps_map.pkl","wb+") as f:
            pickle.dump(timestamp_to_index,f)
        
    return timestamp_to_index


@monitor_elapsed_time
def make_mapped_graph(original_digraph, mapping_user_id, timestamp_to_index,path):
    
    ## MAPPED GRAPH
    #print(datetime.now(),"Create mapped_graph")
    mapped_graph = nx.DiGraph()
    m = mapping_user_id
    for e in original_digraph.edges(data=True):
        src, dest, t = e
        # use maps
        src = m[src]
        dest = m[dest]
        t = t['timestamp']
        mapped_graph.add_edge(src,dest, t =  timestamp_to_index[t])
        #break
        
    if path:
        nx.write_gpickle(mapped_graph, path + "_mapped.pkl")
    
    return mapped_graph

@monitor_elapsed_time
def get_neighborhoods(mapped_graph,path):
     #Neighboorhoods per generate_triads.py
    neighborhoods = { v: mapped_graph.succ[v].keys()|mapped_graph.pred[v].keys() for v in mapped_graph }
    neighborhoods_list = { v: sorted(list(s)) for v,s in neighborhoods.items() }
    
    if path:
        # with open(path +"_neighborhoods.pkl","wb+") as f:
        #     pickle.dump(neighborhoods,f)
        with open(path +"_neighborhoods_lists.pkl","wb+") as f:
            pickle.dump(neighborhoods_list,f)
    
    return neighborhoods

@monitor_elapsed_time    
def calc_pagerank_elite(mapped_graph,path):
    
    #print(datetime.now(),"Main: Pageranks")
    pageranks = nx.pagerank(mapped_graph)
    # Mi serve sapere chi sono coloro nella top 1% -> posso trovare il 99-percentile e usarlo come soglia
    pr_thr = np.percentile([ p for p in pageranks.values()],99)

    elite_pageranks = defaultdict(int)

    for n,pr in pageranks.items():
        if pr>= pr_thr:
            elite_pageranks[n] = 1
            
    if path:
        with open(path +"_elite_pageranks.pkl","wb+") as f:
            pickle.dump(elite_pageranks,f)
        
    return elite_pageranks


@monitor_elapsed_time
def calc_outdegree_elite(mapped_graph,path):
    
    #print(datetime.now(),"Main: outdegrees")
    
    outdegrees = nx.out_degree_centrality(mapped_graph)
    outdeg_thr = np.percentile([ p for p in outdegrees.values()],99)
    elite_outdegrees = defaultdict(int)

    for n,outdeg in outdegrees.items():
        if outdeg >= outdeg_thr:
            elite_outdegrees[n] = 1
            
    if path:
        with open(path +"_elite_outdegrees.pkl","wb+") as f:
            pickle.dump(elite_outdegrees,f)
        
    return elite_outdegrees


@monitor_elapsed_time
def preprocess_snapshot(original_digraph, path, graph_name, mapping_user_id, timestamp_to_index, is_mapped = False):
    
    ### fai una sottocartella all'interno di path per lo snapshot e i file aggiuntivi    
    path = path+"/"+graph_name
    
    try:
        os.mkdir(path)
    except OSError:
        print ("Creation of the directory %s failed" % path)
    else:
        print ("Successfully created the directory %s " % path)    

    
    print(nx.info(original_digraph))

    path = path+"/"+graph_name
    
    ## grafo snapshot senza usernames e timestamp
    if is_mapped:
        mapped_graph = original_digraph
        nx.write_gpickle(mapped_graph, path + "_mapped.pkl")
    else: 
        mapped_graph = make_mapped_graph(original_digraph, mapping_user_id, timestamp_to_index, path)

    #Neighboorhoods e altre strutture per velocizzare  i passi successivi come generate_triads.py
    get_neighborhoods(mapped_graph,path)

    calc_pagerank_elite(mapped_graph,path)
    
    calc_outdegree_elite(mapped_graph,path)
        
    return path
    

@monitor_elapsed_time
def generate_snapshot(G, current_T_limit, timestamp_key = "timestamp"):
    
    G_snapshot = nx.DiGraph() 
    print(datetime.now(), "t <= ",current_T_limit)
    
    for e in G.edges(data=True):
        src, dest, t = e
        t = t[timestamp_key]
        ## prima di inserire controlla la data dell'edge
        if t <= current_T_limit:
            G_snapshot.add_edge(src,dest,timestamp = t)

    print(datetime.now(),"Snapshot till",current_T_limit,"ord/size:",G_snapshot.order(), G_snapshot.size())
    
    return G_snapshot


def make_folder(dirname):
    
    import os
    import errno

    print("Creating:", dirname)

    try:
        os.mkdir(dirname)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        print("Folder:",dirname,"already exists, we are going on")
        print()
        pass

#     return dirname

@monitor_elapsed_time
def new_generate_snapshot(G, params, current_T_limit = None):
    
    # params["SNAPSHOTS_PATH"] = f"./{params['DATA_FOLDER']}/{params['DATASET_NAME']}/" # Attenzione allo "/" finale, altrimenti non scrive nella sottocartella
    params["SNAPSHOTS_PATH"] = f"{params['DATA_FOLDER']}/{params['DATASET_NAME']}/" # Attenzione allo "/" finale, altrimenti non scrive nella sottocartella

    make_folder(params["DATA_FOLDER"])
    make_folder(params["SNAPSHOTS_PATH"])

    # TODO SE CI SONO GIA FILE NON SERVE GENERARE - puoi controllare qua o direttamente nelle funzioni
    
    mapping_user_id, mapping_id_user  = make_mapping_user_id(G,path=params["SNAPSHOTS_PATH"])
    timestamp_map = make_timestamp_map(G,path=params["SNAPSHOTS_PATH"], timestamp_key = params["TIMESTAMP_KEY"] )
    
    for k in ["_mapping_user_to_id.json","_mapping_id_to_user.json","_timestamps_map.pkl"]:
        params[f"path{k.split('.')[0]}"] = params["SNAPSHOTS_PATH"] + k
        
    if current_T_limit is None:
        current_T_limit = max(timestamp_map)
        params["current_T_limit"] = current_T_limit
        print("timestamp limit was not defined, we use all data so limit is:", current_T_limit)
        print()
    else:
        params["current_T_limit"] = current_T_limit 
    
    params["GRAPH_NAME"] = f"{params['DATASET_NAME']}_{current_T_limit}"
    print(params["GRAPH_NAME"])
    print()
   
    ## Snapshot fino a current_T_limit

    G_snapshot = generate_snapshot(G, current_T_limit, timestamp_key = params["TIMESTAMP_KEY"])
    
    
    


    ## preprocessing prima di fare scrittura per risparmiare spazio in RAM

    path = preprocess_snapshot(original_digraph=G_snapshot,
                                        path=params["SNAPSHOTS_PATH"],
                                        graph_name = params["GRAPH_NAME"],
                                        mapping_user_id=mapping_user_id,
                                        timestamp_to_index=timestamp_map,
                                        is_mapped = False)
    
    params["base_prefix_for_snapshot_data"] = path

    
    #### Store original snapshot (requested for stored triads usage)
    params["path_original_snapshot"] = path + "_original_snapshot.pkl"
    nx.write_gpickle(G_snapshot, params["path_original_snapshot"])

    
    params["number_of_nodes"] = G_snapshot.number_of_nodes()
    params["number_of_edges"] = G_snapshot.number_of_edges()
    
    for k in ["_mapped.pkl","_neighborhoods_lists.pkl", "_elite_pageranks.pkl", "_elite_outdegrees.pkl"]: # "_neighborhoods.pkl",
        params[f"path{k.split('.')[0]}"] = path + k
    
    import json

    
    with open(os.path.join(path + "_MY_SNAP_SETTINGS.json") , "w+") as f:
        json.dump(params,f, indent=4, default=lambda o: '<not serializable>') 
    
    generate_sparse_matrix_for_snapshot(params)
    
    with open(os.path.join(path + "_MY_SNAP_SETTINGS.json") , "w+") as f:
        json.dump(params,f, indent=4, default=lambda o: '<not serializable>') 
    
    return path + "_MY_SNAP_SETTINGS.json"


def generate_sparse_matrix_for_snapshot(params):
    
    import networkx as nx
    import ctypes
    import scipy.sparse
    import pickle
    import argparse

    graph_name = params["base_prefix_for_snapshot_data"]

    print("Graph name without extension:",graph_name)
    #G = nx.read_gpickle("../snapshots/snap-2017-11-23/snap-2017-11-23_mapped.pkl")
    print("Reading graph:", graph_name +"_mapped.pkl")
    G = nx.read_gpickle(graph_name +"_mapped.pkl")

    print("Converting to Matrix")
    #data_adj = nx.to_scipy_sparse_matrix(G,dtype=ctypes.c_int,weight="t")

    coo_sparse = nx.to_scipy_sparse_matrix(G,dtype=ctypes.c_int,weight="t", format="coo")
    print("Get the new indices")
    #The rows and columns are ordered according to the nodes in nodelist.
    #If nodelist is None, then the ordering is produced by G.nodes().
    node_in_matrix = { n:i for i,n in enumerate(G.nodes())}

    matrix_to_node = {n:i for i,n in node_in_matrix.items()}

    sparse_matrix_csr = coo_sparse.tocsr()

    #sparse_matrix_csc = coo_sparse.tocsc()


    #print("MB:",(data_adj.data.nbytes + data_adj.indptr.nbytes + data_adj.indices.nbytes)/(1024*1024))

    print("MB:",(sparse_matrix_csr.data.nbytes + sparse_matrix_csr.indptr.nbytes + sparse_matrix_csr.indices.nbytes)/(1024*1024))
    #print("MB:",(sparse_matrix_csc.data.nbytes + sparse_matrix_csc.indptr.nbytes + sparse_matrix_csc.indices.nbytes)/(1024*1024))


    print("Dropping graph variable")
    del G

    print("Saving Matrix")

    #scipy.sparse.save_npz(graph_name + '_sparse_matrix.npz', data_adj)

    #scipy.sparse.save_npz(graph_name + '_sparse_matrix_coo.npz', coo_sparse)
    scipy.sparse.save_npz(graph_name + '_sparse_matrix_csr.npz', sparse_matrix_csr)
    #scipy.sparse.save_npz(graph_name + '_sparse_matrix_csc.npz', sparse_matrix_csc)


    print("Saving new indices")

    with open(graph_name + "_node_in_matrix.pkl","wb+") as f:
        pickle.dump(node_in_matrix,f)

    with open(graph_name + "_matrix_to_node.pkl","wb+") as f:
        pickle.dump(matrix_to_node,f)
        
        
    for k in ['_sparse_matrix_csr.npz',"_node_in_matrix.pkl","_matrix_to_node.pkl" ]: #'_sparse_matrix_coo.npz',
        params[f"path{k.split('.')[0]}"] = params["base_prefix_for_snapshot_data"] + k