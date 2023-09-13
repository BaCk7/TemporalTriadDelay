from multiprocessing import Pool#,RawArray
import time

from collections import Counter, OrderedDict, defaultdict

import networkx as nx

from datetime import datetime

import argparse
import sys
import json

import numpy as np
import os
import pickle

import json

import math

import gzip
#import random

import pickle

import scipy.sparse

#python Triadi/Scripts/census_list.py -p 15 -g snapshots4/snap-2017-04-30/snap-2017-04-30 -lv 300
#17.40

# from datetime import datetime,timedelta

# import time

# def monitor_elapsed_time(func):
#     def wrapper(*args, **kwargs):
#         print(datetime.now(), 'Starting', func.__name__)
#         t_start = time.time()
#         ret = func(*args, **kwargs)
#         print(datetime.now(),'Completed',func.__name__ ,'in ', time.time()-t_start)
#         return ret
#     return wrapper
from .enum_commons import monitor_elapsed_time


#### 0 - 5 open triads 6 casi
#### 6 - 12 closed triads 7 casi
mapping_census_to_baseline = {'003': 13, ####null triad
 '012': 14, #### diadic
 '102': 15, #### diadic
 '021D': 0,  '021U': 3,  '021C': 1,  '111D': 4,  '111U': 2,  '030T': 6,
 '030C': 7, '201': 5,  '120D': 10,  '120U': 8,  '120C': 9,  '210': 11,
 '300': 12
}

####: The integer codes representing each type of triad.
####:
####: Triads that are the same up to symmetry have the same code.
TRICODES = (1, 2, 2, 3, 2, 4, 6, 8, 2, 6, 5, 7, 3, 8, 7, 11, 2, 6, 4, 8, 5, 9,
            9, 13, 6, 10, 9, 14, 7, 14, 12, 15, 2, 5, 6, 7, 6, 9, 10, 14, 4, 9,
            9, 12, 8, 13, 14, 15, 3, 7, 8, 11, 7, 12, 14, 15, 8, 14, 13, 15,
            11, 15, 15, 16)

####: The names of each type of triad.
TRIAD_NAMES = ('003', '012', '102', '021D', '021U', '021C', '111D', '111U',
               '030T', '030C', '201', '120D', '120U', '120C', '210', '300')

####: A dictionary mapping triad code to triad name.
TRICODE_TO_NAME = {i: TRIAD_NAMES[code - 1] for i, code in enumerate(TRICODES)}

def tricode(G, v, u, w):
    """Returns the integer code of the given triad.

    This is some fancy magic that comes from Batagelj and Mrvar's paper. It
    treats each edge joining a pair of ``v``, ``u``, and ``w`` as a bit in
    the binary representation of an integer.

    """
    combos = ((v, u, 1), (u, v, 2), (v, w, 4), (w, v, 8), (u, w, 16),
              (w, u, 32))
    return sum(x for u, v, x in combos if v in G[u])


#@monitor_elapsed_time
def define_triad_order_new(triangle):

    edges_by_time = sorted(triangle.edges().data(), key=lambda x: x[2]['t'])

    #print(edges_by_time)

    a,b = edges_by_time[0][0],edges_by_time[0][1]

    #print("a,b:",a,b)

    b1,c1 = edges_by_time[1][0],edges_by_time[1][1]

    #print("b1,c1", b1,c1)

    # controllo se non ho preso arco reciproco di ab , altrimenti prendi il successivo, che sicuramente coinvolge b e c
    if a == c1 and b == b1:
        b1,c1 = edges_by_time[2][0],edges_by_time[2][1]
        #print("b1,c1 new",b1,c1)


    a_and_b = {a,b}
    b_and_c = {b1,c1}

    b = a_and_b.intersection(b_and_c)

    a = (a_and_b - b).pop()

    c = (b_and_c - b).pop()

    b = b.pop()

    return {"ordered_triad": (a,b,c)}, edges_by_time#, "ordered_edges": edges_by_time}

#@monitor_elapsed_time
def define_triad_order_new_timestamp_key(triangle, timestamp_key ='t'):

    edges_by_time = sorted(triangle.edges().data(), key=lambda x: x[2][timestamp_key])

    #print(edges_by_time)

    a,b = edges_by_time[0][0],edges_by_time[0][1]

    #print("a,b:",a,b)

    b1,c1 = edges_by_time[1][0],edges_by_time[1][1]

    #print("b1,c1", b1,c1)

    # controllo se non ho preso arco reciproco di ab , altrimenti prendi il successivo, che sicuramente coinvolge b e c
    if a == c1 and b == b1:
        b1,c1 = edges_by_time[2][0],edges_by_time[2][1]
        #print("b1,c1 new",b1,c1)


    a_and_b = {a,b}
    b_and_c = {b1,c1}

    b = a_and_b.intersection(b_and_c)

    a = (a_and_b - b).pop()

    c = (b_and_c - b).pop()

    b = b.pop()

    return {"ordered_triad": (a,b,c)}, edges_by_time#, "ordered_edges": edges_by_time} #Mi sa che ho dimenticato di mettere in dict


#@monitor_elapsed_time
def triad_code_pagerank(triad_abc, pageranks):

    pr_bits = "".join([str(pageranks[n]) for n in triad_abc])
    return pr_bits

#@monitor_elapsed_time
def triad_code_outdegree(triad_abc, outdegrees):
    outdeg_bits = "".join([ str(outdegrees[n]) for n in triad_abc])
    return outdeg_bits

def count_for_open(result, pr_bits, outdeg_bits,triangle,triad_abc,edges_by_time):
    result['pageranks_open'][pr_bits] +=1
    result['outdegrees_open'][outdeg_bits] +=1
    ####istances_open[istance_bits] +=1


def analyze_triad_closure(triangle, ordered_triad, edges_by_time):

    a,b,c = ordered_triad
    subgraph = nx.DiGraph()

    subgraph.add_nodes_from(triangle)

    before_closure = None
    closure_state = None

    i = 0
    for e in edges_by_time:
    ####	aggiungi archi fino a che non individui chiusura
        subgraph.add_edge(e[0], e[1])
        i+=1
                ####print("Subgraph edges",list(subgraph.edges()))
                ####code = tricode(subgraph, a, b, c)
                ####triad_type = mapping_census_to_baseline[TRICODE_TO_NAME[code]]
        triad_name = [k for k,v in nx.triadic_census(subgraph).items() if v==1][0]
        triad_type = mapping_census_to_baseline[triad_name]
                ####print(triad_type)
        if triad_type in [0,1,2,3,4,5,13,14,15]:
            before_closure = triad_type
        else:
            closure_state = triad_type
            src = e[0]
            target = e[1]
            closing_time = e[2]['t']
            break
	#### all'uscita so in che stato era prima della chiusura e in cosa si è trasformata quando si è chiusa
	#### devo solo controllare che dopo il link di chiusura, non si sia formato quello inverso.
	####Se così fosse, aggiorno closure state, in quanto mi interessa sapere se a<->c aggiungo al subgraph e rifaccio tricode
	####extra_edge = [e for e in edges_by_time[i:] if e[0]==c and e[1] ==a ]
    extra_edge = triangle.has_edge(target,src)

    if extra_edge:
		####e = extra_edge.pop()
		####print("extra edge",e)
        subgraph.add_edge(target,src)
		####code = tricode(subgraph, a, b, c)
		####triad_type = mapping_census_to_baseline[TRICODE_TO_NAME[code]]
        triad_name = [k for k,v in nx.triadic_census(subgraph).items() if v==1][0]
        triad_type = mapping_census_to_baseline[triad_name]
        closure_state = triad_type

    return {"evolution": (str(before_closure), str(closure_state)), "closing_time":closing_time}


def count_for_closed(result, pr_bits, outdeg_bits,triangle,triad_abc,edges_by_time):
    #edges_by_time = res['ordered_edges']

    res = analyze_triad_closure(triangle, triad_abc, edges_by_time)
    evolution = res["evolution"]
    ###closing_time = result["closing_time"]
    ###closing_dates[closing_time] +=1 ####closing_dates[closing_time.strftime("%d-%m-%Y")]+=1

    evol = "".join(list(evolution))
    #print("EVOLUTION HERE: ------>",evolution, evol,result)
    result["evolutions"][evol] +=1

    result['pageranks_closed'][pr_bits] +=1
    result['outdegrees_closed'][outdeg_bits] +=1
    ####istances_closed[istance_bits] +=1
    a,b,c = triad_abc
    closing_edge_type = str(int(triangle.has_edge(a,c))) + "" + str(int(triangle.has_edge(c,a)))

    result['closing_edge_pagerank'][(pr_bits+closing_edge_type)] +=1
    result['closing_edge_outdegrees'][(outdeg_bits+closing_edge_type)] +=1

    
def init_worker(params,dummy_variable): #,PR, PR_shape):
    
    import pickle
    import copy
    import scipy
    # Using a dictionary is not strictly necessary. You can also
    # use global variables.
    
    # print("Init worker start")
    # print(params)
    
    global var_dict # https://stackoverflow.com/questions/38795826/optimizing-multiprocessing-pool-with-expensive-initialization
    var_dict = copy.deepcopy(params)
    
    
    with open(params["path_elite_pageranks"],"rb" ) as f:
        ####pageranks = pickle.load(f)
        var_dict['elite_pageranks'] = pickle.load(f)

    #### read/generate outdegrees
    with open(params["path_elite_outdegrees"],"rb" ) as f:
        var_dict['elite_outdegrees'] = pickle.load(f)


    # print("Reading files: _sparse_matrix, node_in_matrix")

    var_dict['sparse'] = scipy.sparse.load_npz(params["path_sparse_matrix_csr"])

    with open(params["path_node_in_matrix"],"rb") as f:
        var_dict['nim'] = pickle.load(f)

    #graph_name = graph_name.split(".")[0]
  
    with open(params["path_neighborhoods_lists"],"rb") as f:
        ##print("reading neighborhoods_sorted.pkl")
        var_dict['neighborhoods']  = pickle.load(f)

    # print("Init worker FINISH")
    

#@monitor_elapsed_time
def enumerate_triadic_census(chunk_index):
    
    chunk_index = chunk_index[0] # non serve più la funzione intermedia che spacchetta
    
    import pickle
    
    from datetime import datetime,timedelta

    import time
    import gzip
    import json
    import os
    
    from . import enum_commons
    from . import logging

    dispatcher = {
        0: count_for_open,
        1: count_for_open,
        2: count_for_open,
        3: count_for_open,
        4: count_for_open,
        5: count_for_open,
        6: count_for_closed,
        7: count_for_closed,
        8: count_for_closed,
        9: count_for_closed,
        10: count_for_closed,
        11: count_for_closed,
        12: count_for_closed
    }

    directory_path = var_dict['directory_path']
    chunk_size = var_dict['chunk_size']
    limit = var_dict["max_subprocesses"] #['limit']
    #graph_name = var_dict['graph_name']

    node_in_matrix = var_dict['nim']
    sparse_matrix = var_dict['sparse']

    elite_pageranks = var_dict['elite_pageranks']
    elite_outdegrees = var_dict['elite_outdegrees']
    
    STORE_CLOSED = var_dict["STORE_CLOSED"]

    ## ricavo il range
    start = chunk_index*chunk_size
    stop = start + chunk_size

    neighborhoods_sorted = var_dict['neighborhoods']

    selected_nodes = sorted(list(neighborhoods_sorted.keys()))


    limitv = limit #,limitu,limitw = limit

    step = limitv
    indexes = [ (i*step) + chunk_index for i in range(chunk_size)]
    selected_nodes = [selected_nodes[i] for i in indexes if i < len(selected_nodes)]
    
    
    log_folder_path = var_dict["log_folder_path"]
    LOG_FILE = log_folder_path+ "/" +str(chunk_index)+"_log.txt"
    logging.setup_log_file(LOG_FILE)
    logging.printlog(datetime.now(),"Processing:",chunk_index," - RANGE:",start,stop, "- Nodes are:", selected_nodes )
    
    # print(datetime.now(),"Processing:",chunk_index," - RANGE:",start,stop, "- Nodes are:", selected_nodes )

    census = {name: 0 for name in TRIAD_NAMES}

    ############ All'inizio del census
    pageranks_open = {'000': 0, '001': 0, '010': 0, '011': 0, '100': 0, '101': 0, '110': 0, '111': 0}
    outdegrees_open = {'000': 0, '001': 0, '010': 0, '011': 0, '100': 0, '101': 0, '110': 0, '111': 0}
    ####istances_open = {'000': 0, '001': 0, '010': 0, '011': 0, '100': 0, '101': 0, '110': 0, '111': 0}

    pageranks_closed = {'000': 0, '001': 0, '010': 0, '011': 0, '100': 0, '101': 0, '110': 0, '111': 0}
    outdegrees_closed = {'000': 0, '001': 0, '010': 0, '011': 0, '100': 0, '101': 0, '110': 0, '111': 0}
    ####istances_closed = {'000': 0, '001': 0, '010': 0, '011': 0, '100': 0, '101': 0, '110': 0, '111': 0}

    evolutions = {'06': 0, '010': 0,
                  '16': 0, '17': 0, '19': 0,
                  '28': 0, '29': 0, '211': 0,
                  '36': 0, '38': 0,
                  '49': 0, '410': 0, '411': 0,
                  '511': 0, '512': 0}

    ####closing_dates = defaultdict(int)

    closing_edge_pagerank = {'00001': 0, '00010': 0, '00011': 0, '00101': 0, '00110': 0, '00111': 0, '01001': 0, '01010': 0, '01011': 0, '01101': 0, '01110': 0, '01111': 0, '10001': 0, '10010': 0, '10011': 0, '10101': 0, '10110': 0, '10111': 0, '11001': 0, '11010': 0, '11011': 0, '11101': 0, '11110': 0, '11111': 0}
    closing_edge_outdegrees = {'00001': 0, '00010': 0, '00011': 0, '00101': 0, '00110': 0, '00111': 0, '01001': 0, '01010': 0, '01011': 0, '01101': 0, '01110': 0, '01111': 0, '10001': 0, '10010': 0, '10011': 0, '10101': 0, '10110': 0, '10111': 0, '11001': 0, '11010': 0, '11011': 0, '11101': 0, '11110': 0, '11111': 0}
    ####closing_edge_istance = {'00001': 0, '00010': 0, '00011': 0, '00101': 0, '00110': 0, '00111': 0, '01001': 0, '01010': 0, '01011': 0, '01101': 0, '01110': 0, '01111': 0, '10001': 0, '10010': 0, '10011': 0, '10101': 0, '10110': 0, '10111': 0, '11001': 0, '11010': 0, '11011': 0, '11101': 0, '11110': 0, '11111': 0}

    result = {
            'census': census,
            'evolutions': evolutions,
            'pageranks_open': pageranks_open,
            'outdegrees_open': outdegrees_open,
            ####'istances_open': istances_open,
            'pageranks_closed': pageranks_closed,
            'outdegrees_closed': outdegrees_closed,
            ####'istances_closed': istances_closed,
            ####'closing_dates':closing_dates,
            'closing_edge_pagerank':closing_edge_pagerank,
            'closing_edge_outdegrees':closing_edge_outdegrees####,
            ####'closing_edge_istance':closing_edge_istance
            }

    #buffer_limit = limitw
    #print("Limit ws per line:",limitw)
    
    
    
    
    positive_examples = []

    ## Crea file associato al chunk
    filepath = directory_path+"/"+str(chunk_index)+".json.gz"

    #with open(filepath,"w+") as f:
    with gzip.open(filepath, 'wt+', encoding="utf-8") as zipfile:

        processed = 0
        total = len(selected_nodes)
        t_started = time.time()

        vnbrs = []
        unbrs = []
        for v in selected_nodes:#[:3]:
        # for i, v in tqdm(selected_nodes, total=len(selected_nodes), leave=False):    

            cont = 0
            vnbrs = neighborhoods_sorted[v] ## vicini del seed

            for u in vnbrs:
                if u <= v:
                    continue
                ##neighbors = (vnbrs | set(G.succ[u]) | set(G.pred[u])) - {u, v}
                #neighbors = (vnbrs| neighborhoods_sorted[u]) - {u,v}
                unbrs = neighborhoods_sorted[u]

                buffer = []
                ws = []

                for w in enum_commons.iterator_union_sorted_array_helper(vnbrs,unbrs,skip_to=v):#neighbors:
                    if u < w or (v < w < u and
                                       ##v not in G.pred[w] and
                                       ##v not in G.succ[w]):
                                       v not in neighborhoods_sorted[w]):

                        #ws.app2022-03-03 11:56:15.219817 Fine 0 RANGE: 0 9end(w)
                        triangle = enum_commons.get_subgraph_from_sparse(sparse_matrix,node_in_matrix,v,u,w)
                        #code = _tricode(G, v, u, w)
                        code = tricode(triangle, v, u, w)
                        

                        census[TRICODE_TO_NAME[code]] += 1
                            
                        triad_type = mapping_census_to_baseline[TRICODE_TO_NAME[code]]

                        res,edges_by_time = define_triad_order_new(triangle)

                        triad_abc = res['ordered_triad']
                        pr_bits = triad_code_pagerank(triad_abc,elite_pageranks)
                        outdeg_bits = triad_code_outdegree(triad_abc,elite_outdegrees)
                        #print("Ready for dispatcher with ", triad_type)
                        dispatcher[triad_type](result, pr_bits, outdeg_bits,triangle,triad_abc,edges_by_time)
                        #print("Dispatcher ok for ",triad_type)
                        
                        if STORE_CLOSED and ( triad_type in [6,7,8,9,10,11,12] ):
                            positive_examples.append(enum_commons.store_triad( vals=(v,u,w) ) ) 
                
            processed +=1
            completion = "{:.2f}".format(processed*100/total)
            uptime =  timedelta(seconds=int(time.time() - t_started))
            # print("CHUNK",str(chunk_index),". Done v:", v, "vnbrs", len(vnbrs),";",
            #       str(processed)," out of", total,":",completion,"%","uptime",uptime)
            logging.printlog("CHUNK",str(chunk_index),". Done v:", v, "vnbrs", len(vnbrs),";",
                  str(processed)," out of", total,":",completion,"%","uptime",uptime)


        json.dump(result, zipfile, indent=4, sort_keys=True)
        

    filepath2 = directory_path+"/"+str(chunk_index)+"_COMPLETED.json.gz"
    os.rename(filepath,filepath2)
    
    
    if STORE_CLOSED:
        #print("STORE_CLOSED")
        storage_path = var_dict["storage_path"]
        
        filepath3 = storage_path+"/"+str(chunk_index)+"_TRIADS.csv.gz"
                
        with gzip.open( filepath3, 'wb') as f_out:
            f_out.writelines(positive_examples)
        
        

    # print(datetime.now(),"Fine",chunk_index,"RANGE:",start,stop)
    logging.printlog(datetime.now(),"Fine",chunk_index,"RANGE:",start,stop)
    
    # return {"census":census}
    return result #{"census":census}
    

    
# def get_triad_type(nxG, v,u,w):
#     #code = _tricode(G, v, u, w)
#     code = tricode(nxG, v, u, w)

#     #census[TRICODE_TO_NAME[code]] += 1

#     triad_type = mapping_census_to_baseline[TRICODE_TO_NAME[code]]
    
#     return triad_type

# def get_tricode_name(nxG, v,u,w):
#     code = tricode(nxG, v, u, w)
#     return TRICODE_TO_NAME[code]


def get_triad_type(nxG):
    v,u,w =  list(nxG.nodes())
    code = tricode(nxG, v, u, w)
    triad_type = mapping_census_to_baseline[TRICODE_TO_NAME[code]]
    
    return triad_type

def get_tricode_name(nxG):
    v,u,w =  list(nxG.nodes())
    code = tricode(nxG, v, u, w)
    return TRICODE_TO_NAME[code]