import os

def directed_triadic_census_elite(G, timestamp_key = "t", save_name= "karatetestgraph", folder="triadic_data", store_closed=True, processes=2, max_subprocesses=None):
    
    from . import preprocessLib as preprocessLib
    from . import enum_commons as enum_commons
    from . import directed_census_utils as directed_census_utils

    
    params = {
        "DATA_FOLDER": folder, # folder where the library can store the data, will be created
        "DATASET_NAME": save_name, # subfolder for this dataset, multiple snapshots can be stored here. it will be created
        "TIMESTAMP_KEY": timestamp_key,
        "STORE_CLOSED":store_closed # wheter or not to save data
    }
    
 
    import contextlib
    
    # with open('Log.txt','a+') as f:
    #     with contextlib.redirect_stdout(f):

    preprocessLib.new_generate_snapshot(G, params, current_T_limit=None)
    print()
    #print(params)
    print("Starting parallel census function")
    #print()
    
    if max_subprocesses is None:
        max_subprocesses = processes * 4
        
    aggregated_results = enum_commons.parallelize_task(params, 
                                                       init_worker=directed_census_utils.init_worker,
                                                       runnable_function= directed_census_utils.enumerate_triadic_census, 
                                                       processes=processes, 
                                                       max_subprocesses=max_subprocesses)

    
    
    return {"aggregated_results":aggregated_results, "result_path":params["result_path"] }


def graph_shuffle_edge_attribute(G, attribute ="timestamp", seed = 42, in_place = False):
    
    import copy
    import networkx as nx
    import random
    random.seed(seed)
    
    if in_place:
        G_shuffled = G 
    else:
        G_shuffled = copy.deepcopy(G)    
        
#     print([ e for e in G_shuffled.edges(data=True)][:5])
#     print(all([ attribute in e[2]  for e in G_shuffled.edges(data=True)]) )

#     print(( [ e for e in G_shuffled.edges(data=True) if attribute not in e[2]]) )

    attrs = [ e[2][attribute] for e in G_shuffled.edges(data=True) ]
    # shuffle the edges values list and use nx.set_attribute to override the selected attribute

    random.shuffle( attrs )

    # Note we may avoid generating attrs list, just shuffling the edges indices
    # However, I'm not sure how efficient the access to G[u][v]

    new_attribute_map = { (e[0],e[1]): t for e,t in zip(G_shuffled.edges(), attrs) }

    nx.set_edge_attributes(G_shuffled, new_attribute_map, name= attribute)

    return G_shuffled


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

def directed_triadic_census_elite_significativity(G, timestamp_key = "t", save_name= "karatetestgraph",
                                                  folder="triadic_data", store_closed=True, processes=2, 
                                                  max_subprocesses=None, N=10, graph_shuffle_func = None):

    import os
    import gzip
    import json
    import numpy as np

    from datetime import datetime
    
    if graph_shuffle_func == None:
        #graph_shuffle_func = graph_shuffle_edge_attribute
        print("No shuffling function specified!!!")
    
    now_time = str(datetime.now()).split(".")[0].replace(":","-")
    print(now_time)

    folder_exp =  os.path.join(os.getcwd(),folder, save_name + "_" +now_time)  
    
    make_folder(folder_exp)

    run_name = save_name + "_main"

    print(folder_exp,"e poi", run_name)

    results = {"folder_exp":folder_exp, "main":None, "shuffled_runs":[]}

    # Main run
    result = directed_triadic_census_elite(G, timestamp_key = timestamp_key, 
                                                                    save_name= run_name, 
                                                                    folder= folder_exp, 
                                                                    processes=processes,
                                                                    store_closed = True)

    results["main"] = result
    
    
    print(N)
    print()
    # Shuffled and run N times
    
    for i in range(N):
        run_name = save_name + "_" + str(i)
        print(run_name)

        ### shuffle G

        G_shuffled = graph_shuffle_func(G, attribute= timestamp_key, seed = i) # different shuffle every run but same N to reproduce

        r = {}
        r = directed_triadic_census_elite(G_shuffled, timestamp_key= timestamp_key, 
                                                                save_name= run_name, 
                                                                folder= folder_exp, 
                                                                processes= processes,
                                                                store_closed = False)

        results["shuffled_runs"].append(r)


    # Let's compute the averages right away

    results["averages"] = {}

    for selected in results["main"]["aggregated_results"].keys():
        # print(selected)
        # Assumo che ce ne sia almeno 1 random, lo prendo per le chiavi e preparo liste.
        selected_agg_dict = { k:[] for k in results["shuffled_runs"][0]["aggregated_results"][selected].keys()}

        for r in results["shuffled_runs"]:
            for k,v in r["aggregated_results"][selected].items():
                selected_agg_dict[k].append(v)

        selected_avg_std_dict = { k: (np.average(v),np.std(v)) for k,v in selected_agg_dict.items()}

        results["averages"][selected] = selected_avg_std_dict
    
    

    filepath = os.path.join(folder_exp,"results.json.gz")
    print(filepath)

    with gzip.open(filepath, 'wt+', encoding="utf-8") as zipfile:
        json.dump(results, zipfile, indent=4, sort_keys=True)

    results["result_filepath"] = filepath

    return results

def parallel_triadic_census_plain(G,save_name, timestamp_key="min_timestamp", folder="triadic_data", store_closed=True, n_processes=2):

    results = directed_triadic_census_elite_significativity_args(G, 
                                                        timestamp_key= timestamp_key, #known, simple_construction_from_dataframe
                                                        save_name= save_name, 
                                                        folder = folder, 
                                                        processes=n_processes,
                                                        N=0,
                                                        graph_shuffle_func = None,
                                                        graph_shuffle_func_args_dict= None                  
                                                    )
    
    
    return results

def parallel_triadic_census_shuffled(G, save_name, shuffle_type, timestamp_key="min_timestamp", folder="triadic_data", store_closed=True, n_processes=2, N=3, shuffle_nswapfraction=None):

    graph_shuffle_func_args_dict = {}
    graph_shuffle_func = None
    
    if shuffle_type =="time":
        graph_shuffle_func = graph_shuffle_edge_attribute
        #graph_shuffle_func_args_dict = {}
    
    if shuffle_type =="structure":
        graph_shuffle_func = graph_shuffle_structure
        graph_shuffle_func_args_dict = {"nswap_fraction":shuffle_nswapfraction}
    
    if shuffle_type =="time_structure":
        graph_shuffle_func = graph_shuffle_time_structure
        graph_shuffle_func_args_dict = {"nswap_fraction":shuffle_nswapfraction}
        
    print("Shuffle:",shuffle_type, graph_shuffle_func_args_dict)
    
    if graph_shuffle_func is None:
        print(f"No shuffle function behavior for the specified shuffle_type parameter: {shuffle_type}" )
        print("Current valid values: time, structure, time_structure. Check typos!!!!" )
        #sys.exit(f"No shuffle function behavior for the specified shuffle_type parameter: {shuffle_type} . Exiting ...")
    
    
    results = directed_triadic_census_elite_significativity_args(G, 
                                                        timestamp_key= timestamp_key, #known, simple_construction_from_dataframe
                                                        save_name= save_name, 
                                                        folder = folder, 
                                                        processes=n_processes,
                                                        N=N,
                                                        graph_shuffle_func = graph_shuffle_func,
                                                        graph_shuffle_func_args_dict= graph_shuffle_func_args_dict                  
                                                    )
    
    
    return results


# https://networkx.org/documentation/latest/_modules/networkx/algorithms/swap.html#directed_edge_swap

# https://github.com/networkx/networkx/pull/6149

# https://math.stackexchange.com/questions/22272/reaching-all-possible-simple-directed-graphs-with-a-given-degree-sequence-with-2

import networkx as nx
#@py_random_state(3)
@nx.utils.py_random_state(3)
# @nx.utils.not_implemented_for("undirected")
def directed_edge_swap(G, *, nswap=1, max_tries=100, seed=None):
    """Swap three edges in a directed graph while keeping the node degrees fixed.

    A directed edge swap swaps three edges such that a -> b -> c -> d becomes
    a -> c -> b -> d. This pattern of swapping allows all possible states with the
    same in- and out-degree distribution in a directed graph to be reached.

    If the swap would create parallel edges (e.g. if a -> c already existed in the
    previous example), another attempt is made to find a suitable trio of edges.

    Parameters
    ----------
    G : DiGraph
       A directed graph

    nswap : integer (optional, default=1)
       Number of three-edge (directed) swaps to perform

    max_tries : integer (optional, default=100)
       Maximum number of attempts to swap edges

    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    Returns
    -------
    G : DiGraph
       The graph after the edges are swapped.

    Raises
    ------
    NetworkXError
        If `G` is not directed, or
        If nswap > max_tries, or
        If there are fewer than 4 nodes in `G`
    NetworkXAlgorithmError
        If the number of swap attempts exceeds `max_tries` before `nswap` swaps are made

    Notes
    -----
    Does not enforce any connectivity constraints.

    The graph G is modified in place.

    References
    ----------
    .. [1] Erdős, Péter L., et al. “A Simple Havel-Hakimi Type Algorithm to Realize
           Graphical Degree Sequences of Directed Graphs.” ArXiv:0905.4913 [Math],
           Jan. 2010. https://doi.org/10.48550/arXiv.0905.4913.
           Published  2010 in Elec. J. Combinatorics (17(1)). R66.
           http://www.combinatorics.org/Volume_17/PDF/v17i1r66.pdf
    .. [2] “Combinatorics - Reaching All Possible Simple Directed Graphs with a given
           Degree Sequence with 2-Edge Swaps.” Mathematics Stack Exchange,
           https://math.stackexchange.com/questions/22272/. Accessed 30 May 2022.
    """
    if nswap > max_tries:
        raise nx.NetworkXError("Number of swaps > number of tries allowed.")
    if len(G) < 4:
        raise nx.NetworkXError("Graph has less than four nodes.")

    # Instead of choosing uniformly at random from a generated edge list,
    # this algorithm chooses nonuniformly from the set of nodes with
    # probability weighted by degree.
    tries = 0
    swapcount = 0
    keys, degrees = zip(*G.degree())  # keys, degree
    cdf = nx.utils.cumulative_distribution(degrees)  # cdf of degree
    discrete_sequence = nx.utils.discrete_sequence

    attibutes = {}
    
    while swapcount < nswap:
        # choose source node index from discrete distribution
        start_index = discrete_sequence(1, cdistribution=cdf, seed=seed)[0]
        start = keys[start_index]
        tries += 1

        if tries > max_tries:
            msg = f"Maximum number of swap attempts ({tries}) exceeded before desired swaps achieved ({nswap})."
            raise nx.NetworkXAlgorithmError(msg)

        # If the given node doesn't have any out edges, then there isn't anything to swap
        if G.out_degree(start) == 0:
            continue
        second = seed.choice(list(G.succ[start]))
        if start == second:
            continue

        if G.out_degree(second) == 0:
            continue
        third = seed.choice(list(G.succ[second]))
        if second == third:
            continue

        if G.out_degree(third) == 0:
            continue
        fourth = seed.choice(list(G.succ[third]))
        if third == fourth:
            continue

        if (
            third not in G.succ[start]
            and fourth not in G.succ[second]
            and second not in G.succ[third]
        ):
            # Swap nodes
            
            # G.add_edge(start, third)
            # G.add_edge(third, second)
            # G.add_edge(second, fourth)
            # # 09/11 we need to preserve the attributes before deletion
            # # You can't assign to the view
            # G[start][third]=G[start][second]
            # G[third][second]=G[third][fourth]
            # G[second][fourth]= G[second][third]
            
            G.add_edges_from( [ (start, third, G[start][second]), 
                               (third, second, G[third][fourth]), 
                               (second, fourth, G[second][third])
                              ])
            

            G.remove_edge(start, second)
            G.remove_edge(second, third)
            G.remove_edge(third, fourth)
            swapcount += 1

    print("swapcount", swapcount)
    return G

def graph_shuffle_structure(G, attribute ="timestamp", nswap_fraction = 0.2, seed = 42, in_place = False):
    
    import copy
    import networkx as nx
    import random

    if in_place:
        G_shuffled = G 
    else:
        G_shuffled = copy.deepcopy(G)

    nswap = int(G_shuffled.number_of_edges() * nswap_fraction)
    max_tries = G_shuffled.number_of_edges() * 50
    
    print("we want ", nswap,"swaps, we can try maximum",  max_tries, "times")
    
    G_shuffled = directed_edge_swap(G_shuffled, nswap=nswap, max_tries=max_tries, seed=seed) 
    # Swap edges in the graph while keeping the node degrees fixed. 
    # Does not enforce any connectivity constraints. 
    # The graph G is modified in place.

    return G_shuffled


def graph_shuffle_time_structure(G, attribute ="timestamp", nswap_fraction= 0.2 , seed = 42):

    import copy
    import networkx as nx
    import random
    
    print("Graph:", G, )
    
    G_shuffled = graph_shuffle_structure(G, nswap_fraction=nswap_fraction, seed = seed, in_place = False) # Make a copy and shuffle
    
    # print([ e for e in G_shuffled.edges(data=True)][:5])
    # print(attribute)
    
    G_shuffled = graph_shuffle_edge_attribute(G_shuffled, attribute = attribute, seed = seed, in_place = True) #Shuffle the copy

    # print([ e for e in G_shuffled.edges(data=True)][:5])
    
    return G_shuffled # it's a copy with both shuffling


def directed_triadic_census_elite_significativity_args(G, timestamp_key = "t", save_name= "karatetestgraph",
                                                  folder="triadic_data", store_closed=True, processes=2, 
                                                  max_subprocesses=None, N=10, graph_shuffle_func = None, 
                                                  graph_shuffle_func_args_dict={}):

    import os
    import gzip
    import json
    import numpy as np

    from datetime import datetime
    
    if graph_shuffle_func == None:
        #graph_shuffle_func = graph_shuffle_edge_attribute
        print("No shuffling function specified!!!")
    
    now_time = str(datetime.now()).split(".")[0].replace(":","-")
    print(now_time)

    folder_exp =  os.path.join(os.getcwd(),folder, save_name + "_" +now_time)  
    
    make_folder(folder_exp)

    run_name = save_name + "_main"

    print(folder_exp,"e poi", run_name)

    results = {"folder_exp":folder_exp, "main":None, "shuffled_runs":[]}

    # Main run
    result = directed_triadic_census_elite(G, timestamp_key = timestamp_key, 
                                                                    save_name= run_name, 
                                                                    folder= folder_exp, 
                                                                    processes=processes,
                                                                    store_closed = True)

    results["main"] = result
    
    if N > 0: 
        print(N)
        print()
        # Shuffled and run N times

        for i in range(N):
            run_name = save_name + "_" + str(i)
            print(run_name)

            ### shuffle G

            graph_shuffle_func_args_dict["G"] = G
            graph_shuffle_func_args_dict["seed"] = i
            graph_shuffle_func_args_dict["attribute"] = timestamp_key # Note: structure shuffle should not need attribute param


            G_shuffled = graph_shuffle_func(**graph_shuffle_func_args_dict) # different shuffle every run but same N to reproduce

            r = {}
            r = directed_triadic_census_elite(G_shuffled, timestamp_key= timestamp_key, 
                                                                    save_name= run_name, 
                                                                    folder= folder_exp, 
                                                                    processes= processes,
                                                                    store_closed = False)

            results["shuffled_runs"].append(r)


        # Let's compute the averages right away

        results["averages"] = {}

        for selected in results["main"]["aggregated_results"].keys():
            # print(selected)
            # Assumo che ce ne sia almeno 1 random, lo prendo per le chiavi e preparo liste.
            selected_agg_dict = { k:[] for k in results["shuffled_runs"][0]["aggregated_results"][selected].keys()}

            for r in results["shuffled_runs"]:
                for k,v in r["aggregated_results"][selected].items():
                    selected_agg_dict[k].append(v)

            selected_avg_std_dict = { k: (np.average(v),np.std(v)) for k,v in selected_agg_dict.items()}

            results["averages"][selected] = selected_avg_std_dict
    else:
        print("N is :", N)

    filepath = os.path.join(folder_exp,"results.json.gz")
    print(filepath)

    with gzip.open(filepath, 'wt+', encoding="utf-8") as zipfile:
        json.dump(results, zipfile, indent=4, sort_keys=True)

    results["result_filepath"] = filepath

    return results