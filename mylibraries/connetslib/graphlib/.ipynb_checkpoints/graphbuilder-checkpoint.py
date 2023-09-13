#### graph construction utilities
# restituisce uno alla volta i grafi. Sta a te decidere cosa fare con ognuno di questi nel loop - se vuoi aggiungere info a calcolare le stats con le funzioni fornite in libreria, quali salvare
import networkx as nx

GRAPH_TYPE_OPTIONS = {"digraph": nx.DiGraph,
                      "multidigraph": nx.MultiDiGraph}  # , "multigraph": nx.MultiGraph ,"graph": nx.Graph }


# TODO: gestire i casi undirected.
# Per graph vuoi che vengano uniti es: quando arrivano src, target ordini in modo che vengano inseriti sempre su stesso link
# Per multigraph bisogna riflettere se basta lo stesso trucchetto


# transactions = glib_storage.load_transactions_dataframe(PROCESSED_DATA_FILEPATH)

# naive_construction_from_dataframe(transactions,
#                                   start_string = "2017-11-23 00:00:00",
#                                   end_string = "2018-02-21 00:00:00",
#                                   GRAPH_TYPE = "digraph",
#                                   add_edge_func = add_edge_function_weight_amount
#                                  )


def simple_construction_from_dataframe(transactions,
                                      start_string="2017-11-23 00:00:00",
                                      end_string="2018-02-21 00:00:00",
                                      timestamp_key="timestamp",
                                      GRAPH_TYPE="digraph",
                                      add_edge_func=None):
    import networkx as nx
    import pandas as pd

    # GRAPH_TYPE_OPTIONS = {"multigraph": nx.MultiGraph ,"digraph": nx.DiGraph, "multidigraph": nx.MultiDiGraph, "graph": nx.Graph }

    if GRAPH_TYPE in GRAPH_TYPE_OPTIONS:
        G = GRAPH_TYPE_OPTIONS[GRAPH_TYPE]()
        print(GRAPH_TYPE, G.__class__)
    else:
        raise Exception("Graph type not valid, here the valid options", GRAPH_TYPE_OPTIONS)

    start = pd.Timestamp(start_string)
    end = pd.Timestamp(end_string)

    print(end)

    try:
        # from tqdm import tqdm
        # from tqdm.notebook import tqdm as tqdm
        from tqdm.auto import tqdm

    except ImportError:
        def tqdm(iterator, *args, **kwargs):
            return iterator

    data = transactions[(transactions[timestamp_key] >= start) & (transactions[timestamp_key] < end)]

    for i, t in tqdm(data.iterrows(), total=data.shape[0]):  # , leave=False):
        # if t["timestamp"] < end:
        add_edge_func(G, t)

    print(G)
    return G


# Create intervals

def get_start_time(transactions, timestamp_key="timestamp"):
    # Start time derived from available dataset
    return min(transactions[timestamp_key]).strftime("%Y-%m-%d %H:%M:%S")  # .to_pydatetime()


def get_end_time(transactions, timestamp_key="timestamp"):
    return max(transactions[timestamp_key]).strftime("%Y-%m-%d %H:%M:%S")  # .to_pydatetime()


def get_intervals(START_TIME, END_TIME,
                  days=1, hours=0, minutes=0, seconds=0,
                  ALWAYS_END_AT_MIDNIGHT=False):
    import datetime
    import pandas as pd

    START_TIME = datetime.datetime.strptime(START_TIME, "%Y-%m-%d %H:%M:%S")
    END_TIME = datetime.datetime.strptime(END_TIME, "%Y-%m-%d %H:%M:%S")

    WINDOW_SIZE = datetime.timedelta(days=days, seconds=seconds,
                                     microseconds=0, milliseconds=0,
                                     minutes=minutes, hours=hours, weeks=0)
    WINDOW_SIZE_STRING = f"{days}d_{hours}h_{minutes}m_{seconds}s"

    print(WINDOW_SIZE, WINDOW_SIZE_STRING)

    print(START_TIME, "-", END_TIME)

    if ALWAYS_END_AT_MIDNIGHT:
        print("ALWAYS_END_AT_MIDNIGHT is on, last timestamp always 23:59")

    SNAPSHOTS = {}

    START_SNAP = START_TIME
    END_SNAP = START_TIME

    # First snapshot
    SNAP_INDEX = 0

    END_SNAP = END_SNAP + WINDOW_SIZE

    if ALWAYS_END_AT_MIDNIGHT:
        END_SNAP = pd.Timestamp(END_SNAP.date()).to_pydatetime()

    SNAPSHOTS[SNAP_INDEX] = {"start": START_SNAP.strftime('%Y-%m-%d %H:%M:%S'),
                             "end": END_SNAP.strftime('%Y-%m-%d %H:%M:%S')}

    # Next ones

    while END_SNAP < END_TIME:

        SNAP_INDEX += 1

        START_SNAP = END_SNAP

        END_SNAP = END_SNAP + WINDOW_SIZE

        # Check: i can't end the new snap after END_TIME
        if END_SNAP > END_TIME:
            print("Notice that END_SNAP > END_TIME, last snapshot will cover less time")
            print("END_SNAP will be adjusted to END_TIME")
            END_SNAP = END_TIME

        SNAPSHOTS[SNAP_INDEX] = {"start": START_SNAP.strftime('%Y-%m-%d %H:%M:%S'),
                                 "end": END_SNAP.strftime('%Y-%m-%d %H:%M:%S')}

    params = {
        # "WINDOW_SIZE":WINDOW_SIZE,
        "WINDOW_SIZE_STRING": WINDOW_SIZE_STRING,
        "ALWAYS_END_AT_MIDNIGHT": ALWAYS_END_AT_MIDNIGHT,
        "START_TIME": START_TIME.strftime('%Y-%m-%d %H:%M:%S'),
        "END_TIME": END_TIME.strftime('%Y-%m-%d %H:%M:%S')
    }

    return {"params": params, "intervals": SNAPSHOTS}


def get_regular_one_month_intervals(START_TIME, END_TIME, ALWAYS_END_AT_MIDNIGHT=False):
    import copy

    def get_next_month(mydate):

        """
            # d = datetime.datetime.strptime( "2017-12-15 00:00:00","%Y-%m-%d %H:%M:%S")
            # get_next_month(mydate = d)
        """

        # check 1: day in range 1,28, otherwise no can do!

        if mydate.day > 28:
            raise ValueError("For regular snapshots, day can only be in range 1-28")

        next_month_mydate = copy.deepcopy(mydate)

        # check 2: if we are in december, we also have to increase year
        if mydate.month == 12:
            next_month_mydate = next_month_mydate.replace(year=mydate.year + 1)
            next_month_mydate = next_month_mydate.replace(month=1)
        else:
            next_month_mydate = next_month_mydate.replace(month=(mydate.month + 1))

        return next_month_mydate

    # per aver snapshot sempre al primo del mese oppure sempre il 23 del mese al tempo del fork
    # posso usarlo solo con giorni fino al 28 - perchè mesi di lunghezza diversa is a bitch

    import datetime
    import pandas as pd

    START_TIME = datetime.datetime.strptime(START_TIME, "%Y-%m-%d %H:%M:%S")
    END_TIME = datetime.datetime.strptime(END_TIME, "%Y-%m-%d %H:%M:%S")

    WINDOW_SIZE_STRING = "1monthregular"

    if ALWAYS_END_AT_MIDNIGHT:
        print("ALWAYS_END_AT_MIDNIGHT is on, last timestamp always 23:59")

    SNAPSHOTS = {}

    START_SNAP = START_TIME
    END_SNAP = START_TIME

    # First snapshot
    SNAP_INDEX = 0

    END_SNAP = get_next_month(START_SNAP)  # END_SNAP + WINDOW_SIZE

    if ALWAYS_END_AT_MIDNIGHT:
        END_SNAP = pd.Timestamp(END_SNAP.date()).to_pydatetime()

    SNAPSHOTS[SNAP_INDEX] = {"start": START_SNAP.strftime('%Y-%m-%d %H:%M:%S'),
                             "end": END_SNAP.strftime('%Y-%m-%d %H:%M:%S')}

    # Next ones

    while END_SNAP < END_TIME:

        SNAP_INDEX += 1

        START_SNAP = END_SNAP

        END_SNAP = get_next_month(START_SNAP)  # END_SNAP + WINDOW_SIZE

        # Check: i can't end the new snap after END_TIME
        if END_SNAP > END_TIME:
            print("Notice that END_SNAP > END_TIME, last snapshot will cover less time")
            print("END_SNAP will be adjusted to END_TIME")
            END_SNAP = END_TIME

        SNAPSHOTS[SNAP_INDEX] = {"start": START_SNAP.strftime('%Y-%m-%d %H:%M:%S'),
                                 "end": END_SNAP.strftime('%Y-%m-%d %H:%M:%S')}

    params = {
        "WINDOW_SIZE_STRING": WINDOW_SIZE_STRING,
        "ALWAYS_END_AT_MIDNIGHT": ALWAYS_END_AT_MIDNIGHT,
        "START_TIME": START_TIME.strftime('%Y-%m-%d %H:%M:%S'),
        "END_TIME": END_TIME.strftime('%Y-%m-%d %H:%M:%S')
    }

    return {"params": params, "intervals": SNAPSHOTS}


def get_monthly_intervals_from_dataframe(dataframe_or_path, SNAPSHOTS_DATA_PATH, DATASET_NAME, START_TIME=None,
                                         END_TIME=None):
    import pandas as pd

    if isinstance(dataframe_or_path, pd.DataFrame):
        transactions = dataframe_or_path
    elif isinstance(dataframe_or_path, str):
        print("Loading dataframe")
        transactions = glib_storage.load_transactions_dataframe(dataframe_or_path)
    else:
        raise Exception("path to pandas dataframe or pandas dataframe accepted")

    print("Computing snaps")
    if START_TIME is None:
        START_TIME = get_start_time(transactions)

    if END_TIME is None:
        END_TIME = get_end_time(transactions)

    INTERVALS = get_regular_one_month_intervals(START_TIME=START_TIME,
                                            END_TIME=END_TIME,
                                            ALWAYS_END_AT_MIDNIGHT=True)

    return INTERVALS


# Where to store
from . import storage as glib_storage

import os


def writer_prep_folders(SNAPSHOTS_DATA_PATH, DATASET_NAME, SNAPSHOT_TYPE, GRAPH_TYPE, params):
    p = params["params"]

    dirname = os.path.join(SNAPSHOTS_DATA_PATH, DATASET_NAME)

    glib_storage.make_folder(dirname)

    subdirname = os.path.join(dirname,
                              f"{SNAPSHOT_TYPE}_{GRAPH_TYPE}_{p['WINDOW_SIZE_STRING']}_{p['START_TIME']}_{p['END_TIME']}/")

    glib_storage.make_folder(subdirname)

    #     params["SNAP_FOLDER"] = subdirname

    #     expdirname = os.path.join(subdirname,"experiments")

    #     params["EXPERIMENTS_FOLDER"] = expdirname

    #     make_folder(expdirname)

    test = {"INTERVALS_params": p, "SNAPSHOT_TYPE": SNAPSHOT_TYPE, "GRAPH_TYPE": GRAPH_TYPE,
            "functions": params["functions"]}
    test_path = os.path.join(subdirname, "SETTING")
    glib_storage.save_dict_as_compressed_json(test, fname=test_path)

    return {"subdirname": subdirname}


# Some simple building functions -- users can define their own accordingly

try:
    # from tqdm import tqdm
    # from tqdm.notebook import tqdm as tqdm
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterator, *args, **kwargs):
        return iterator


## Chunk getters
def get_chunk_from_dataframe(transactions, START_SNAP, END_SNAP, timestamp_key="timestamp"):
    import datetime
    import pandas as pd

    start = datetime.datetime.strptime(START_SNAP, "%Y-%m-%d %H:%M:%S")
    end = datetime.datetime.strptime(END_SNAP, "%Y-%m-%d %H:%M:%S")
    # idea è che possa essere sostituito con get_chunk da file diversi per esempio
    chunk = transactions[(transactions[timestamp_key] >= start) & (transactions[timestamp_key] < end)]

    return chunk


# NOTA: tutte le add_edge assumono che ci sia dictionary - magari facciamo funzione che permette di indicare gli indici giusti in una tupla
# Add edge
def add_edge_function(G, t):
    G.add_edge(t['from'], t['to'])


# check if has_edge, if it does, increase weight else new edge with weight 1

def add_edge_function_weight(G, t):
    src, dest = t['from'], t['to']

    if G.has_edge(src, dest):
        G[src][dest]["weight"] += 1
    else:
        G.add_edge(src, dest, weight=1)


def add_edge_function_weight_amount(G, t):
    src, dest, amount = t['from'], t['to'], t["amount"]

    if G.has_edge(src, dest):
        G[src][dest]["weight"] += 1
        G[src][dest]["amount"] += t["amount"]
    else:
        G.add_edge(src, dest, weight=1, amount=t["amount"])

# check if has_edge, if it does, increase weight else new edge with weight 1
def add_edge_function_weight_temporal(G, t):
    src, dest, timestamp = t['from'], t['to'], t["timestamp"].strftime("%Y-%m-%d %H:%M:%S")

    if G.has_edge(src, dest):
        G[src][dest]["weight"] += 1
        
        if timestamp < G[src][dest]["min_timestamp"]:
             G[src][dest]["min_timestamp"] = timestamp
                
        if timestamp > G[src][dest]["max_timestamp"]:
             G[src][dest]["max_timestamp"] = timestamp
        
    else:
        G.add_edge(src, dest, weight=1, min_timestamp = timestamp, max_timestamp = timestamp  )
        

def add_edge_function_weight_amount_temporal(G, t):
    src, dest, timestamp, amount = t['from'], t['to'], t["timestamp"].strftime("%Y-%m-%d %H:%M:%S"), t["amount"]

    if G.has_edge(src, dest):
        G[src][dest]["weight"] += 1
        G[src][dest]["amount"] += t["amount"]
        
        if timestamp < G[src][dest]["min_timestamp"]:
             G[src][dest]["min_timestamp"] = timestamp
                
        if timestamp > G[src][dest]["max_timestamp"]:
             G[src][dest]["max_timestamp"] = timestamp
        
        
    else:
        G.add_edge(src, dest, weight=1, amount=t["amount"], min_timestamp = timestamp, max_timestamp = timestamp  )
        
        
# basic for multidigraph, just add u,v, timestamp.
def add_edge_function_multidigraph(G, t):
    G.add_edge(t['from'], t['to'], timestamp= t["timestamp"].strftime("%Y-%m-%d %H:%M:%S") )


# basic for multidigraph, just add u,v, timestamp.
def add_edge_function_multidigraph_amount(G, t):
    G.add_edge(t['from'], t['to'], timestamp= t["timestamp"].strftime("%Y-%m-%d %H:%M:%S"), amount=t["amount"])


# Build from chunk

def build_from_chunk_dataframe(G, add_edge_function, chunk):
    for i, t in tqdm(chunk.iterrows(), total=chunk.shape[0], leave=False):
        # For each edge
        add_edge_function(G, t)


def at_snapshot_end(G, SNAP_INDEX, START_SNAP, END_SNAP, snapshot_folder_path):
    import os

    stats = {}

    stats = get_snapshot_metrics(G, START_SNAP, END_SNAP)

    stats["index"] = SNAP_INDEX

    stats_path = os.path.join(snapshot_folder_path, f"snap_{SNAP_INDEX}_stats")
    glib_storage.save_dict_as_compressed_json(stats, fname=stats_path)


def store_snapshot(G, SNAP_INDEX, snapshot_folder_path):
    import os
    snapshot_path = os.path.join(snapshot_folder_path, f"snap_{SNAP_INDEX}.pkl.xz")
    print(snapshot_path)

    #     # snapshot_params["SNAP_INDEX"] = SNAP_INDEX
    #     # snapshot_params["START_SNAP"] = START_SNAP
    #     # snapshot_params["END_SNAP"] = END_SNAP
    #     # snapshot_params["snapshot_path"] = snapshot_path
    #     # G.graph["name"] = f"snap_{snapshot_params['SNAP_INDEX']}"
    #     # G.remove_edges_from(list(nx.selfloop_edges(G)))
    #     # G.remove_nodes_from(list(nx.isolates(G)))
    #     # G.graph["params"] = snapshot_params
    #     # print(snapshot_params['SNAP_INDEX'], G)

    glib_storage.save_snapshot_lzma(G, snapshot_path)


def load_and_create_snapshots_from_transactions_dataframe(
        path_to_dataframe=None,
        INTERVALS=None,
        SNAPS_TO_SAVE=[],
        SNAPSHOTS_DATA_PATH=None,  # "/home/jupyter-cheick_ba/nas/data_on_nas/datasets/snapshots/"
        DATASET_NAME=None,  # "OSF" #"eth_ntfs_moonstream" #"OSF" #"steemit"
        SNAPSHOT_TYPE="EVOLVING",  # "INTERVAL" # Puoi mettere if nel loop e resettare G ad ogni snap
        GRAPH_TYPE="digraph",  # multidigraph or multigraph (undirected) or graph (undirected)):
        get_chunk_func=get_chunk_from_dataframe,
        add_edge_func=add_edge_function_weight,
        build_snapshot_func=build_from_chunk_dataframe,
        at_snapshot_end_func=at_snapshot_end,
        store_snapshot_func=store_snapshot):
    
    if path_to_dataframe is None:
        raise Exception("path_to_dataframe not defined")

    print("Loading transactions dataframe")
    transactions = glib_storage.load_transactions_dataframe(path_to_dataframe)

    print("Now: create snapshots")
    storing_folders = create_snapshots_from_transactions_dataframe(
        INTERVALS=INTERVALS,
        INTERVALS_TO_SAVE=SNAPS_TO_SAVE,
        SNAPSHOTS_DATA_PATH=SNAPSHOTS_DATA_PATH,
        DATASET_NAME=DATASET_NAME,  # "OSF" #"eth_ntfs_moonstream" #"OSF" #"steemit"
        SNAPSHOT_TYPE=SNAPSHOT_TYPE,  # "INTERVAL" # Puoi mettere if nel loop e resettare G ad ogni snap
        GRAPH_TYPE=GRAPH_TYPE,  # multidigraph or multigraph (undirected) or graph (undirected)):
        DATASOURCE=transactions,  # o altrimenti sarà chainreader/lista di file
        get_chunk_func=get_chunk_func,
        add_edge_func=add_edge_func,
        build_snapshot_func=build_snapshot_func,
        at_snapshot_end_func=at_snapshot_end_func,
        store_snapshot_func=store_snapshot_func
    )

    return storing_folders


def create_snapshots_from_transactions_dataframe(
        INTERVALS=None,
        INTERVALS_TO_SAVE=[],
        SNAPSHOTS_DATA_PATH=None,  # "/home/jupyter-cheick_ba/nas/data_on_nas/datasets/snapshots/"
        DATASET_NAME=None,  # "OSF" #"eth_ntfs_moonstream" #"OSF" #"steemit"
        SNAPSHOT_TYPE="EVOLVING",  # "INTERVAL" # Puoi mettere if nel loop e resettare G ad ogni snap
        GRAPH_TYPE="digraph",  # multidigraph or multigraph (undirected) or graph (undirected)):
        DATASOURCE=None,  # o altrimenti sarà chainreader/lista di file
        get_chunk_func=get_chunk_from_dataframe,
        add_edge_func=add_edge_function_weight,
        build_snapshot_func=build_from_chunk_dataframe,
        at_snapshot_end_func=at_snapshot_end,
        store_snapshot_func=store_snapshot):
    import networkx as nx
    import pandas as pd
    import datetime

    # try:
    #     from tqdm.notebook import tqdm as tqdm
    # except ImportError:
    #     def tqdm(iterator, *args, **kwargs):
    #         return iterator

    if INTERVALS is None:
        raise Exception("INTERVALS not defined")

    if SNAPSHOTS_DATA_PATH is None:
        raise Exception("SNAPSHOTS_DATA_PATH not defined")

    if DATASET_NAME is None:
        raise Exception("DATASET_NAME not defined")

    if DATASOURCE is None:
        raise Exception("DATASOURCE not defined")

    if isinstance(DATASOURCE, pd.DataFrame):
        DATASOURCE = DATASOURCE
    elif isinstance(DATASOURCE, str):
        print("Loading dataframe")
        DATASOURCE = glib_storage.load_transactions_dataframe(DATASOURCE)
    else:
        raise Exception("path to pandas dataframe or pandas dataframe accepted")

    # GRAPH_TYPE_OPTIONS = {"multigraph": nx.MultiGraph ,"digraph": nx.DiGraph, "multidigraph": nx.MultiDiGraph, "graph": nx.Graph }

    if GRAPH_TYPE in GRAPH_TYPE_OPTIONS:
        G = GRAPH_TYPE_OPTIONS[GRAPH_TYPE]()
        print(GRAPH_TYPE, G.__class__)
    else:
        raise Exception("Graph type not valid, here the valid options", GRAPH_TYPE_OPTIONS)

    # Prep folders and data structures

    building_params = {}
    building_params["GRAPH_CLASS"] = str(G.__class__)
    building_params["get_chunk_func"] = str(get_chunk_func.__name__)
    building_params["add_edge_func"] = str(add_edge_func.__name__)
    building_params["build_snapshot_func"] = str(build_snapshot_func.__name__)
    building_params["at_snapshot_end_func"] = str(at_snapshot_end_func.__name__)
    building_params["store_snapshot_func"] = str(store_snapshot_func.__name__)
    # DATASOURCE = transactions # o altrimenti sarà chainreader/lista di file

    INTERVALS["functions"] = building_params

    storing_folders = writer_prep_folders(SNAPSHOTS_DATA_PATH, DATASET_NAME,
                                          SNAPSHOT_TYPE, GRAPH_TYPE=GRAPH_TYPE, params=INTERVALS)

    print(building_params)

    ### Loop through intervals

    for SNAP_INDEX in tqdm(INTERVALS["intervals"].keys(), total=len(INTERVALS["intervals"])):  # one graph at the time

        # select a subset of the edges
        START_SNAP = INTERVALS["intervals"][SNAP_INDEX]["start"]
        END_SNAP = INTERVALS["intervals"][SNAP_INDEX]["end"]

        # build
        chunk = get_chunk_func(DATASOURCE, START_SNAP, END_SNAP)

        # glib_builder.build_from_chunk_dataframe(G = G, add_edge_function = glib_builder.add_edge_function, chunk = chunk)
        build_snapshot_func(G=G, add_edge_function=add_edge_func, chunk=chunk)

        # what to do with every snapshot - for storage see next part of loop
        at_snapshot_end_func(G, SNAP_INDEX, START_SNAP, END_SNAP, snapshot_folder_path=storing_folders["subdirname"])

        # store snapshot
        if END_SNAP in INTERVALS_TO_SAVE:  # # end window, decide if we need to write this snapshot

            # printlog("Reached", t["timestamp"],">",END_SNAP,"so we write snap", SNAP_INDEX)
            print("To save:", END_SNAP)
            store_snapshot_func(G, SNAP_INDEX, snapshot_folder_path=storing_folders["subdirname"])
            # qua si potrebbe salvare lista grafi - ma richiede attenzione - non importante ora

        # per grafi interval, qua basterebbe creare un nuovo G
        if SNAPSHOT_TYPE == "INTERVAL":
            G = GRAPH_TYPE_OPTIONS[GRAPH_TYPE]()

    # se grafo evolving può aver senso salvare il grafo finale - magari meglio se mettiamo come parametro poi
    if (END_SNAP not in INTERVALS_TO_SAVE) and (SNAPSHOT_TYPE == "EVOLVING"):
        print("Saving last snapshot, since it is EVOLVING and END SNAP was not in SNAPS_TO_SAVE")
        at_snapshot_end_func(G, SNAP_INDEX, START_SNAP, END_SNAP, snapshot_folder_path=storing_folders["subdirname"])
        store_snapshot_func(G, SNAP_INDEX, snapshot_folder_path=storing_folders["subdirname"])

    return storing_folders


### SNAP_END_CALLBACK: what to do on the snapshot
# ! pip install networkit

def nk_reciprocity(G):
    n_all_edge = G.numberOfEdges()
    # n_overlap_edge=sum([1 for u, v in tqdm(G.iterEdges()) if G.hasEdge(v,u) == True])
    n_overlap_edge = sum([1 for u, v in G.iterEdges() if G.hasEdge(v, u) == True])

    if n_all_edge == 0:
        raise NetworkXError("Not defined for empty graphs")

    return float(n_overlap_edge) / float(n_all_edge)


def snap_end_stats_callback(G):  # params):

    import networkit as nk
    import networkx as nx

    # Help from networkit for large scale computation of metrics
    nkG = nk.nxadapter.nx2nk(G)
    #   nk.overview(nkG)

    # required for networkit diameter algorithms
    Gundirected = nk.graphtools.toUndirected(nkG)
    Gunweighted = nk.graphtools.toUnweighted(Gundirected)

    degrees = nk.centrality.DegreeCentrality(nkG, ignoreSelfLoops=nkG.numberOfSelfLoops() == 0).run().scores()

    # Connected components
    cc = nk.components.ConnectedComponents(Gundirected)
    cc.run()

    scc = nk.components.StronglyConnectedComponents(nkG)
    scc.run()

    wcc = nk.components.WeaklyConnectedComponents(nkG)
    wcc.run()

    d = {
        # "Index": G.graph["name"],
        # "Start": params["START_SNAP"].strftime('%Y/%m/%d %H:%M:%S'),
        # "End": params["END_SNAP"].strftime('%Y/%m/%d %H:%M:%S'),
        "Nodes": G.number_of_nodes(),
        "Edges": G.number_of_edges(),
        "Density": nx.density(G),
        #          "n": nkG.numberOfNodes(),
        #          "e": nkG.numberOfEdges(),
        "Diameter": nk.distance.Diameter(Gunweighted, algo=1).run().getDiameter()[0],
        "Average Local Clustering": nk.globals.clustering(nkG),
        "Reciprocity": nk_reciprocity(nkG),
        "Degree assortatvity": nk.correlation.Assortativity(nkG, degrees).run().getCoefficient(),
        #"N_cc": cc.numberOfComponents(),
        #          "Largest_cc_%":  max(cc.getComponentSizes().values()) / Gundirected.numberOfNodes(),
        "N_cc_weak": wcc.numberOfComponents(),
        "Largest_cc_%_weak": max(wcc.getComponentSizes().values()) / Gundirected.numberOfNodes(),
        "N_cc_strong": scc.numberOfComponents(),
        "Largest_cc_%_strong": max(scc.getComponentSizes().values()) / Gundirected.numberOfNodes(),
        "transitivity": nx.transitivity(G),
        #"avg_shortest_path":nx.average_shortest_path_length(G)
        
    }  # "diameter": nx.diameter(G)}

    return d


def get_snapshot_metrics(G, START_SNAP, END_SNAP):
    import networkx as nx

    # stats[SNAP_INDEX] = get_snapshot_metrics(G, START_SNAP, END_SNAP)
    if isinstance(G, nx.MultiDiGraph) or isinstance(G, nx.MultiGraph):
        res = snap_end_stats_callback_nx(G)
    else:
        res = snap_end_stats_callback(G)

    # utility function that computes some stats, using both networkx and networkit
    # we can compute and add more stats/info in here if needed

    # res["Index"] = G.graph["name"],
    res["Start"] = START_SNAP  # .strftime('%Y/%m/%d %H:%M:%S')
    res["End"] = END_SNAP  # .strftime('%Y/%m/%d %H:%M:%S')

    return res


def snap_end_stats_callback_nx(G):  # params):

    import networkx as nx

    # Help from networkit for large scale computation of metrics

    d = {
        # "Index": G.graph["name"],
        # "Start": params["START_SNAP"].strftime('%Y/%m/%d %H:%M:%S'),
        # "End": params["END_SNAP"].strftime('%Y/%m/%d %H:%M:%S'),
        "Nodes": G.number_of_nodes(),
        "Edges": G.number_of_edges(),
        "Density": nx.density(G),
        # "Diameter": nx.diameter(G)

        #          "Diameter":nk.distance.Diameter(Gunweighted,algo=1).run().getDiameter()[0],
        #          "Average Local Clustering":nk.globals.clustering(nkG),
        #          "Reciprocity":nk_reciprocity(nkG),
        #          "Degree assortatvity":nk.correlation.Assortativity(nkG, degrees).run().getCoefficient(),
        #          "N_cc": cc.numberOfComponents(),
        # #          "Largest_cc_%":  max(cc.getComponentSizes().values()) / Gundirected.numberOfNodes(),
        # #          "N_cc_weak": wcc.numberOfComponents(),
        #          "Largest_cc_%_weak":  max(wcc.getComponentSizes().values()) / Gundirected.numberOfNodes(),
        #          "N_cc_strong":  scc.numberOfComponents(),
        #          "Largest_cc_%_strong": max(scc.getComponentSizes().values()) / Gundirected.numberOfNodes()
    }  # "diameter": nx.diameter(G)}

    return d


###### Funzioni di generazione



def multidigraph_from_dataframe(transactions, SNAPSHOTS_DATA_PATH, DATASET_NAME, INTERVALS, SNAPSHOT_TYPE = "EVOLVING", has_amount = False):
    
    if has_amount:
        add_edge_func = add_edge_function_multidigraph_amount
    else:
        add_edge_func = add_edge_function_multidigraph
    
    stored = create_snapshots_from_transactions_dataframe(
        DATASOURCE=transactions,  # o altrimenti sarà chainreader/lista di file
        INTERVALS=INTERVALS,
        INTERVALS_TO_SAVE=[],
        SNAPSHOTS_DATA_PATH=SNAPSHOTS_DATA_PATH,
        DATASET_NAME=DATASET_NAME,  # "OSF" #"eth_ntfs_moonstream" #"OSF" #"steemit"
        SNAPSHOT_TYPE= SNAPSHOT_TYPE,  # "INTERVAL" # Puoi mettere if nel loop e resettare G ad ogni snap
        GRAPH_TYPE="multidigraph",  # graph, digraph, multidigraph or multigraph:
        get_chunk_func=get_chunk_from_dataframe,
        add_edge_func = add_edge_func,  # glib_builder.add_edge_function_weight,
        build_snapshot_func=build_from_chunk_dataframe,
        at_snapshot_end_func=at_snapshot_end,
        store_snapshot_func=store_snapshot
    )

    folder = stored["subdirname"]
    print(folder)
    glib_analysis.get_recap(folder, return_also_files=False)

    return folder


def digraph_from_dataframe(transactions, SNAPSHOTS_DATA_PATH, DATASET_NAME, INTERVALS, SNAPSHOT_TYPE = "EVOLVING", has_amount=False):
    
    
    if has_amount:
        add_edge_func = add_edge_function_weight_amount_temporal #add_edge_function_weight_amount
    else:
        add_edge_func = add_edge_function_weight_temporal #add_edge_function_weight
        
    stored = create_snapshots_from_transactions_dataframe(
        DATASOURCE=transactions,  # o altrimenti sarà chainreader/lista di file
        INTERVALS=INTERVALS,
        INTERVALS_TO_SAVE=[],
        SNAPSHOTS_DATA_PATH=SNAPSHOTS_DATA_PATH,
        DATASET_NAME=DATASET_NAME,  # "OSF" #"eth_ntfs_moonstream" #"OSF" #"steemit"
        SNAPSHOT_TYPE= SNAPSHOT_TYPE,  # "INTERVAL" # Puoi mettere if nel loop e resettare G ad ogni snap
        GRAPH_TYPE="digraph",  # graph, digraph, multidigraph or multigraph:
        get_chunk_func=get_chunk_from_dataframe,
        add_edge_func=add_edge_func,  # glib_builder.,
        build_snapshot_func= build_from_chunk_dataframe,
        at_snapshot_end_func=at_snapshot_end,
        store_snapshot_func=store_snapshot
    )

    folder = stored["subdirname"]
    print(folder)
    glib_analysis.get_recap(folder, return_also_files=False)

    return folder




# # Esempi with SNAPS
# def get_evolving_graph_from_dataframe_and_intervals(dataframe_or_path, SNAPSHOTS_DATA_PATH, DATASET_NAME, SNAPS = None):

#     import pandas as pd

#     if isinstance(dataframe_or_path, pd.DataFrame):
#         transactions = dataframe_or_path
#     elif isinstance(dataframe_or_path,str):
#         print("Loading dataframe")
#         transactions = glib_storage.load_transactions_dataframe(dataframe_or_path)
#     else:
#         raise Exception("path to pandas dataframe or pandas dataframe accepted")

#     print("Checking snaps")

#     if SNAPS is None:
#         raise Exception("You need to generate a SNAPS object with the intervals. Use get_regular_one_month_intervals() or get_intervals() ")

#     print(f"Proceding with {len(SNAPS['intervals'])} intervals")

#     folder = evolving_graph_from_dataframe(transactions, SNAPSHOTS_DATA_PATH, DATASET_NAME, SNAPS )

#     return folder

# # Automatic monthly snaps
# def get_monthly_evolving_graph_from_dataframe(dataframe_or_path, SNAPSHOTS_DATA_PATH, DATASET_NAME, START_TIME = None, END_TIME = None):

#     import pandas as pd

#     if isinstance(dataframe_or_path, pd.DataFrame):
#         transactions = dataframe_or_path
#     elif isinstance(dataframe_or_path,str):
#         print("Loading dataframe")
#         transactions = glib_storage.load_transactions_dataframe(dataframe_or_path)
#     else:
#         raise Exception("path to pandas dataframe or pandas dataframe accepted")

#     print("Computing snaps")
#     if START_TIME is None:
#         START_TIME = get_start_time(transactions)

#     if END_TIME is None:
#         END_TIME = get_end_time(transactions)

#     SNAPS = get_regular_one_month_intervals(START_TIME = START_TIME,
#                                              END_TIME= END_TIME,
#                                              ALWAYS_END_AT_MIDNIGHT = True )

#     print(f"Proceding with {len(SNAPS['intervals'])} intervals")

#     folder = evolving_graph_from_dataframe(transactions, SNAPSHOTS_DATA_PATH, DATASET_NAME, SNAPS )

#     return folder


