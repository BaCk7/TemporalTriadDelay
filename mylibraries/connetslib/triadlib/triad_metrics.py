import sys
nas_folder = "." 
sys.path.insert(1, f'{nas_folder}/mylibraries')

import connetslib.graphlib.graphbuilder as glib_builder
import connetslib.graphlib.storage as glib_storage
import connetslib.graphlib.analysis as glib_analysis

import connetslib.triadlib.census_functions as census_functions
import connetslib.triadlib.enum_commons as enum_commons
import connetslib.triadlib.plotting_functions as plotting
import connetslib.triadlib.directed_census_utils as directed_census_utils
import connetslib.triadlib.aggregated_functions as agg_funcs

import sys
import os
from collections import OrderedDict, Counter, defaultdict

try:
    # from tqdm import tqdm
    #from tqdm.notebook import tqdm as tqdm
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterator, *args, **kwargs):
        return iterator

import networkx as nx
def get_early_times_for_delay(triangle, ordered_triad, edges_by_time, timestamp_key = "t"):

    a,b,c = ordered_triad
    t_ac = triangle.get_edge_data(a, c, default={timestamp_key:'9999-12-31 23:59:59'})[timestamp_key] # if edge does not exist, return a "infinite date"
    t_ca = triangle.get_edge_data(c, a, default={timestamp_key:'9999-12-31 23:59:59'})[timestamp_key]
    
    birth_ac =  min(t_ac, t_ca ) # either both edges exist and it's a real min(), or only one exists, then i will for sure get it.
    
    early_closing_time = min(t_ac, t_ca )
    
    t_ab = triangle.get_edge_data(a, b, default={timestamp_key:'9999-12-31 23:59:59'})[timestamp_key]
    t_ba = triangle.get_edge_data(b, a, default={timestamp_key:'9999-12-31 23:59:59'})[timestamp_key]
    
    birth_ab = min(t_ab,t_ba)
    
    t_bc = triangle.get_edge_data(b, c, default={timestamp_key:'9999-12-31 23:59:59'})[timestamp_key]
    t_cb = triangle.get_edge_data(c, b, default={timestamp_key:'9999-12-31 23:59:59'})[timestamp_key]
    
    birth_bc = min(t_bc,t_cb)
    
    early_opening_time = max(birth_ab, birth_bc)
   
    return early_opening_time, early_closing_time

def compute_triadic_closure_delays(save_name, result_path):
    
    from datetime import datetime

    import pandas as pd

    from collections import defaultdict
    
    k = save_name
    fname = result_path

    print(k)
    census = defaultdict(int)  
    triadic_closure_delays_days = defaultdict(int)
    time_closure = defaultdict(int)
    how_many_instant = 0

    agg = enum_commons.load_census_results(SNAP_SETTINGS_PATH=fname)
    total = sum([v for k,v in agg["census"].items() if directed_census_utils.mapping_census_to_baseline[k] in [6,7,8,9,10,11,12] ])

    # for triangle in tqdm(get_stored_triads_debug(fname),total=total ):
    for triangle in tqdm(enum_commons.get_stored_triads(fname),total=total ):
        # triangle is a nx graph , subgraph from the snapshot
        # here i show example of triadic census, we can do what we want here
        #tt = directed_census_utils.get_tricode_name(triangle)#.get_triad_type(triangle, v,u,w)
        # census[tt] +=1 

        res, edges_by_time = directed_census_utils.define_triad_order_new_timestamp_key(triangle, timestamp_key ='timestamp')
        t0,t1 = get_early_times_for_delay(triangle, res["ordered_triad"], edges_by_time, timestamp_key="timestamp")

        time_closure[t1]+=1

        delta = pd.Timestamp(t1) - pd.Timestamp(t0) #datetime.strptime(day1, '%Y-%m-%d')  - datetime.strptime(day0, '%Y-%m-%d')
        delta_days = delta.days
        triadic_closure_delays_days[delta_days] +=1

        if delta.total_seconds() == 0:
            how_many_instant += 1

        #break
    #census

    #how_many_instant
    triadic_closure_delays_days = dict(sorted(triadic_closure_delays_days.items()))
    time_closure = dict(sorted(time_closure.items()))

    # get closures per day because currently is closures per timestamp
    day_closure= defaultdict(int)
    
    for d,v in time_closure.items():
        day_closure[d[:10]] += v
    
    infos_delay = {
        "how_many_instant":how_many_instant, 
        "triadic_closure_delays_days":triadic_closure_delays_days,
        "time_closure":time_closure,
        "day_closure": day_closure
    }

    #import pickle
    #with open(f"demofiles/triadic_data_delays/{k}.pkl","wb+") as f:
        #pickle.dump(infos_delay,f)
        
    return infos_delay

def compute_links_per_day(result_path, timestamp_key="timestamp"):
    
    G = enum_commons.get_original_snapshot(result_path)
    link_ts = [ e[2][timestamp_key][:10] for e in G.edges(data=True)]        
    # print(len(link_ts), len(set(link_ts))) # corresponds to every ts of the projected graph
    links_per_day = OrderedDict( sorted(Counter(link_ts).items() ))
        
    return links_per_day

