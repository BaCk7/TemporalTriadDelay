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


import matplotlib.pyplot as plt; 
import matplotlib.pyplot as plt; 
from matplotlib.dates import DateFormatter
from matplotlib.dates import DayLocator
plt.rcdefaults()
import numpy as np
import matplotlib.colors as mcolors
from datetime import datetime

def plot_triads_delay(data, save_name):

    focus="triads-delay"
    #for k in info_delays_all:
    #print(k)
    #selected = info_delays_all[k] 
    k = save_name
    selected = data["info_delays"]
    triadic_closure_delays_days = selected["triadic_closure_delays_days"]

    x = [ pair[0] for pair in triadic_closure_delays_days.items()]
    y = [ pair[1] for pair in triadic_closure_delays_days.items()]
    y = np.cumsum([ pair[1] for pair in sorted(triadic_closure_delays_days.items())])/sum([ pair[1] for pair in triadic_closure_delays_days.items()])

    plt.figure(figsize=(6,4))
    plt.plot(x,y,color=None, lw = 5)

    # line_dates=[4,8,12,20,30]
    # for l in line_dates:
    #     plt.axvline(l,color='r',alpha=0.5)
    
    plt.ylim((0,1))
    plt.xscale("log")
    plt.grid()
    
    # ax = plt.gca()
    # #ax.get_xaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
    # ax.get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
    # ax.grid(visible=True, which='major', color='darkgrey', linewidth=1.0)
    # ax.grid(visible=True, which='minor', color='darkgrey', linewidth=0.5)
    
    save_current_image_in_folder(fname=f"{k}-{focus}", folder_path=".")
    
    plt.show()
    
    
def plot_ratio(data, save_name, running_avg_window = 7):
    import numpy as np
    from scipy.ndimage.filters import uniform_filter1d
    
    focus = "ratio"

    interval = 90
    #plt.figure(figsize=(6,4))
    fig, ax = plt.subplots(figsize=(8,5))      

    # Closures per day
    selected = data["info_delays"] #info_delays_all[k]  
    day_closure = selected["day_closure"]
    x2 = [ datetime.strptime(pair[0], '%Y-%m-%d') for pair in day_closure.items()]
    y2 = [ pair[1] for pair in day_closure.items()]
    
    y2 = uniform_filter1d(y2, size = running_avg_window)
    
    plt.plot(x2,y2,alpha=0.9, color="tab:blue", label="Triads", zorder=1)
    # plt.fill_between(x2, y2, alpha=0.4, color="tab:blue", zorder=1)

    # Links per day
    links_per_day = data["links_per_day"]
    x = [ datetime.strptime(pair[0], '%Y-%m-%d') for pair in links_per_day.items()]
    y = [ pair[1] for pair in links_per_day.items()]
    y = uniform_filter1d(y, size = running_avg_window)
    
    plt.plot(x,y, alpha=0.9, color="tab:orange", label="Links", zorder=2)
    # plt.fill_between(x, y, alpha=0.4, color="tab:orange", zorder=2)
    
    
    formatter = DateFormatter('%b %Y')
    plt.gcf().axes[0].xaxis.set_major_formatter(formatter)
    plt.gcf().axes[0].xaxis.set_major_locator(DayLocator(interval=interval))
    #plt.gcf().axes[0].xaxis.grid(True, which="minor")
    #plt.gcf().autofmt_xdate()
    plt.ylabel('Links and triads')

    plt.xticks(rotation='-25', fontsize = 10)
    plt.yscale("log")
    # plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0), useMathText=True)

    plt.legend(loc = "upper left")
    # Ratio 
    
    ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel('Triads/Links')#, color=color)  # we already handled the x-label with ax1
    
    # Note: triangles / links. There cannot be a closure without links. We could have link not leading to closures, but it's unlikely.
    # So we should have that len(triangles_per_day) <= len(links_per_day)
    x3 = []
    y3 = []
    for d,v in links_per_day.items():
        x3.append(d)
        if d in day_closure:
            n = day_closure[d]
        else:
            n = 0
        ratio = n/v
        y3.append(ratio)
    x3 = [ datetime.strptime(d, '%Y-%m-%d') for d in x3]
    y3 = uniform_filter1d(y3, size = running_avg_window)
    
    plt.plot(x3,y3,color="firebrick", label="Ratio", lw=2)
    
    plt.ylim((0,None))
    # plt.grid()
    plt.legend(loc="upper right")
    # ax = plt.gca()
    # #ax.get_xaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
    # ax.get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
    # ax.grid(visible=True, which='major', color='darkgrey', linewidth=1.0)
    # ax.grid(visible=True, which='minor', color='darkgrey', linewidth=0.5)
    
    save_current_image_in_folder(fname=f"{k}-{focus}", folder_path=".")

    plt.show()