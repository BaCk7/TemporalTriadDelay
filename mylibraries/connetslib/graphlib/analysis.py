# Where to store
from . import storage as glib_storage
#from . import storage as glib_storage
import os
import pprint

# list all files ending with stats.json.gzip - each one is a row for the dataframe stored["subdirname"]
# crea dataframe
# metti in libreria per report

# folder = stored["subdirname"]
# get_snapshot_stats(folder)


def read_setting_file(folder):
    
    # setting usato ( quelli che ci sono già per ora, se ci sono lol)
    setting_path = os.path.join(folder,"SETTING.json.gzip")
    setting = glib_storage.read_gzip_json_data(setting_path)
    return setting


def get_snapshot_stats(folder):
    
    return sorted([ os.path.join(folder,fname)  for fname in os.listdir(folder) if fname.endswith("stats.json.gzip")  ])
    

def get_snapshot_stats_df(stat_files):

    import pandas as pd
    
    stat_rows = []
    
    for fpath in stat_files:
        row = glib_storage.read_gzip_json_data(fpath)
        stat_rows.append(row)
        
    return pd.DataFrame(stat_rows).sort_values("Start").set_index("index")


def get_snap_files(folder):
    
    # lista degli snapshot salvati disponibili
    snap_files = sorted([ os.path.join(folder,fname) for fname in os.listdir(folder) if fname.endswith(".pkl.xz")  ])
    return snap_files
    
    
# funzione che mostra lista degli snapshot disponibili, stats e settings usati ( quelli che ci sono già per ora, se ci sono lol)
def get_recap(folder, return_also_files = True):
    
    print()
    print("RECAP:")
    
    
    setting = read_setting_file(folder)
    pprint.pprint(setting)
    
    # stats
    stat_files = get_snapshot_stats(folder)
    df_stats = get_snapshot_stats_df(stat_files)
    print(f"There are {df_stats.shape[0]} stat files")
    
    # lista degli snapshot salvati disponibili
    snap_files = get_snap_files(folder)
    print(f"There are {len(snap_files)} snap files")
    print()
    
    if return_also_files:
        return {"setting": setting , "stat_files": stat_files, "snap_files":snap_files}
    else:
        return {"setting": setting}
    
    #get_recap(folder, return_also_files= False)
    
    
# funzione che carica dato in input subdir e snap index

def load_selected_snapshot(folder, SNAP_INDEX):
    
    gpath = os.path.join(folder,f"snap_{SNAP_INDEX}.pkl.xz")
    G = glib_storage.load_snapshot_lzma(gpath)
    
    return G

def load_last_snapshot(folder):
    last_snap = glib_analysis.get_snap_files(folder)[-1]
    G = glib_storage.load_snapshot_lzma(last_snap)
    return G
