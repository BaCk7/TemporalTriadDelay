import os
import errno
import pickle

def load_transactions_dataframe(PROCESSED_DATA_FILEPATH):
    import pandas as pd
    transactions = pd.read_csv(PROCESSED_DATA_FILEPATH, parse_dates=["timestamp"])
    print(transactions.head(1))
    print(transactions.shape)
    return transactions

def make_folder(dirname):
    
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


import json

# https://stackoverflow.com/questions/26646362/numpy-array-is-not-json-serializable
class MyEncoder(json.JSONEncoder):

    """ Special json encoder for numpy types extended for other cases """
    def default(self, obj):
        import numpy as np
        from datetime import datetime
    
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, datetime):
            return obj.__str__()
        return json.JSONEncoder.default(self, obj)


def save_dict_as_compressed_json(mydict, fname="mydict"):
    import json
    import gzip

    fname = fname + ".json.gzip"
    # print(fname)
    with gzip.open(fname, 'wt', encoding="ascii") as zipfile:
        json.dump(mydict, zipfile, indent=4, cls=MyEncoder )
    
         
def read_gzip_json_data(fname):
    import json
    import gzip

    with gzip.open(fname, 'rt', encoding="ascii") as zipfile:
        return json.load(zipfile)

    

# import lzma

def save_snapshot_lzma(G, GRAPH_PATH ):
    import lzma

    with lzma.open(GRAPH_PATH, "wb") as f:
        pickle.dump(G,f)
        
def load_snapshot_lzma(GRAPH_PATH):
    
    import lzma

    print(GRAPH_PATH)
    with lzma.open(GRAPH_PATH,"rb") as f:
        G = pickle.load(f)
        
    print(G)
        
    return G

    
# def save_current_figures_data(FIGURES_DATA, fname="figures_data.json"):
#     import json
#     import gzip
#     # with open(fname,"w+") as f:
#     #     json.dump(FIGURES_DATA, f, indent=4, cls=MyEncoder )
    
#     fname = fname + ".json.gzip"
#     print(fname)
#     with gzip.open(fname, 'wt', encoding="ascii") as zipfile:
#         json.dump(FIGURES_DATA, zipfile, indent=4, cls=MyEncoder )
        

    
    
