import os
import errno

def save_edgelist(df, DATASET_NAME, keys = ["from","to","timestamp","amount"]):
    
    dirname = os.path.join(PROCESSED_DATA_PATH,DATASET_NAME)
    
    try:
        os.mkdir(dirname)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass
    
    edge_list = df[keys].rename(columns= dict(zip(keys,["from","to","timestamp","amount"] ) ))
    edge_list = edge_list.sort_values(by=["timestamp"], ascending=True)
    
    print(edge_list.head(1))
    print(edge_list.shape)
    
    path = f"{DATASET_NAME}.csv.gz"
    path = os.path.join(dirname,path)
    edge_list.to_csv(path, compression='gzip', index=False)
    