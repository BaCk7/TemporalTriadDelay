from datetime import datetime
def now(comment=""):
    print( f"{str(datetime.now())[:19]} {comment}" )
    
import pandas as pd
import os
import errno

def store_csv(df, PROCESSED_DATA_PATH, DATASET_NAME, suffix ="_edges", keys = ["from","to","timestamp","amount"], rename_as = ["from","to","timestamp","amount"],sort_by="timestamp"):

    dirname = os.path.join(PROCESSED_DATA_PATH,DATASET_NAME)

    try:
        os.mkdir(dirname)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass

    edge_list = df[keys].rename(columns= dict(zip(keys, rename_as) ))
    
    if sort_by is not None and sort_by in edge_list:
        
        edge_list = edge_list.sort_values(by=[sort_by], ascending=True)

    print(edge_list.head(1))
    print(edge_list.shape)

    path = f"{DATASET_NAME}{suffix}.csv.gz"
    path = os.path.join(dirname,path)
    print(path)
    edge_list.to_csv(path, compression='gzip', index=False)

    return path


import matplotlib.pyplot as plt; 
import matplotlib.pyplot as plt; 
from matplotlib.dates import DateFormatter
from matplotlib.dates import DayLocator
plt.rcdefaults()
import numpy as np
import matplotlib.colors as mcolors
from datetime import datetime
#import matplotlib.pyplot as plt
SMALL_SIZE = 14
MEDIUM_SIZE = 16
BIGGER_SIZE = 18

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

#import matplotlib.pyplot as plt
#%matplotlib inline

def save_current_image_in_folder(fname, folder_path):
    
    from PIL import Image
    from io import BytesIO
    import os


    
    images_directory = os.path.join(folder_path,"images") # always create a images folder if not available
    make_folder(images_directory)

    FOLDER_IMAGES_PNG = os.path.join(images_directory, "png")
    make_folder(FOLDER_IMAGES_PNG)

    #FOLDER_IMAGES_TIFF = "data/images/tiff/"
    FOLDER_IMAGES_PDF = os.path.join(images_directory, "pdf")
    make_folder(FOLDER_IMAGES_PDF)


    DPI = 300#600

    # print("Note: there are tight layout and grid settings in save_current_image function")
    # plt.tight_layout()
    # plt.grid()

    # save figure
    # (1) save the image in memory in PNG format
    png1 = BytesIO()
    plt.savefig(png1, format='png', dpi = DPI)

    # (2) load this image into PIL
    png2 = Image.open(png1)

    # (3) save as TIFF
    
#     path =  "{}{}.tiff".format(FOLDER_IMAGES_TIFF,fname)
#     print(path,"- dpi:",DPI)
    
#     png2.save(path,format="tiff",compression ="tiff_adobe_deflate")
    
    
    # (3) save as PNG
    path = "{}/{}.png".format(FOLDER_IMAGES_PNG,fname)
    
    print(path,"- dpi:", DPI)
    
    png2.save(path,format="png")
    
    # (3) save as pdf
    path = "{}/{}.pdf".format(FOLDER_IMAGES_PDF,fname)
    
    print(path,"- dpi:",DPI)
    
    plt.savefig(fname= path, format='pdf', dpi = DPI)
    
    
    png1.close()

    #save_current_image(fname)

import seaborn as sns

#increase font size of all elements
sns.set(font_scale=1.0)
sns.set_style("whitegrid")

import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage,AnnotationBbox

def offset_image(coord, name, ax):
    #img = get_flag(name)
    
    path = './triads-graphics/census/{}.png'.format(name) #.title())
    img = plt.imread(path)
    
    im = OffsetImage(img, zoom=0.15)
    im.image.axes = ax

    ab = AnnotationBbox(im, (coord, 0),  xybox=(0., -19.), frameon=False,
                        xycoords='data',  boxcoords="offset points", pad=0)

    ax.add_artist(ab)

def offset_image_ev(coord, name, ax):
    #img = get_flag(name)
    
    path = './triads-graphics/ev-c/ev{}.png'.format(name)
    # path = './triads-graphics/ev-vert/ev0-6.png'
    img = plt.imread(path)
    
    im = OffsetImage(img, zoom=0.11)
    im.image.axes = ax

    ab = AnnotationBbox(im, (coord, 0),  xybox=(0., -15.), frameon=False,
                        xycoords='data',  boxcoords="offset points", pad=0)

    ax.add_artist(ab)

def get_closing_edge_counters(aggregated, key_closing ="closing_edge_pagerank"):
    closing_edge_pagerank = aggregated[key_closing]
    counter_10 = { k[:3] : v for k,v in closing_edge_pagerank.items() if k[3:] == '10'}
    counter_01 = { k[:3] : v for k,v in closing_edge_pagerank.items() if k[3:] == '01'}
    counter_11 = { k[:3] : v for k,v in closing_edge_pagerank.items() if k[3:] == '11'}
    
    return [counter_10, counter_01, counter_11]


import scipy.stats

def get_zscore_p_value_for_table_normalized(key, agg_res, agg_res_avg, alpha=0.01, verbose = False):
    
    # return the row e.g. 3-6:-0.6, 3-7:+0.3
    row = {}
    
    total = sum([ agg_res[key][ev] for ev in agg_res[key]] )
        
    
    for ev in agg_res[key]:
        f_real = agg_res[key][ev] 
        f_random_avg =  agg_res_avg[key][ev][0]
        f_random_std = agg_res_avg[key][ev][1]
    

        
        if  agg_res_avg[key][ev][1] > 0: # Can't divide by zero
            z = ( f_real - f_random_avg ) / f_random_std
        else:
            # print()
            z = 0
            
        row[ev]=z
        

        # finding p-value
        p_value = scipy.stats.norm.sf(abs(z))*2
        
        if p_value < alpha:
            # s = "{:.2f}({:.3f})".format(z, p_value)
            #s = "{:.2f}".format(z)#, p_value)
            s = "{:.0f} {:.2f}\% ({:.2f})".format(f_real,f_real/total*100, z)
        else:
            #s = "NS"
            s = "{:.0f} {:.2f}\% (NS)".format(f_real,f_real/total*100)
        
        row[ev] = s
        
        if verbose:
            print("Rule:",ev)
            print("x", f_real, "r_avg", f_random_avg, "r_std", f_random_std,"zscore",z)
            print('p value is : ' + str(p_value))
            
    return row


# Plots

def plot_open(cnt, k, focus):
    
    import matplotlib as mpl
    from collections import OrderedDict
    
    cnt_perc = { k: i/sum(cnt.values())  for k,i in cnt.items() }

    plotting.draw_barplot(cnt_perc,
             title="",#"Distribution of open triads",
             xlabel="", 
             ylabel='',
             legend=False,
             ylim=(0,1),
             normalized=False)

    ax = plt.gca()
    countries = [ str(k) for k in cnt.keys()]
    for i, c in enumerate(countries):
        offset_image(i, c, ax)


    #ax.get_xaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax.get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax.grid(visible=True, which='major', color='darkgrey', linewidth=1.0)
    ax.grid(visible=True, which='minor', color='darkgrey', linewidth=0.5)


    save_current_image_in_folder(fname=f"{k}-{focus}", folder_path=".")
    plt.show()

    
def plot_closed(cnt, k, focus):
    import matplotlib as mpl
    from collections import OrderedDict
    cnt_perc = { k: i/sum(cnt.values())  for k,i in cnt.items() }

    plotting.draw_barplot(cnt_perc,
             title="", #"Distribution of closed triads",
             xlabel="", #"Triad type", 
             ylabel="", #'Percentage',
             legend=False,
             ylim=(0,1),
             normalized=False)

    ax = plt.gca()
    countries = [ str(k) for k in cnt.keys()]
    for i, c in enumerate(countries):
        offset_image(i, c, ax)


    #ax.get_xaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax.get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax.grid(visible=True, which='major', color='darkgrey', linewidth=1.0)
    ax.grid(visible=True, which='minor', color='darkgrey', linewidth=0.5)

    save_current_image_in_folder(fname=f"{k}-{focus}", folder_path=".")
    plt.show()

def plot_evolutions(cnt, k, focus):

    import matplotlib as mpl
    from collections import OrderedDict   
 
    plotting.draw_barplot_evolutions(cnt,
                 title="",#"Closure probability for each evolution",
                 xlabel="",#"Open triad", 
                 ylabel="",# 'Percentuale', 
                 ylim=(0, 0.35),
                 normalized=True,
                 figsize=(11,3),#(16,10),
                 legend = False,
                 colors = None)

    ax = plt.gca()
    ax.set_xticklabels(["" for l in  ax.get_xticklabels()])

    countries = [ str(k[0])+"-"+str(k[1]) for k in cnt.keys()]
    for i, c in enumerate(countries):
        offset_image_ev(i, c, ax)

    #ax.get_xaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax.get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax.grid(visible=True, which='major', color='darkgrey', linewidth=1.0)
    ax.grid(visible=True, which='minor', color='darkgrey', linewidth=0.5)

    save_current_image_in_folder(fname=f"{k}-{focus}", folder_path=".")
    plt.show()