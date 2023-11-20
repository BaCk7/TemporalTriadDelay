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
import connetslib.triadlib.triad_metrics as triad_metrics


def build_graph(transaction_filepath, start_time = None, end_time = None):

    transactions = glib_storage.load_transactions_dataframe(transaction_filepath)
    transactions['from'] = transactions['from'].astype(str)
    transactions['to'] = transactions['to'].astype(str)
    # TODO: prendi min/max o fai un mese
    if start_time is None:
        start_time = glib_builder.get_start_time(transactions, timestamp_key="timestamp")

    if end_time is None:
        end_time = glib_builder.get_end_time(transactions, timestamp_key="timestamp")

    G = glib_builder.simple_construction_from_dataframe(transactions, 
                                      start_string = start_time,
                                      end_string = end_time, 
                                      GRAPH_TYPE = "digraph", 
                                      add_edge_func = glib_builder.add_edge_function_weight_temporal,
                                     )
    
    return G
    

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
#SMALL_SIZE = 14
#MEDIUM_SIZE = 16
#BIGGER_SIZE = 18

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
from collections import OrderedDict

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
    
    k = save_name
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