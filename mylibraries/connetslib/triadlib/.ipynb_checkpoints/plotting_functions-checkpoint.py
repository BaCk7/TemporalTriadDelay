# imports
import json
import pickle
import itertools as it

import networkx as nx
import pandas as pd

from datetime import datetime
from datetime import timedelta

from collections import Counter, OrderedDict,defaultdict
from statsmodels.distributions.empirical_distribution import ECDF

#matplotlib
import matplotlib.pyplot as plt; 
from matplotlib.dates import DateFormatter
from matplotlib.dates import DayLocator
plt.rcdefaults()
import numpy as np
import matplotlib.colors as mcolors

#display
from IPython.display import clear_output
from IPython.display import Image
from IPython.core.display import HTML 
from IPython.display import display # to display images


from IPython.display import display, HTML
import os

# 0 - 5 open triads 6 casi
# 6 - 12 closed triads 7 casi
mapping_census_to_baseline = {'003': 13, #null triad
 '012': 14, # diadic
 '102': 15, # diadic
 '021D': 0,  '021U': 3,  '021C': 1,  '111D': 4,  '111U': 2,  '030T': 6,
 '030C': 7, '201': 5,  '120D': 10,  '120U': 8,  '120C': 9,  '210': 11,
 '300': 12
}

# https://stackoverflow.com/a/42665818
clist = [(0, "darkblue"), (0.125, "blue"), (0.25, "green"), (0.4, "green"),  (0.5, "yellow"), 
         (0.7, "orange"), (0.75, "red"), (1, "red")]

rvb = mcolors.LinearSegmentedColormap.from_list("", clist)


def insert_image(folder="Images/huang",image="4a.PNG",height=175):
    html = '<img src="{}" style="display:inline;height:{}px;margin:1px"/>'.format(os.path.join(folder, image),height)
    display(HTML(html))
"""    
def draw_barplot(counter,title="Triads evolution", ylabel='Closure probabilty', xlabel="", 
                 figsize=None, normalized = False, legend=True, xticks = None, ylim = None, save=False):
    objects = counter.keys()

    y_pos = np.arange(len(objects))
    
    performance = [i for i in counter.values() ]
    
    if normalized:
        performance = [i/sum(counter.values()) for i in counter.values() ]
    
    if figsize:
        plt.figure(figsize=figsize)
    #figsize=(18,10)
    
    if ylim:
        plt.ylim(ylim)
        
    for i,e in enumerate(performance):
        plt.bar(y_pos[i], performance[i], color=rvb(y_pos/ len(objects))[i], align='center', alpha=0.8)
    
    if not xticks:
        plt.xticks(y_pos, objects)
    else:
        plt.xticks(y_pos, labels = xticks,rotation='horizontal', fontsize = 10)

    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.title(title)
    
    if legend:
        plt.legend(objects)
    
    if save:
        fig1 = plt.gcf()
        fig1.savefig(title+'.png',dpi=1200)
    #plt.show()
"""

def draw_barplot(counter,title="Triads evolution", ylabel='Closure probabilty', xlabel="", 
                 figsize=None, normalized = False, legend=True, xticks = None, ylim = None, save=False):
    objects = counter.keys()

    y_pos = np.arange(len(objects))
    
    performance = [i for i in counter.values() ]
    
    if normalized:
        performance = [i/sum(counter.values()) for i in counter.values() ]
    
    if figsize:
        plt.figure(figsize=figsize)
    #figsize=(18,10)
    
    if ylim:
        plt.ylim(ylim)
        
    for i,e in enumerate(performance):
        plt.bar(y_pos[i], performance[i], color=rvb(y_pos/ len(objects))[i], align='center', alpha=0.8)
    
    if not xticks:
        plt.xticks(y_pos, objects)
    else:
        plt.xticks(y_pos, labels = xticks,rotation='horizontal', fontsize = 10)

    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.title(title)
    
    if legend:
        plt.legend(objects)
    
    if save:
        fig1 = plt.gcf()
        fig1.savefig(title+'.png',dpi=1200)
    #plt.show()

"""    
def draw_stacked_barplot(counters, title="Triads evolution", ylabel='Closure probabilty', xlabel="", 
                 figsize=None, normalized = False, legend=True, skip_zeros=False, ylim = None, save=False):
    
    counter = counters[0]
    
    if skip_zeros:
        counter = {k:v for k,v in counter.items() if v !=0}
    
    objects = counter.keys()
    y_pos = np.arange(len(objects))
    performance = [i for i in counter.values() ]
    plt.bar(y_pos,performance,color='blue',edgecolor='black', alpha=0.8)

    counter1 = counters[1]
    if skip_zeros:
        counter1 = {k:v for k,v in counter1.items() if v !=0}
        
    objects1 = counter1.keys()
    y_pos1 = np.arange(len(objects1))
    performance1 = [i for i in counter1.values() ]
    plt.bar(y_pos1,performance1,color='limegreen', align='center', alpha=0.8, edgecolor='black', bottom=performance)


    counter2 = counters[2]
    
    if skip_zeros:
        counter2 = {k:v for k,v in counter2.items() if v !=0}
        
    objects2 = counter2.keys()
    y_pos2 = np.arange(len(objects2))
    performance2 = [i for i in counter2.values() ]

    for k,v in enumerate(performance): 
        performance1[k] += v

    plt.bar(y_pos2,performance2,color='red',edgecolor='black', bottom = performance1, alpha=0.8)

    plt.xticks(y_pos, objects)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.title(title)
    
    if ylim:
        plt.ylim(ylim)
        
    if legend:
        #plt.legend(objects)
        plt.legend(["A->C", "A<-C","A<->C"])
    if save:
        fig1 = plt.gcf()
        fig1.savefig(title+'.png',dpi=1200)

    #plt.show()
"""
def draw_stacked_barplot(counters, title="Triads evolution", ylabel='Closure probabilty', xlabel="", 
                 figsize=None, normalized = False, legend=True, skip_zeros=False, ylim = None, save=False, colors=[]):
    
    i = 0
    
    if not colors:
        colors=['blue','limegreen',"red","orange","yellow","purple"]
    
    counters_total = sum([ sum(list(counter.values())) for counter in counters])
    
    counter = counters[0]
    
    if skip_zeros:
        counter = {k:v for k,v in counter.items() if v !=0}
    
    objects = counter.keys()
    y_pos = np.arange(len(objects))
    performance = [i for i in counter.values() ]
    
    if normalized:
        performance = [i/counters_total for i in counter.values() ]
    
    plt.bar(y_pos,performance,color=colors[i],edgecolor='black', alpha=0.8)
    

    for counter in counters[1:]:
        i +=1 
        counter1 = counter
        if skip_zeros:
            counter1 = {k:v for k,v in counter1.items() if v !=0}

        objects1 = counter1.keys()
        y_pos1 = np.arange(len(objects1))
        performance1 = [i for i in counter1.values() ]
        
        if normalized:
            performance1 = [i/counters_total for i in counter1.values() ]
        
        plt.bar(y_pos1,performance1,color=colors[i], align='center', alpha=0.8, edgecolor='black', bottom=performance)
        
        for k,v in enumerate(performance1): 
            performance[k] += v
            
    plt.xticks(y_pos, objects)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.title(title)
    
    if ylim:
        plt.ylim(ylim)
        
    if legend:
        plt.legend(legend)
    else:
        plt.legend(["A->C", "A<-C","A<->C"])
    if save:
        fig1 = plt.gcf()
        fig1.savefig(title+'.png',dpi=1200)

    #plt.show()

    
def draw_barplot_time(counter, title="Barplot time", ylabel='Number', xlabel="", 
                 figsize=None, normalized = False, legend=True, xticks = None):

    formatter = DateFormatter('%d/%m')

    x = [ pair for pair in counter.keys()]
    y = [ pair for pair in counter.values()]

    ax  = plt.figure(figsize=figsize)
    plt.bar(x,y)

    plt.gcf().axes[0].xaxis.set_major_formatter(formatter)
    plt.gcf().axes[0].xaxis.set_major_locator(DayLocator(interval=5))
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)#('Triad types')

    #plt.yticks(np.arange(0,300000, 50000))
    
    if xticks:
        xticks = [ "T"+str(time) for time in counter.keys()]
        plt.xticks(x, labels = xticks, rotation='horizontal', fontsize = 10)
    else:
        plt.xticks(rotation='horizontal', fontsize = 8)
    fig1 = plt.gcf()
    fig1.savefig(title+'.png',dpi=1200)
    #plt.show()
    
def evolutions_counter_formatter(counter):
    counter_new = {}
    for k,v in counter.items():
        k_new = k[0] + "->" + k[1]
        counter_new[k_new] = v 
    return counter_new
    
def draw_barplot_evolutions(counter,title="Triads evolution", ylabel='Closure probabilty', xlabel="", 
                 figsize=None, normalized = False, legend=True, xticks = None, colors = None,
                 ylim=None, save=False):
    

    counter = OrderedDict(sorted(counter.items(), key= lambda v: ( int(v[0][0]),int(v[0][1]))))
    
    objects = counter.keys()

    y_pos = np.arange(len(objects))
    
    performance = [i for i in counter.values() ]
    
    if normalized:
        performance = [i/sum(counter.values()) for i in counter.values() ]
    
    if figsize: 
        plt.figure(figsize=figsize)
        #figsize=(18,10)
    

    if ylim:
        plt.ylim(ylim)
        
    if colors:
        colors = colors
    else:
        #colors = rvb(y_pos/ len(objects))
        colors =  { str(i):c for i,c in zip(range(6,13),["orangered","firebrick","lightgreen","yellow","orange","deepskyblue","plum"])}
    
        
    objects = [ o for o in objects]
    print(objects)
    barWidth = 0.8
    for i,e in enumerate(performance):
        plt.bar(y_pos[i], performance[i], width=barWidth,
                color= colors[objects[i][1]], align='center', alpha=0.8, label=str(objects[i][1]))
        #print(i)

    
    if not xticks:
        plt.xticks(y_pos, [o[0]+"-->"+o[1] for o in objects])# objects)
    else:
        plt.xticks(y_pos, labels = xticks,rotation='horizontal', fontsize = 10)

    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.title(title)
    
    if legend:
        #plt.legend()#([o[0] for o in objects] )
        ax = plt.gca()
        handles, labels = ax.get_legend_handles_labels()
        unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
        ax.legend(*zip(*unique))
    

    #for i,e in enumerate(performance):     
    #    if i < len(performance) -1 and objects[i][0] != objects[i+1][0]:
    #        plt.axvline(y_pos[i] + barWidth/2 + barWidth/8, color='k', linestyle='-', linewidth=0.7)
            
    #fig1 = plt.gcf()
    #fig1.savefig(title+'.png',dpi=1200)
    #plt.show()
    
    if save:
        fig1 = plt.gcf()
        fig1.savefig(title+'.png',dpi=1200)
        

"""
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score, recall_score

def plot_model_performance_table(model, X_test,Y_test):
    rows_list =[]

    for cl in range(0,6):
        #print("Triad type",cl)
        X_test_open_class = [ (x,y) for x,y in zip(X_test,Y_test) if x.index(1) == cl ] 
        Y_test_open_class = [ y for x,y in X_test_open_class]
        X_test_open_class = [ x for x,y in X_test_open_class]
        #print(len(X_test_open_class))
        result = model.predict(X_test_open_class)
        #print("Acc", accuracy_score(Y_test_open_class,result))
        #print("Prec", precision_score(Y_test_open_class,result))
        #print("Recall", recall_score(Y_test_open_class,result))
        #print("F1",f1_score(Y_test_open_class,result))
        acc = accuracy_score(Y_test_open_class,result)
        prec = precision_score(Y_test_open_class,result)
        rec = recall_score(Y_test_open_class,result)
        f1 = f1_score(Y_test_open_class,result)

        rows_list.append([cl,acc,prec,rec,f1])

    df = pd.DataFrame(rows_list, columns=['Type', 'Acc', 'Prec','Recall',"F1"])               

    #display(df)
    return df
"""
#
# LDA    
#    
def plot_scatter_similarities_pairs(rows):

    plt.figure(figsize=(20,5))

    plt.subplot(1, 3, 1)
    
    xs = [tup[0] for tup in rows]
    ys = [tup[1] for tup in rows]
    ls = [tup[3] for tup in rows]

    colors = ["cornflowerblue","salmon"]

    plt.scatter(xs, ys, c=[ colors[l] for l in ls], alpha=0.5)
    plt.xlabel('AB')
    plt.ylabel('BC')

    
    plt.subplot(1,3,2)
    xs = [tup[1] for tup in rows]
    ys = [tup[2] for tup in rows]
    ls = [tup[3] for tup in rows]

    colors = ["cornflowerblue","salmon"]

    plt.scatter(xs, ys, c=[ colors[l] for l in ls], alpha=0.5)
    plt.xlabel('BC')
    plt.ylabel('AC')

    plt.subplot(1,3,3)
    xs = [tup[0] for tup in rows]
    ys = [tup[2] for tup in rows]
    ls = [tup[3] for tup in rows]

    colors = ["cornflowerblue","salmon"]

    plt.scatter(xs, ys, c=[ colors[l] for l in ls], alpha=0.5)
    plt.xlabel('AB')
    plt.ylabel('AC')
    
def plot_scatter_similarities_close_state(rows):

    plt.figure(figsize=(20,5))

    plt.subplot(1, 3, 1)
    
    xs = [tup[0] for tup in rows]
    ys = [tup[0] for tup in rows]
    ls = [tup[3] for tup in rows]

    colors = ["cornflowerblue","salmon"]

    plt.scatter(xs, ys, c=[ colors[l] for l in ls], alpha=0.5)
    plt.xlabel('AB')
    plt.ylabel('AB')

    
    plt.subplot(1,3,2)
    xs = [tup[1] for tup in rows]
    ys = [tup[1] for tup in rows]
    ls = [tup[3] for tup in rows]

    colors = ["cornflowerblue","salmon"]

    plt.scatter(xs, ys, c=[ colors[l] for l in ls], alpha=0.5)
    plt.xlabel('BC')
    plt.ylabel('BC')

    plt.subplot(1,3,3)
    xs = [tup[2] for tup in rows]
    ys = [tup[2] for tup in rows]
    ls = [tup[3] for tup in rows]

    colors = ["cornflowerblue","salmon"]

    plt.scatter(xs, ys, c=[ colors[l] for l in ls], alpha=0.5)
    plt.xlabel('AC')
    plt.ylabel('AC')
    


def draw_triadic_evolutions(agg_res):
    
    #thanks to Galdeman Alessia
    # import os
    # print(os.getcwd())
    
    from pathlib import Path
    
    base_path = Path(__file__).parent
    file_path = (base_path / "./triads-graphics/").resolve()
    # print(file_path)

    fig, ax = plt.subplots(5, 3, figsize = (15,25))
    supports_list = [(k,s) for k,s in agg_res['evolutions'].items()]
    supports_list.sort(key = lambda x:x[1], reverse = True)
    for i,info in enumerate(supports_list):
        row, col = i//3, i%3
        ev, s = info
        ifrom, ito = ev[0], ev[1:]
        img_path = f"{file_path}/ger{ifrom}-{ito}.png"
        # print(img_path)
        ger = plt.imread(img_path)
        ax[row, col].axis('off')
        ax[row, col].set_title(f"   support:\n   {s}\n", color = '#193286', weight = 'bold')
        ax[row,col].imshow(ger)
        
        

def draw_triadic_evolutions_significativity(agg_res, agg_res_avg,verbose=False):
    
    #thanks to Galdeman Alessia
    # import os
    # print(os.getcwd())
    
    from pathlib import Path
    
    base_path = Path(__file__).parent
    file_path = (base_path / "./triads-graphics/").resolve()
    # print(file_path)

    fig, ax = plt.subplots(5, 3, figsize = (15,25))
    supports_list = [(k,s) for k,s in agg_res['evolutions'].items()]
    supports_list.sort(key = lambda x:x[1], reverse = True)
    
    for i,info in enumerate(supports_list):
        row, col = i//3, i%3
        ev, s = info
        ifrom, ito = ev[0], ev[1:]
        img_path = f"{file_path}/ger{ifrom}-{ito}.png"
        # print(img_path)
        ger = plt.imread(img_path)
        
        f_real = agg_res['evolutions'][ev] 
        f_random_avg =  agg_res_avg['evolutions'][ev][0]
        f_random_std = agg_res_avg['evolutions'][ev][1]
        
        if verbose:
            print(f_real, f_random_avg, f_random_std)
        
        if  agg_res_avg['evolutions'][ev][1] > 0: # Can't divide by zero
            z = ( f_real - f_random_avg ) / f_random_std
        else:
            # print()
            z = 0
        
        ax[row, col].axis('off')
        ax[row, col].set_title(f"   support:\n   {s} \n   z-score:\n   {z:.3f}   \n", color = '#193286', weight = 'bold')
        ax[row,col].imshow(ger)        