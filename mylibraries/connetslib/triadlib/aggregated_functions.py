import os
import pickle
from collections import Counter, OrderedDict,defaultdict

# 0 - 5 open triads 6 casi
# 6 - 12 closed triads 7 casi
mapping_census_to_baseline = {'003': 13, #null triad
 '012': 14, # diadic
 '102': 15, # diadic
 '021D': 0,  '021U': 3,  '021C': 1,  '111D': 4,  '111U': 2,  '030T': 6,
 '030C': 7, '201': 5,  '120D': 10,  '120U': 8,  '120C': 9,  '210': 11,
 '300': 12
}



def aggregate_results(path):
    
    #thanks to Galdeman Alessia
    import os
    import json
    import gzip
    
    # file = [json.load(gzip.open(f, 'rt', encoding='utf-8')) for f in [ path+name for name in os.listdir(path)]]
    file = [json.load(gzip.open(f, 'rt', encoding='utf-8')) for f in [ os.path.join(path,name) for name in os.listdir(path)]]

    aggregated_result = {key: [partial[key] for partial in file] for key in file[0].keys()}
    super_agg = {key:{p:sum([d[p] for d in values]) for p in values[0].keys()} for  key,values in aggregated_result.items()}
    return super_agg

"""
def aggregate_from_folder(DIR):
    pageranks_open = {'000': 0, '001': 0, '010': 0, '011': 0, '100': 0, '101': 0, '110': 0, '111': 0}
    outdegrees_open = {'000': 0, '001': 0, '010': 0, '011': 0, '100': 0, '101': 0, '110': 0, '111': 0}
    istances_open = {'000': 0, '001': 0, '010': 0, '011': 0, '100': 0, '101': 0, '110': 0, '111': 0}

    pageranks_closed = {'000': 0, '001': 0, '010': 0, '011': 0, '100': 0, '101': 0, '110': 0, '111': 0}
    outdegrees_closed = {'000': 0, '001': 0, '010': 0, '011': 0, '100': 0, '101': 0, '110': 0, '111': 0}
    istances_closed = {'000': 0, '001': 0, '010': 0, '011': 0, '100': 0, '101': 0, '110': 0, '111': 0}

    evolutions = {'06': 0, '010': 0, 
                  '16': 0, '17': 0, '19': 0,  
                  '28': 0, '29': 0, '211': 0, 
                  '36': 0, '38': 0, 
                  '49': 0, '410': 0, '411': 0,
                  '511': 0, '512': 0}

    #closing_dates = defaultdict(int)

    closing_edge_pagerank = {'00001': 0, '00010': 0, '00011': 0, '00101': 0, '00110': 0, '00111': 0, '01001': 0, '01010': 0, '01011': 0, '01101': 0, '01110': 0, '01111': 0, '10001': 0, '10010': 0, '10011': 0, '10101': 0, '10110': 0, '10111': 0, '11001': 0, '11010': 0, '11011': 0, '11101': 0, '11110': 0, '11111': 0}
    closing_edge_outdegrees = {'00001': 0, '00010': 0, '00011': 0, '00101': 0, '00110': 0, '00111': 0, '01001': 0, '01010': 0, '01011': 0, '01101': 0, '01110': 0, '01111': 0, '10001': 0, '10010': 0, '10011': 0, '10101': 0, '10110': 0, '10111': 0, '11001': 0, '11010': 0, '11011': 0, '11101': 0, '11110': 0, '11111': 0}
    closing_edge_istance = {'00001': 0, '00010': 0, '00011': 0, '00101': 0, '00110': 0, '00111': 0, '01001': 0, '01010': 0, '01011': 0, '01101': 0, '01110': 0, '01111': 0, '10001': 0, '10010': 0, '10011': 0, '10101': 0, '10110': 0, '10111': 0, '11001': 0, '11010': 0, '11011': 0, '11101': 0, '11110': 0, '11111': 0}

    closing_istance_by_type = {'6-000': 0, '6-001': 0, '6-010': 0, '6-100': 0, '6-111': 0, '7-000': 0, '7-001': 0, '7-010': 0, '7-100': 0, '7-111': 0, '8-000': 0, '8-001': 0, '8-010': 0, '8-100': 0, '8-111': 0, '9-000': 0, '9-001': 0, '9-010': 0, '9-100': 0, '9-111': 0, '10-000': 0, '10-001': 0, '10-010': 0, '10-100': 0, '10-111': 0, '11-000': 0, '11-001': 0, '11-010': 0, '11-100': 0, '11-111': 0, '12-000': 0, '12-001': 0, '12-010': 0, '12-100': 0, '12-111': 0}
        
    #: The names of each type of triad.
    TRIAD_NAMES = ('003', '012', '102', '021D', '021U', '021C', '111D', '111U',
                   '030T', '030C', '201', '120D', '120U', '120C', '210', '300')

    census = {name: 0 for name in TRIAD_NAMES}

    aggregated_results = {
                'census': census,
                'evolutions': evolutions,
                'pageranks_open': pageranks_open,
                'outdegrees_open': outdegrees_open,
                'istances_open': istances_open, 
                'pageranks_closed': pageranks_closed,
                'outdegrees_closed': outdegrees_closed,
                'istances_closed': istances_closed,
                #'closing_dates':closing_dates,
                'closing_edge_pagerank':closing_edge_pagerank,
                'closing_edge_outdegrees':closing_edge_outdegrees,
                'closing_edge_istance':closing_edge_istance,
                'closing_istance_by_type':closing_istance_by_type
                }  

    for f in [os.path.join(DIR, name) for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))]:
        #conteggi[f] = simplecount(f)
        with open(f,"rb") as f:
            r = pickle.load(f)
            #print(map_result)
            for k,obj in r.items():
                for key in obj:
                    aggregated_results[k][key] += obj[key]
    #print(aggregated_results)
    return aggregated_results
"""

def get_census_full(aggregated):
    census = aggregated['census']
    census_mapped = {mapping_census_to_baseline[k]:v for k,v in census.items()}
    #counter = OrderedDict(sorted(census_mapped.items(), key=lambda kv: kv[1], reverse=True))
    counter = OrderedDict(sorted(census_mapped.items(), key=lambda kv: kv[0]))
    return counter

def get_census_open(aggregated):
    
    census = aggregated['census']
    census_mapped = {mapping_census_to_baseline[k]:v for k,v in census.items()}

    open_triads = [0,1,2,3,4,5]
    census_open = {k:v for k,v in census_mapped.items() if k in open_triads}
    #counter = OrderedDict(sorted(census_open.items(), key=lambda kv: kv[1], reverse=True))
    counter = OrderedDict(sorted(census_open.items(), key=lambda kv: kv[0]))
    return counter

def get_census_closed(aggregated):
    census = aggregated['census']
    census_mapped = {mapping_census_to_baseline[k]:v for k,v in census.items()}
    closed_triads = [6,7,8,9,10,11,12]
    census_closed = {k:v for k,v in census_mapped.items() if k in closed_triads}
    #counter = OrderedDict(sorted(census_closed.items(), key=lambda kv: kv[1], reverse=True))
    counter = OrderedDict(sorted(census_closed.items(), key=lambda kv: kv[0]))
    return counter

def get_pageranks_closure_probabilty(aggregated):
    pageranks_open = aggregated['pageranks_open']
    pageranks_closed = aggregated['pageranks_closed']
    d = {}
    for k in pageranks_open:
        total = pageranks_open[k] + pageranks_closed[k]
        #print(pageranks_open[k],pageranks_closed[k], total)
        closure_probability = pageranks_closed[k]/total
        #print(closure_probability)
        d[k] = closure_probability

    counter = OrderedDict(sorted(d.items()))
    return counter

def get_outdegrees_closure_probability(aggregated):
    
    outdegrees_open = aggregated['outdegrees_open']
    outdegrees_closed = aggregated['outdegrees_closed']
    
    d = {}
    for k in outdegrees_open:
        total = outdegrees_open[k] + outdegrees_closed[k]
        #print(outdegrees_open[k],outdegrees_closed[k], total)
        closure_probability = outdegrees_closed[k]/total
        #print(closure_probability)
        d[k] = closure_probability

    counter = OrderedDict(sorted(d.items()))
    return counter

def get_istances_closure_probability(aggregated):
    istances_open = aggregated['istances_open']
    istances_closed =  aggregated['istances_closed']
    d = {}
    for k in istances_open:
        total = istances_open[k] + istances_closed[k]
        #print(istances_open[k],istances_closed[k], total)
        if total>0:
            closure_probability = istances_closed[k]/total
            #print(closure_probability)
            d[k] = closure_probability
        #else:
        #    d[k] = 0
    counter = OrderedDict(sorted(d.items()))
    return counter

def get_open_closure_probability(aggregated):
    evolutions = aggregated['evolutions']
    #print(evolutions)
    counter = { (k[0],k[1:]):v for k,v in evolutions.items()}
    #print(counter)
    closed_by_open = defaultdict(int)

    for state in [0,1,2,3,4,5]:
        for ev, count in counter.items():
            if ev[0] == str(state):
                #print(state,ev,count)
                closed_by_open[state] += count
    #print("closed_by_open",closed_by_open)

    census = aggregated['census']
    census_mapped = {mapping_census_to_baseline[k]:v for k,v in census.items()}

    open_triads = [0,1,2,3,4,5]
    census_open = {k:v for k,v in census_mapped.items() if k in open_triads}
    
    #print(census_open)
    closure_prob_per_open = {}

    for state in [0,1,2,3,4,5]:
        #print(census_open[state], closed_by_open[state])
        #print(state,closed_by_open[state], census_open[state] + census_open[state])
        #closure_prob = closed_by_open[state] / ( census_open[state] + census_open[state])
        closure_prob = closed_by_open[state] / ( census_open[state] + closed_by_open[state])
        closure_prob_per_open[state] = closure_prob
    #closure_prob_per_open
    
    #counter = OrderedDict(sorted(closure_prob_per_open.items(), key=lambda kv: kv[1], reverse=True))
    counter = OrderedDict(sorted(closure_prob_per_open.items(), key=lambda kv: kv[0]))
    
    return counter

def get_evolutions_probability(aggregated):
    
    evolutions = aggregated['evolutions']
    counter = { (k[0],k[1:]):v for k,v in evolutions.items()}
    closed_by_open = defaultdict(int)

    for state in [0,1,2,3,4,5]:
        for ev, count in counter.items():
            if ev[0] == str(state):
                #print(state,ev,count)
                closed_by_open[state] += count
    #closed_by_open
    
    census = aggregated['census']
    census_mapped = {mapping_census_to_baseline[k]:v for k,v in census.items()}

    open_triads = [0,1,2,3,4,5]
    census_open = {k:v for k,v in census_mapped.items() if k in open_triads}
    
    counter = { (k[0],k[1:]):v for k,v in evolutions.items()}

    ev_probs = {}

    for ev, count in counter.items():
        state = int(ev[0])
        #print(state, closed_by_open[state], census_open[state], count)
        closure_prob = count/(census_open[state] + closed_by_open[state])
        #print(closure_prob)
        ev_probs[ev] = closure_prob
    #ev_probs
    #counter = OrderedDict(sorted(ev_probs.items(), key=lambda kv: kv[1], reverse=True))
    counter = OrderedDict(sorted(ev_probs.items(), key=lambda kv: kv[0]))
    return counter

def get_closing_edge_probabilities(aggregated, key_closing ="closing_edge_pagerank", key_open="pageranks_open"):
    closing_edge_pagerank = aggregated[key_closing]
    counter_10 = { k[:3] : v for k,v in closing_edge_pagerank.items() if k[3:] == '10'}
    counter_01 = { k[:3] : v for k,v in closing_edge_pagerank.items() if k[3:] == '01'}
    counter_11 = { k[:3] : v for k,v in closing_edge_pagerank.items() if k[3:] == '11'}
    
    pageranks_open = aggregated[key_open]
    
    pageranks_closed_list = [counter_10,counter_01,counter_11]
    counters = []
    for pageranks_closed in pageranks_closed_list:
        #print(pageranks_closed)
        d = {}
        for k in pageranks_open:
            total = pageranks_open[k] + counter_10[k] + counter_01[k] +counter_11[k]
            #print(k,pageranks_open[k],pageranks_closed[k], total)
            if total!=0:
                closure_probability = pageranks_closed[k]/total
                #print(closure_probability)
                d[k] = closure_probability
            else:
                d[k] = 0

        counter = OrderedDict(sorted(d.items()))
        counters.append(counter)
        #    return counter
    return counters

def get_new_closures_per_day(aggregated):    
    date_time_str = "2018-01-21 16:45:00.000000" # primo link del file growth
    growth_start_date = datetime.strptime(date_time_str, '%Y-%m-%d %H:%M:%S.%f')
    #growth_start_date
    closing_dates = aggregated['closing_dates']
    #len(closing_dates)
    closures_per_day = { datetime.strptime(k, "%d-%m-%Y"):v for k,v in closing_dates.items()}
    new_closures_per_day = { datetime.strptime(k, "%d-%m-%Y"):v for k,v in closing_dates.items()}
    new_closures_per_day = {d:c for d,c in  new_closures_per_day.items() if d>growth_start_date}
    return new_closures_per_day

## TODO:  CONTROLLA: devi avere 9 timestep
def get_new_closures_every_4_days(aggregated):
        
    #intervalli 
    new_closures_per_day = get_new_closures_per_day(aggregated)
    new_closures_per_day_list = [(k,v) for k,v in new_closures_per_day.items()]
        
    new_closures_per_day_list = sorted(new_closures_per_day_list)
    intervalli = math.ceil(len(new_closures_per_day_list)/4)
    
    offset_days = 4
    conteggi_ogni_4_gg = []
    # considero le date come se fossero continue. Le prendo a gruppi di 4 e sommo il conteggio.
    i = 0
    for n in range(intervalli):
        #print(i)
        #print(new_closures_per_day_list[i:i+offset_days])
        elements = new_closures_per_day_list[i:i+offset_days]
        #print([el[1] for el in elements])
        #print("Somma", sum([el[1] for el in elements]))
        conteggi_ogni_4_gg.append(sum([el[1] for el in elements]))

        i+=offset_days

        x = [n for n in range(1,intervalli+1)]
        #y = conteggi_ogni_4_gg
    return {k:v for k,v in zip(x,conteggi_ogni_4_gg)}

### Grafico per valutare in quanti casi ho che istanza di A ==  quella di B o se è !=
def closed_triads_probability_istance(aggregated):
    closing_istance_by_type = aggregated['closing_istance_by_type']
    keys =sorted(list(set([ k.split("-")[1] for k in closing_istance_by_type.keys()])))

    d = { (k.split("-")[1],k.split("-")[0]) :v for k,v in closing_istance_by_type.items() }
    counters_list = []
    for k in keys:
        counter_k = {}
        for i,v in d.items():
            if i[0] == k:
                counter_k[i[1]]=v
        #print(k,counter_k)
        counters_list.append(counter_k)
        
    return keys, counters_list

def group_by_edge_couple(keys,counters_list,equality_criteria):
    
    if equality_criteria == 0:
        group_keys = ["A==B", "A!=B"]
    if equality_criteria == 1:
        group_keys = ["B==C", "B!=C"]
    if equality_criteria == 2 :
        group_keys = ["A==C", "A!=C"]
    print(group_keys)
    counter_equal_istance = {str(k):0 for k in range(6,13)}

    counter_different_istance = {str(k):0 for k in range(6,13)}

    for el in zip(keys,counters_list):
        k,d = el
        if k[equality_criteria] == '1': # bit impostato a 1. Se criterio = 0, sto valutando quando a==b
            #print("equal",k)
            for i in d:
                counter_equal_istance[i]+=d[i]
        else:
            #print("diff",k)
            for i in d:
                counter_different_istance[i]+=d[i]
    
    print(counter_equal_istance)
    print(counter_different_istance)
    
    return group_keys, [counter_equal_istance, counter_different_istance]