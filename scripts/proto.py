import os, sys
import random
import numpy as np
from tqdm import tqdm
import networkx as nx
import itertools
import requests



#import time
import asyncio
import itertools

#from concurrent.futures import ThreadPoolExecutor
#from alexber.utils.thread_locals import exec_in_executor

#import nest_asyncio
#nest_asyncio.apply()



#def take_until_timeout(iterator, max_count=5, timeout=1):
def take_until_timeout(iterator,
                       max_count=5,
                       timeout=2,
                       total_timeout=5,
                       loop=None):
    # https://docs.python.org/3/library/asyncio-task.html#asyncio.wait_for

    async def get_elements(start):
        elements = []
        #with ThreadPoolExecutor() as executor:
        for i in itertools.count():
              if max_count is not None and i >= max_count:
                  break
              if total_timeout is not None and loop.time() - start > total_timeout:
                  break
              try:
                  item = await asyncio.wait_for(
                      asyncio.to_thread(next, iterator),
                      #exec_in_executor(executor, next, iterator),
                      #exec_in_executor(executor, asyncio.to_thread, next, iterator),
                      timeout=timeout
                  )
              except asyncio.TimeoutError:
                  break
              except StopIteration:
                  break
              #yield item#
              elements.append(item)
          #executor.shutdown(wait=False)
        return elements

    loop = asyncio.get_event_loop() if loop is None else loop
    return loop.run_until_complete(get_elements(loop.time()))




def normalize(x, min, max):
    x = float(x)
    if x <= min:
        return 0.0
    if x > max:
        return 0.99999
    return (x - min) / (max - min)


# Retrieves current block height from API
# in case of fail, will return a default block height
def getBlockHeight(default=True):
    if default:
        return 697000
    API_URL = "https://api.blockcypher.com/v1/btc/main"
    try:
        CBR = requests.get(API_URL).json()['height']
        print("Block height used:", CBR)
        return CBR
    except:
        print("Block height not found, using default 697000")
        return 697000

### GENERAL
BASE_TIMESTAMP = 1681234596.2736187
BLOCK_HEIGHT = getBlockHeight()
### LND
LND_RISK_FACTOR = 0.000000015
A_PRIORI_PROB = 0.6
### ECL
MIN_AGE = 505149
MAX_AGE = BLOCK_HEIGHT
MIN_DELAY = 9
MAX_DELAY = 2016
MIN_CAP = 1
MAX_CAP = 100000000
DELAY_RATIO = 0.15
CAPACITY_RATIO = 0.5
AGE_RATIO = 0.35
### CLN
C_RISK_FACTOR = 10
RISK_BIAS = 1
DEFAULT_FUZZ = 0.05
FUZZ = random.uniform(-1, 1)
LOG_SPACE = np.logspace(0, 7, 10**6)


def cost_function(e, amount, proto_type='LND'):
    fee = e['fee_base_msat'] + amount * e['fee_rate']
    if proto_type == 'LND':
        cost = (amount + fee) * e['delay'] * LND_RISK_FACTOR + fee  # + calc_bias(e['last_failure'])*1e6
                                                                    # we don't consider failure heuristic at this point
    elif proto_type == 'ECL':
        n_capacity = 1 - (normalize(e['capacity_msat'], MIN_CAP, MAX_CAP))
        n_age = normalize(e['age'], MIN_AGE, MAX_AGE)
        n_delay = normalize(e['delay'], MIN_DELAY, MAX_DELAY)
        cost = fee * (n_delay * DELAY_RATIO + n_capacity * CAPACITY_RATIO + n_age * AGE_RATIO) 
        
    elif proto_type == 'CLN':
        fee = fee * (1 + DEFAULT_FUZZ * FUZZ)
        cost = (amount + fee) * e['delay'] * C_RISK_FACTOR + RISK_BIAS

    else:
        cost = 1
    cost = 1e-6 if cost < 0 else cost
    return cost


def get_shortest_paths(G, u, v, amount, proto_type='LND', max_count=5, timeout=1):

    def weight_function(u, v, e):
        return cost_function(e, amount, proto_type=proto_type)
    
    return take_until_timeout(nx.all_shortest_paths(G, u, v, weight=weight_function), 
                              max_count=max_count, timeout=timeout)


def random_amount(): # SAT
        
        # Возвращает массив значений от 10^0 = 1 до 10^7, равномерно распределенных на логарифмической шкале
        # https://coingate.com/blog/post/lightning-network-bitcoin-stats-progress
        # The highest transaction processed is 0.03967739 BTC, while the lowest is 0.000001 BTC. The average payment size is 0.00508484 BTC;
        # highest: 3967739.0 SAT
        # average: 508484.0 SAT
        # lowest: 100.0 SAT
        return LOG_SPACE[random.randrange(0, 10**6)] + 100


def gen_txset(G, transacitons_count=10000, seed=1313):
        
    def shortest_path_len(u, v):
        path_len = 0
        try:
              path_len = nx.shortest_path_length(G, u, v)
        except:
              pass
        return path_len
    
    random.seed(seed)
    np.random.seed(seed)

    tx_set = []
    nodes = list(G.nodes)
    max_path_length = 0
    for _ in tqdm(range(1, transacitons_count + 1), leave=False):
            while True:
              u = nodes[random.randrange(0, len(nodes))]
              v = nodes[random.randrange(0, len(nodes))]
              p = shortest_path_len(u, v)
              max_path_length = max(max_path_length, p)
              if v != u and p >= 2 and (u, v) not in tx_set:
                break
            tx_set.append((u, v))
    tx_set = [(tx[0], tx[1], random_amount() * 1000) for tx in tx_set]
    return tx_set


country_landmarks = {}
continent_landmarks = {}
#degree_centrality = {}

def get_landmarks(G, u, v, type='country', limit=1000):
    '''
    continent = G.nodes[u]['geojson']['continent']
    country = G.nodes[u]['geojson']['country']
    if type == 'continent':
        landmarks = [i for i in G.nodes #list(G.neighbors(u)) + list(G.neighbors(v))
                        if G.nodes[i]['geojson']['continent'] == continent]
    if type == 'country':
        landmarks = [i for i in G.nodes #list(G.neighbors(u)) + list(G.neighbors(v))
                        if G.nodes[i]['geojson']['country'] == country]
        
    landmarks = sorted(landmarks, key=lambda x: G.nodes[x]['carbon'] / G.degree(x), 
                            reverse=False)[:1000]
    
    #landmarks = sorted(landmarks, key=lambda x: G.degree[x], 
    #                        reverse=True)[:1000]
    
    landmarks = [i for i in landmarks if i in set(list(G.neighbors(u)) + list(G.neighbors(v)))]
    print(len(landmarks))
    if len(landmarks):
        return landmarks[0]#
        #return np.random.choice(landmarks, 1)[0] 
    
    #global degree_centrality
    #if not len(degree_centrality):
    #    degree_centrality = nx.degree_centrality(G)
    #    degree_centrality = dict(sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True))
    #    #print(degree_centrality)
    '''
    if type == 'continent':
        global continent_landmarks
        continent = G.nodes[u]['geojson']['continent']
        if continent not in continent_landmarks:
            landmarks = [i for i in G.nodes if G.nodes[i]['geojson']['continent'] == continent]
            landmarks = sorted(landmarks, key=lambda x: G.nodes[x]['carbon'] / G.degree(x), reverse=False)#[:limit]
            continent_landmarks.update({continent : landmarks})
        #landmarks = [i for i in continent_landmarks[continent] 
        #                    if i in G.neighbors(u) or i in G.neighbors(v)] # / G.degree(x)
        landmarks = continent_landmarks[continent][:limit]

    if type == 'country':
        global country_landmarks
        country = G.nodes[u]['geojson']['country']
        if country not in country_landmarks:
            landmarks = [i for i in G.nodes if G.nodes[i]['geojson']['country'] == country]
            landmarks = sorted(landmarks, key=lambda x: G.nodes[x]['carbon'] / G.degree(x), reverse=False)#[:limit]
            country_landmarks.update({country : landmarks})
        #landmarks = [i for i in country_landmarks[country] 
        #                    if i in G.neighbors(u) or i in G.neighbors(v)]   
        landmarks = country_landmarks[country][:limit]
    
    if len(landmarks):
        #print(len(landmarks))
        landmarks = landmarks + [G.neighbors(i) for i in landmarks]
        return set(landmarks)#[0]#
        #return np.random.choice(landmarks[:limit], 1)[0] 
    

def perform_payment(G, u, v, amount, proto_type='LND', max_count=5, timeout=1):
        
    if proto_type[0] == 'g':
        proto_type=proto_type[1:]
        if v not in G.neighbors(u):
            
            u_country = G.nodes[u]['geojson']['country']
            v_country = G.nodes[v]['geojson']['country']
            u_continent = G.nodes[u]['geojson']['continent']
            v_continent = G.nodes[v]['geojson']['continent']
            lm = None
            if u_country == v_country:
                lm = get_landmarks(G, u, v, type='country')
            elif u_continent == v_continent:
                lm = get_landmarks(G, u, v, type='continent')
            if lm is not None:
                _paths = get_shortest_paths(G, u, lm, amount, 
                                        proto_type=proto_type,
                                        max_count=1000,#max_count, 
                                        timeout=timeout)
                if not set(itertools.chain.from_iterable(_paths)).isdisjoint(lm):
                    print('!!!!!!!!!!!!!!')
                paths = []
                for p in _paths:
                    if not set(p).isdisjoint(lm):
                        paths.append(p)
                        print(p)
                if not len(paths):
                    paths = _paths
            else:
                paths = get_shortest_paths(G, u, v, amount, 
                                    proto_type=proto_type,
                                    max_count=max_count, 
                                    timeout=timeout)



            '''
            if lm is not None:
                #paths = [[u, i, v] for i in lm]
                
                lm_paths = get_shortest_paths(G, u, lm, amount, 
                                        proto_type=proto_type,
                                        max_count=max_count, 
                                        timeout=timeout)
                def_paths = get_shortest_paths(G, lm, v, amount, 
                                        proto_type=proto_type, #None,#
                                        max_count=max_count, 
                                        timeout=timeout)
                paths = []
                for p1, p2 in zip(lm_paths, def_paths):
                    paths.append(p1 + p2[1:])  
                
            else:
                paths = get_shortest_paths(G, u, v, amount, 
                                        proto_type=proto_type,
                                        max_count=max_count, 
                                        timeout=timeout)
            '''
        else:
            paths = get_shortest_paths(G, u, v, amount, 
                                    proto_type=proto_type,
                                    max_count=max_count, 
                                    timeout=timeout)   
    else:
        paths = get_shortest_paths(G, u, v, amount, 
                                proto_type=proto_type,
                                max_count=max_count, 
                                timeout=timeout)
    metrics = {
        'path_attempts': 0,
        'payment_hops': 0,
        'same_country': 0,
        'same_continent': 0,
        'cross_continent': 0,
        'carbon': 0,
    }
    for p in paths:
        metrics['path_attempts'] += 1
        a = amount
        delay = 0
        payment_succeed = True
        for u, v in itertools.pairwise(p):
            metrics['payment_hops'] += 1
            e = G.edges[u, v]
            payment_succeed = payment_succeed and\
                                    np.random.choice([True, False], 1,
                                               p=[1-e['fail_prob'], 
                                                    e['fail_prob']])[0]
            fee = e['fee_base_msat'] + amount * e['fee_rate']
            payment_succeed = payment_succeed and\
                                    amount + fee <= float(e['capacity_msat'])
            if payment_succeed:
                metrics[e['type']] += 1
                metrics['carbon'] += G.nodes[u]['carbon']
                delay += e['delay']
                a += fee
            else:
                break
        if len(p):
            metrics['payment_succeed'] = payment_succeed
            metrics['carbon'] += G.nodes[p[-1]]['carbon']
            metrics['amount'] = amount
            metrics['paths'] = paths
            if payment_succeed:
                metrics['amount_payed'] = a
                metrics['delay'] = delay
                break
        else:
            metrics['payment_succeed'] = False
            break

    return metrics