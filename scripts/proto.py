import os, sys
import random
import numpy as np
from tqdm import tqdm
import networkx as nx
import itertools
import requests
import asyncio



def take_until_timeout(iterator, max_count=5, per_item_timeout=3):

    async def get_elements():
        elements = []
        for _ in range(max_count):
            try:
                path = await asyncio.wait_for(
                    asyncio.to_thread(next, iterator),
                    timeout=per_item_timeout
                )
            except:
                break
            #except asyncio.TimeoutError:
            #    break
            #except StopIteration:
            #    break
            elements.append(path)
        return elements
    
    return asyncio.get_event_loop().run_until_complete(get_elements())


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
    fee = e['fee_base_msat'] + amount * e['fee_rate_msat']
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
    cost = 0 if cost < 0 else cost
    return cost


def get_shortest_paths(G, u, v, amount, proto_type='LND', max_count=5, timeout=5):

    def weight_function(u, v, e):
        return cost_function(e, amount, proto_type=proto_type)
    
    return take_until_timeout(nx.all_shortest_paths(G, u, v, weight=weight_function), 
                              max_count=max_count, per_item_timeout=timeout)


def random_amount(): # SAT
        
        # Возвращает массив значений от 10^0 = 1 до 10^7, равномерно распределенных на логарифмической шкале
        # https://coingate.com/blog/post/lightning-network-bitcoin-stats-progress
        # The highest transaction processed is 0.03967739 BTC, while the lowest is 0.000001 BTC. The average payment size is 0.00508484 BTC;
        # highest: 3967739.0 SAT
        # average: 508484.0 SAT
        # lowest: 100.0 SAT
        return LOG_SPACE[random.randrange(0, 10**6)]


def gen_txset(G, transacitons_count=10000, use_first_component=True, seed=1313):
        
    def shortest_path_len(u, v):
        path_len = 0
        try:
              path_len = nx.shortest_path_length(G, u, v)
        except:
              pass
        return path_len
    
    random.seed(seed)
    np.random.seed(seed)

    if use_first_component:
        components = [G.subgraph(c) 
                        for c in sorted(nx.connected_components(G), 
                            key=len, reverse=True)]
        G = components[0]

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
    tx_set = [(tx[0], tx[1], random_amount() + 100) for tx in tx_set]
    return tx_set


def get_landmark(G, n, type='country'):
    country = G.nodes[n]['geojson']['country']
    continent = G.nodes[n]['geojson']['continent']
    neighbors = [i for i in G.neighbors(n)]
    if type == 'continent':
        continent_neighbors = [i for i in neighbors if G.nodes[i]['geojson']['continent'] == continent]
        if len(continent_neighbors):
            continent_neighbors = sorted(continent_neighbors, 
                                    key=lambda x: G.degree[x],
                                    reverse=True)
            #print(continent_neighbors)
            return continent_neighbors[0]
    country_neighbors = [i for i in neighbors if G.nodes[i]['geojson']['country'] == country]
    if len(country_neighbors):
        country_neighbors = sorted(country_neighbors, 
                                   key=lambda x: G.degree[x],
                                   reverse=True)
        #print(country_neighbors)
        return country_neighbors[0]
    neighbors = sorted(neighbors, 
                        key=lambda x: G.degree[x],
                        reverse=True)
    return neighbors[0]


def perform_payment(G, u, v, amount, proto_type='LND', max_count=5, timeout=3):
        
    if proto_type[0] == 'g':
        proto_type=proto_type[1:]
        u_country = G.nodes[u]['geojson']['country']
        v_country = G.nodes[v]['geojson']['country']
        u_continent = G.nodes[u]['geojson']['continent']
        v_continent = G.nodes[v]['geojson']['continent']
        lm = None
        if u_country == v_country:
            lm = get_landmark(G, u, type='country')
        elif u_continent == v_continent:
            lm = get_landmark(G, u, type='continent')
        if lm is not None:
            lm_paths = get_shortest_paths(G, u, lm, amount, 
                                    proto_type=proto_type,
                                    max_count=max_count, 
                                    timeout=timeout)
            def_paths = get_shortest_paths(G, lm, v, amount, 
                                    proto_type=proto_type,
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
        payment_succeed = True
        for u, v in itertools.pairwise(p):
            metrics['payment_hops'] += 1
            e = G.edges[u, v]
            payment_succeed = payment_succeed and\
                                    np.random.choice([True, False], 1,
                                               p=[1-e['fail_prob'], 
                                                    e['fail_prob']])[0]
            fee = e['fee_base_msat'] + amount * e['fee_rate_msat']
            payment_succeed = payment_succeed and\
                                    amount + fee <= float(e['capacity_msat'])
            if payment_succeed:
                metrics[e['type']] += 1
                metrics['carbon'] += G.nodes[u]['carbon']
                a += fee
            else:
                break

        metrics['payment_succeed'] = payment_succeed
        metrics['carbon'] += G.nodes[p[-1]]['carbon']
        metrics['amount'] = amount
        metrics['paths'] = paths
        if payment_succeed:
            metrics['amount_payed'] = a
            break

    return metrics