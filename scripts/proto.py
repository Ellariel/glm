import random
import numpy as np
from tqdm import tqdm
import networkx as nx
import itertools
import requests



def normalize(x, min, max):
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


def cost_function(G, u, v, amount, proto_type='LND'):
    fee = G.edges[u, v]['fee_base_sat'] + amount * G.edges[u, v]['fee_rate_sat']
    if proto_type == 'LND':
        cost = (amount + fee) * G.edges[u, v]['delay'] * LND_RISK_FACTOR + fee # + calc_bias(G.edges[u, v]['last_failure'])*1e6
                                                                               # we don't consider failure heuristic at this point
    elif proto_type == 'ECL':
        n_capacity = 1 - (normalize(G.edges[u, v]['capacity_sat'], MIN_CAP, MAX_CAP))
        n_age = normalize(BLOCK_HEIGHT - G.edges[u, v]['age'], MIN_AGE, MAX_AGE)
        n_delay = normalize(G.edges[u, v]['delay'], MIN_DELAY, MAX_DELAY)
        cost = fee * (n_delay * DELAY_RATIO + n_capacity * CAPACITY_RATIO + n_age * AGE_RATIO) 
            
    elif proto_type == 'CLN':
        fee = fee * (1 + DEFAULT_FUZZ * FUZZ)
        cost = (amount + fee) * G.edges[u, v]['delay'] * C_RISK_FACTOR + RISK_BIAS

    else:
        cost = 1
    cost = 0 if cost < 0 else cost
    return cost


def get_shortest_paths(G, u, v, amount, proto_type='LND', max_count=5):

    def weight_function(u, v, d):
        return cost_function(G, u, v, amount, proto_type=proto_type)

    try:
        return list(itertools.islice(nx.all_shortest_paths(G, u, v, 
                                                      weight=weight_function), max_count))
    except:
        return []


def random_amount(): # SAT
        # Возвращает массив значений от 10^0 = 1 до 10^7, равномерно распределенных на логарифмической шкале
        # https://coingate.com/blog/post/lightning-network-bitcoin-stats-progress
        # The highest transaction processed is 0.03967739 BTC, while the lowest is 0.000001 BTC. The average payment size is 0.00508484 BTC;
        # highest: 3967739.0 SAT
        # average: 508484.0 SAT
        # lowest: 100.0 SAT
        return LOG_SPACE[random.randrange(0, 10**6)]
