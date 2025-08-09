import os, sys
import random
import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm
import zipfile
import pickle

from proto import gen_txset, MIN_AGE, MAX_AGE, MIN_CAP, MAX_CAP

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

base_dir = os.path.dirname(__file__)
data_dir = os.path.abspath(os.path.join(base_dir, 
                                        "..", 
                                        "data"))
os.makedirs(data_dir, exist_ok=True)
print('data_dir:', data_dir)

continents = pd.read_csv(os.path.join(data_dir, 'country-and-continent.csv'), 
                         na_filter=False,
                         dtype=str)

intencity = pd.read_csv(os.path.join(data_dir, 'ourworldindata_2025.csv'),
                        sep=';', parse_dates=True)
intencity = intencity.rename(columns={intencity.columns[1]: 'carbon'})
intencity = intencity.groupby('country_code').last()['carbon'].reset_index()

intencity = pd.merge(intencity, continents, 
                     left_on='country_code',
                     right_on='Three_Letter_Country_Code')\
                        .set_index('Two_Letter_Country_Code').to_dict()['carbon']

intencity_avg = np.mean(list(intencity.values())) # 442.396

continents = continents[['Two_Letter_Country_Code', 'Continent_Code']]\
                .set_index('Two_Letter_Country_Code').to_dict()['Continent_Code']

g = nx.read_gml(zipfile.ZipFile(os.path.join(data_dir,
                                    '20230716.gml.geo.zip'), 'r')\
                                        .open('20230716.gml.geo'))

components = [g.subgraph(c) 
                for c in sorted(nx.connected_components(g), 
                    key=len, reverse=True)]
g = components[0]
print(g)

geoloc_def = [g.nodes[i]['geojson'] 
            for i in tqdm(g.nodes, leave=False) 
                if 'geojson' in g.nodes[i] and\
                   'country' in g.nodes[i]['geojson']]

for i in geoloc_def:
     i['continent'] = continents[i['country']]

countries_def = pd.Series([i['country'] for i in geoloc_def])
countries_def_freq = countries_def.value_counts() / len(countries_def)

continents_def = pd.Series([i['continent'] for i in geoloc_def])
continents_def_freq = continents_def.value_counts() / len(continents_def)

random.seed(1313)
np.random.seed(1313)

for n in tqdm(g.nodes, leave=False):
    if 'geojson' not in g.nodes[n] or\
       'country' not in g.nodes[n]['geojson']:
            idx = random.randint(0, len(geoloc_def) - 1)
            g.nodes[n]['geojson'] = geoloc_def[idx]
    else:
       g.nodes[n]['geojson']['continent'] = continents[g.nodes[n]['geojson']['country']]
    if g.nodes[n]['geojson']['country'] in intencity:
        g.nodes[n]['carbon'] = intencity[g.nodes[n]['geojson']['country']]
    else:
        g.nodes[n]['carbon'] = intencity_avg

geoloc_new = [g.nodes[i]['geojson'] 
            for i in tqdm(g.nodes, leave=False) 
                if 'geojson' in g.nodes[i] and\
                   'country' in g.nodes[i]['geojson']]

countries_new = pd.Series([i['country'] for i in geoloc_new])
countries_new_freq = countries_new.value_counts() / len(countries_new)

continents_new = pd.Series([i['continent'] for i in geoloc_new])
continents_new_freq = continents_new.value_counts() / len(continents_new)

countries = pd.concat([countries_def_freq.rename('def'),
                       countries_new_freq.rename('new')], axis=1)
countries = pd.concat([countries.iloc[:5],
                       pd.DataFrame(countries.iloc[5:].sum().rename('Other')).T], 
                       axis=0)
print((countries * 100))

continents = pd.concat([continents_def_freq.rename('def'),
                       continents_new_freq.rename('new')], axis=1)
print((continents * 100))

channels_new = {
    'same_country': 0,
    'same_continent': 0,
    'cross_continent': 0,
}

for u, v in tqdm(g.edges, leave=False):
    if g.nodes[u]['geojson']['continent'] == g.nodes[v]['geojson']['continent']:
        if g.nodes[u]['geojson']['country'] == g.nodes[v]['geojson']['country']:
            g.edges[u, v]['fail_prob'] = 0.001
            g.edges[u, v]['type'] = 'same_country'
            channels_new['same_country'] += 1
        else:
            g.edges[u, v]['fail_prob'] = 0.005
            g.edges[u, v]['type'] = 'same_continent'
            channels_new['same_continent'] += 1
    else:
        g.edges[u, v]['fail_prob'] = 0.007
        g.edges[u, v]['type'] = 'cross_continent'
        channels_new['cross_continent'] += 1

for u, v in tqdm(g.edges, leave=False):
    g.edges[u, v]['fee_base_msat'] = int(g.edges[u, v]['fee_base_msat'])
    g.edges[u, v]['fee_rate_msat'] = int(g.edges[u, v]['fee_proportional_millionths']) / 1000
    g.edges[u, v]['delay'] = int(g.edges[u, v]['cltv_expiry_delta'])
    g.edges[u, v]['htlc_minimim_msat'] = int(g.edges[u, v]['htlc_minimim_msat'])
    g.edges[u, v]['htlc_maximum_msat'] = int(g.edges[u, v]['htlc_maximum_msat'])
    g.edges[u, v]['age'] = int(random.randint(MIN_AGE, MAX_AGE))
    g.edges[u, v]['capacity_msat'] = random.randint(g.edges[u, v]['htlc_minimim_msat'], 
                                                    g.edges[u, v]['htlc_maximum_msat'])

for k, v in channels_new.items():
    print(k, 100*v/len(g.edges))

result_file = os.path.join(data_dir, 
                            'snapshot.gml.geo')

nx.write_gml(g, result_file)
if os.path.exists(result_file + '.zip'):
    os.remove(result_file + '.zip')
zf = zipfile.ZipFile(result_file + '.zip', 
                        "w", zipfile.ZIP_DEFLATED)
zf.write(result_file, 'snapshot.gml.geo')
zf.close()
os.remove(result_file)

txs = gen_txset(g)
with open(os.path.join(data_dir, 
                            'txs.pkl'), 'wb') as f:
    print(len(txs), 'transactions saved')
    pickle.dump(txs, f)



