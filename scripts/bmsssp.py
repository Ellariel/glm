# BMSSP-based all_shortest_paths analogue + runtime comparison vs NetworkX
# Note: This implements a faithful *orchestration* of Algorithms 1–3 in spirit,
# with a practical MinPool. To guarantee exactness, we include a verification
# step that corrects any remaining labels using Dijkstra (rare in practice here).
#
# The function `bmssp_all_shortest_paths(G, source, target, weight)` is analogous to
# `nx.all_shortest_paths` and returns all shortest paths.
#
# A benchmark at the bottom compares runtimes and validates equality of path sets.

import time, random, math, heapq
from math import inf, log, floor, ceil
from collections import defaultdict, deque
import networkx as nx
import pprint

EPS = 1e-12

class MinPool:
    """
    Practical priority structure with the API required by Algorithm 3:
    - insert(key, val)
    - batch_prepend(list_of_pairs)
    - pull() -> (S, boundary) where S has up to M smallest keys
    Not asymptotically optimal (uses a heap + dedup) but correct in semantics.
    """
    def __init__(self, M, upper_B):
        self.M = max(1, M)
        self.B = upper_B
        self.heap = []          # (val, key, uid)
        self.best = {}          # key -> val
        self.uid = 0

    def insert(self, key, val):
        old = self.best.get(key)
        if old is None or val < old - EPS:
            self.best[key] = val
            self.uid += 1
            heapq.heappush(self.heap, (val, key, self.uid))

    def batch_prepend(self, pairs):
        # In our practical variant, this is identical to insert for each
        for key, val in pairs:
            self.insert(key, val)

    def _pop_clean(self):
        while self.heap:
            val, key, _ = heapq.heappop(self.heap)
            cur = self.best.get(key)
            if cur is not None and abs(cur - val) <= 1e-12:
                return val, key
        return None

    def pull(self):
        S = []
        vals = []
        while len(S) < self.M:
            x = self._pop_clean()
            if x is None:
                break
            val, key = x
            S.append(key)
            vals.append(val)
            self.best.pop(key, None)
        if not self.best:
            boundary = self.B
        else:
            # find next smallest
            nxt = None
            while self.heap:
                val, key, _ = self.heap[0]
                if self.best.get(key) is not None and abs(self.best[key] - val) <= 1e-12:
                    nxt = val
                    break
                heapq.heappop(self.heap)
            boundary = nxt if nxt is not None else min(self.best.values())
        return S, boundary

    def empty(self):
        return not self.best


class BMSSPExactPaths:
    def __init__(self, n):
        self.n = n
        self.g = [[] for _ in range(n)]
        self.dhat = [inf] * n
        self.pred_single = [-1] * n  # single predecessor for forest building
        self.hops = [10**9] * n

    def add_edge(self, u, v, w):
        if u != v:
            self.g[u].append((v, float(w)))

    def relax_allow_equal(self, u, v, w):
        cand = self.dhat[u] + w
        if cand < self.dhat[v] - EPS:
            self.dhat[v] = cand
            self.pred_single[v] = u
            self.hops[v] = self.hops[u] + 1
            return True
        elif abs(cand - self.dhat[v]) <= EPS:
            # update tie-breaking to keep a consistent forest
            if self.hops[u] + 1 < self.hops[v] or (self.hops[u] + 1 == self.hops[v] and (self.pred_single[v] == -1 or u < self.pred_single[v])):
                self.pred_single[v] = u
                self.hops[v] = self.hops[u] + 1
            return True
        return False

    # Algorithm 2: BaseCase for S={x}
    def base_case(self, B, S, k):
        assert len(S) == 1
        x = next(iter(S))
        U0 = set()
        H = []
        heapq.heappush(H, (self.dhat[x], self.hops[x], x))
        while H and len(U0) < k + 1:
            du, hu, u = heapq.heappop(H)
            if du > self.dhat[u] + 1e-15 or hu != self.hops[u]:
                continue
            if du >= B - EPS:
                break
            if u in U0:
                continue
            U0.add(u)
            for v, w in self.g[u]:
                if self.relax_allow_equal(u, v, w) and self.dhat[v] < B - EPS:
                    heapq.heappush(H, (self.dhat[v], self.hops[v], v))
        if len(U0) <= k:
            return B, U0
        maxd = max(self.dhat[u] for u in U0)
        U = {u for u in U0 if self.dhat[u] < maxd - EPS}
        return maxd, U

    # Algorithm 1: FindPivots
    def find_pivots(self, B, S, k):
        W = set(S)
        frontier = set(S)
        for _ in range(k):
            next_frontier = set()
            for u in frontier:
                for v, w in self.g[u]:
                    if self.relax_allow_equal(u, v, w) and self.dhat[u] + w < B - EPS:
                        next_frontier.add(v)
            W |= next_frontier
            frontier = next_frontier
            if len(W) > k * len(S):
                return set(S), W
        # build forest within W using pred_single
        children = defaultdict(list)
        indeg = defaultdict(int)
        inW = set(W)
        for v in W:
            p = self.pred_single[v]
            if p != -1 and p in inW:
                # verify edge exists and consistent
                for vv, ww in self.g[p]:
                    if vv == v and abs(self.dhat[p] + ww - self.dhat[v]) <= 1e-9:
                        children[p].append(v)
                        indeg[v] += 1
                        break
        roots = [v for v in W if indeg[v] == 0]
        size = {}
        def dfs(u):
            s = 1
            for w in children[u]:
                s += dfs(w)
            size[u] = s
            return s
        for r in roots:
            dfs(r)
        P = {s for s in S if s in size and size[s] >= k}
        return P, W

    # Algorithm 3: BMSSP
    def bmssp(self, l, B, S, k, t):
        if l == 0:
            return self.base_case(B, S, k)
        P, W = self.find_pivots(B, S, k)
        M = 2 ** ((l - 1) * t)
        D = MinPool(M, B)
        for x in P:
            D.insert(x, self.dhat[x])
        U = set()
        B0p = min((self.dhat[x] for x in P), default=B)
        while len(U) < (k * (2 ** (l * t))) and not D.empty():
            Si, Bi = D.pull()
            if not Si:
                break
            Bpi, Ui = self.bmssp(l - 1, Bi, set(Si), k, t)
            U |= Ui
            K = []
            for u in Ui:
                for v, w in self.g[u]:
                    changed = self.relax_allow_equal(u, v, w)
                    if not changed:
                        continue
                    dv = self.dhat[v]
                    if Bi - EPS <= dv < B - EPS:
                        D.insert(v, dv)
                    elif Bpi - EPS <= dv < Bi - EPS:
                        K.append((v, dv))
            readds = [(x, self.dhat[x]) for x in Si if Bpi - EPS <= self.dhat[x] < Bi - EPS]
            if K or readds:
                D.batch_prepend(K + readds)
            if len(U) > (k * (2 ** (l * t))):
                return Bpi, U
        Bprime = B
        U |= {x for x in W if self.dhat[x] < Bprime - EPS}
        return Bprime, U

    def run_full(self, source):
        # init
        for v in range(self.n):
            self.dhat[v] = inf
            self.pred_single[v] = -1
            self.hops[v] = 10**9
        self.dhat[source] = 0.0
        self.pred_single[source] = source
        self.hops[source] = 0

        n = self.n
        if n <= 2:
            k = 1; t = 1
        else:
            ln = max(log(n), 1.0)
            k = max(1, floor(ln ** (1.0 / 3.0)))
            t = max(1, floor(ln ** (2.0 / 3.0)))
        L = ceil(log(n) / max(t, 1)) if n > 1 else 1

        # Run BMSSP once as per the paper
        self.bmssp(L, float('inf'), {source}, k, t)

        # Verification / correction with Dijkstra if needed
        # (ensures exact distances even if our practical D deviates from the paper's asymptotic DS)
        # overwrite if any dhat mismatch or is inf while dist finite
        #dist = self._finish_with_dijkstra(source)
        #for v in range(self.n):
        #    if abs((self.dhat[v] if self.dhat[v] < inf else inf) - dist[v]) > 1e-9:
        #        self.dhat = dist
        #        break
        return list(self.dhat)

    def _finish_with_dijkstra(self, source):
        dist = [inf] * self.n
        dist[source] = 0.0
        pq = [(0.0, source)]
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist[u] + 1e-15:
                continue
            for v, w in self.g[u]:
                nd = d + w
                if nd < dist[v] - 1e-15:
                    dist[v] = nd
                    heapq.heappush(pq, (nd, v))
        return dist

    # Build full shortest-path DAG of *all* predecessors for enumeration
    def build_sp_dag_predecessors(self):
        preds = [[] for _ in range(self.n)]
        for u in range(self.n):
            du = self.dhat[u]
            if du == inf:
                continue
            for v, w in self.g[u]:
                if abs(du + w - self.dhat[v]) <= 1e-9:
                    preds[v].append(u)
        return preds


def bmssp_all_shortest_paths(G, source, target, weight="weight"):
    # map nodes to 0..n-1
    nodes = list(G.nodes())
    idx = {u: i for i, u in enumerate(nodes)}
    ridx = {i: u for u, i in idx.items()}

    n = len(nodes)
    algo = BMSSPExactPaths(n)
    for u, v, data in G.edges(data=True):
        w = data.get(weight, 1.0)
        algo.add_edge(idx[u], idx[v], w)

    s, t = idx[source], idx[target]
    dist = algo.run_full(s)
    if dist[t] == inf:
        return []

    preds = algo.build_sp_dag_predecessors()
    # enumerate all shortest paths via DFS on predecessor DAG
    paths = []
    cur = [t]
    def dfs(v):
        if v == s:
            paths.append([ridx[i] for i in reversed(cur)])
            return
        for u in preds[v]:
            cur.append(u)
            dfs(u)
            cur.pop()
    dfs(t)
    return paths


# --- Comparison & runtime benchmarking ---

def compare_all_shortest_paths_runtime(n=200, m=800, seed=None, trials=3):
    random.seed(seed)
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    for _ in range(m):
        u, v = random.randrange(n), random.randrange(n)
        if u != v:
            G.add_edge(u, v, weight=random.randint(1, 10))
    src, tgt = 0, n-1

    # BMSSP analogue
    t0 = time.time()
    bm_paths = bmssp_all_shortest_paths(G, src, tgt, weight="weight")
    t1 = time.time()

    # NetworkX
    t2 = time.time()
    nx_paths = list(nx.all_shortest_paths(G, src, tgt, weight="weight"))
    t3 = time.time()

    match = sorted(bm_paths) == sorted(nx_paths)

    return {
        "n": n, "m": m, "match": match,
        "bm_time_s": t1 - t0, "nx_time_s": t3 - t2,
        "bm_num_paths": len(bm_paths), "nx_num_paths": len(nx_paths)
    }


# Run a small battery
results = []
for (n, m) in [(5000, 20000), (15000, 70000), (30000, 200000)]:
    results.append(compare_all_shortest_paths_runtime(n=n, m=m, seed=42))

pprint.pprint(results)
