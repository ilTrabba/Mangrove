"""
MoTHer Algorithm Main Implementation
Replaces the find_parent_stub function with sophisticated heritage detection
"""

import logging
logger = logging.getLogger(__name__)

import networkx as nx
from typing import Dict, Tuple, Set

class MDST:
    
    import numpy as np
    import copy

    def chu_liu_edmonds(self, graph, root=None):
        """
        Versione robusta dell'algoritmo di Chu-Liu-Edmonds.
        
        Args:
            graph: Dizionario {u: {v: {'weight': w}, ...}}
            root: Il nodo radice (opzionale). Se None, viene trovata la radice ottima.
            
        Returns:
            (arborescenza, radice_usata)
        """
        # 1. Copia profonda per non modificare i dati originali dell'utente
        #graph = copy.deepcopy(graph)
        nodes = list(graph.keys())
        
        # 2. GESTIONE RADICE AUTOMATICA (Virtual Root Strategy)
        original_root = root
        virtual_root = None
        
        if root is None:
            # Trova un ID univoco per il nodo virtuale
            virtual_root = -1
            while virtual_root in nodes:
                virtual_root -= 1
                
            # Calcola un peso penalizzante (somma di tutti i pesi esistenti + 1)
            # Questo forza l'algoritmo a scegliere un solo arco uscente dalla radice virtuale
            all_weights = [d['weight'] for u in graph for v, d in graph[u].items()]
            penalty_weight = sum(all_weights) + 1.0 if all_weights else 100.0
            
            # Aggiungi il nodo virtuale connesso a TUTTI gli altri nodi
            graph[virtual_root] = {}
            for n in nodes:
                graph[virtual_root][n] = {'weight': penalty_weight}
                
            root = virtual_root
            nodes.append(virtual_root)

        # --- FASE 1: Selezione Greedy ---
        min_in_edges = {}
        
        for v in nodes:
            if v == root:
                continue
                
            best_edge = None
            min_weight = float('inf')
            
            # Cerca archi entranti in v
            for u in graph:
                if v in graph[u]:
                    w = graph[u][v]['weight']
                    if w < min_weight:
                        min_weight = w
                        best_edge = (u, v)
            
            if best_edge:
                min_in_edges[v] = (best_edge, min_weight)
            else:
                # EDGE CASE: Nodo irraggiungibile
                # Se siamo nella ricorsione (grafo contratto), è un problema serio.
                # Se siamo al top level e usiamo la virtual root, potrebbe essere normale per componenti disconnesse,
                # ma per un'arborescenza valida (spanning) deve essere tutto connesso.
                pass 

        # Controllo raggiungibilità (Spanning Property)
        if len(min_in_edges) < len(nodes) - 1:
            reachable = set(min_in_edges.keys()) | {root}
            unreachable = set(nodes) - reachable
            raise ValueError(f"Impossibile formare un'arborescenza completa. Nodi irraggiungibili dalla radice {root}: {unreachable}")

        # --- FASE 2: Rilevamento Cicli ---
        cycle = self._find_cycle(min_in_edges, nodes)
        
        # CASO BASE: Nessun ciclo
        if not cycle:
            arborescence = {u: {} for u in nodes}
            for v, (edge, w) in min_in_edges.items():
                u = edge[0]
                arborescence[u][v] = {'weight': w}
                
            # Se abbiamo usato una radice virtuale, dobbiamo ripulire il risultato
            if virtual_root is not None:
                final_arb = {u: {} for u in nodes if u != virtual_root}
                found_real_root = None
                
                # Cerca quale nodo reale è stato puntato dalla radice virtuale
                for v in arborescence[virtual_root]:
                    found_real_root = v
                    # Non aggiungiamo questo arco (virtual->real) nel risultato finale, 
                    # perché l'arborescenza reale inizia da 'v'
                
                if found_real_root is None:
                    raise ValueError("Errore interno: La radice virtuale non ha selezionato nessun figlio.")

                # Copia gli altri archi
                for u in arborescence:
                    if u == virtual_root: continue
                    for v in arborescence[u]:
                        final_arb[u][v] = arborescence[u][v]
                
                return final_arb, found_real_root
            
            return arborescence, root

        # CASO RICORSIVO: Contrazione Ciclo
        # (Logica identica alla precedente, ma adattata per gestire la ricorsione correttamente)
        cycle_nodes = set(cycle)
        new_node = max(nodes) + 1 # Super-nodo
        while new_node in graph: new_node += 1 # Sicurezza collisioni

        contracted_graph = {}
        
        # Costruzione grafo contratto
        for u in graph:
            if u not in cycle_nodes:
                contracted_graph[u] = {}
        contracted_graph[new_node] = {}
        
        cycle_weight_sum = sum(min_in_edges[v][1] for v in cycle)
        
        for u in graph:
            for v in graph[u]:
                w = graph[u][v]['weight']
                if u not in cycle_nodes and v not in cycle_nodes:
                    contracted_graph[u][v] = {'weight': w}
                elif u not in cycle_nodes and v in cycle_nodes:
                    w_in_cycle = min_in_edges[v][1]
                    new_weight = w - w_in_cycle
                    if new_node not in contracted_graph[u] or new_weight < contracted_graph[u][new_node]['weight']:
                        contracted_graph[u][new_node] = {'weight': new_weight, 'real_target': v}
                elif u in cycle_nodes and v not in cycle_nodes:
                    if v not in contracted_graph[new_node] or w < contracted_graph[new_node][v]['weight']:
                        contracted_graph[new_node][v] = {'weight': w, 'real_source': u}

        # Chiamata ricorsiva
        # Nota: la radice passa invariata se non è nel ciclo, altrimenti diventa new_node
        next_root = root if root not in cycle_nodes else new_node
        mst_contracted, _ = self.chu_liu_edmonds(contracted_graph, next_root)
        
        # --- FASE 3: Espansione (Unpacking) ---
        final_edges = []
        # Archi del ciclo
        for v in cycle:
            u = min_in_edges[v][0][0]
            final_edges.append((u, v, min_in_edges[v][1]))
            
        key_to_remove = None
        
        for u in mst_contracted:
            for v in mst_contracted[u]:
                if v == new_node: # Entrante nel ciclo
                    real_target = contracted_graph[u][new_node]['real_target']
                    w_original = graph[u][real_target]['weight']
                    final_edges.append((u, real_target, w_original))
                    key_to_remove = real_target
                elif u == new_node: # Uscente dal ciclo
                    real_source = contracted_graph[new_node][v]['real_source']
                    w_original = graph[new_node][v]['weight']
                    final_edges.append((real_source, v, w_original))
                else:
                    final_edges.append((u, v, mst_contracted[u][v]['weight']))
                    
        result_graph = {n: {} for n in nodes}
        for u, v, w in final_edges:
            if v == key_to_remove and u in cycle_nodes:
                continue
            result_graph[u][v] = {'weight': w}
            
        # Gestione cleanup virtual root al ritorno dalla ricorsione
        if virtual_root is not None:
            # Se siamo qui, significa che c'era un ciclo nel grafo esteso col nodo virtuale.
            # Logica di pulizia simile al caso base.
            final_arb = {u: {} for u in nodes if u != virtual_root}
            found_real_root = None
            for v in result_graph[virtual_root]:
                found_real_root = v
            
            for u in result_graph:
                if u == virtual_root: continue
                for v in result_graph[u]:
                    final_arb[u][v] = result_graph[u][v]
            return final_arb, found_real_root

        return result_graph, root

    def _find_cycle(self, selection, nodes):
        """Helper identico per trovare cicli."""
        visited = set()
        path_set = set()
        parents = {v: edge[0] for v, (edge, w) in selection.items()}

        def visit(node):
            if node in path_set:
                cycle = []
                curr = node
                while True:
                    cycle.append(curr)
                    curr = parents[curr]
                    if curr == node: break
                return cycle
            if node in visited: return None
            visited.add(node)
            path_set.add(node)
            if node in parents:
                res = visit(parents[node])
                if res: return res
            path_set.remove(node)
            return None

        for n in nodes:
            if n not in visited:
                cycle = visit(n)
                if cycle: return cycle
        return None

    # Backward-compatible alias (the original call site used the non-underscored name)
    find_cycle = _find_cycle
