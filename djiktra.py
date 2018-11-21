# -*- coding: utf-8 -*-
import numpy as np


def get_dijkstra_matrix(graph):
    adjacent_matrix = graph['adjacent_matrix']
    adjacent_table = graph['adjacent_table']
    dijkstra_matrix = np.full_like(adjacent_matrix, -1)
    route_matrix = {}
    d = dijkstra_matrix.shape[0]
    for i in range(1, d):
        # d, route_matrix[i] = dijkstra_algor(i,adjacent_table)
        d = dijkstra_algor(i, adjacent_table)
        for k in d.keys(): dijkstra_matrix[i][k] = d[k]
    graph['dijkstra_matrix'] = dijkstra_matrix
    # graph['route_matrix'] = route_matrix


def dijkstra_algor(src, adjacent_table):
    # initialize
    last_visited = src
    node_list = adjacent_table.keys()
    unvisited = set(node_list) - {last_visited}
    d = {node: float('inf') for node in node_list}     # store min distance
    # r = {node: list() for node in node_list}  # store min route
    for node in adjacent_table[src].keys(): d[node] = adjacent_table[src][node]
    d[src] = 0


    # find next one
    while len(unvisited) > 0:
        # update distance
        min_value, min_node = float('inf'), None
        for c in unvisited:
            if c in adjacent_table[last_visited].keys():
                if d[c] >= d[last_visited]+adjacent_table[last_visited][c]:
                    d[c] = d[last_visited]+adjacent_table[last_visited][c]
                    # r[c] = r[last_visited]+[(last_visited, c)]
            if d[c] <= min_value:
                min_value, min_node = d[c], c
        unvisited.remove(min_node)
        last_visited = min_node

    return d





