from config import Configuration
import numpy as np
import random
import operator


def file_to_graph(filename):
    """
    :param filename
    :returns:
        cost_matrix, demand_matrix, basic information
    :type: -1 means no edge between vertices
    """
    basic_infor = get_basic_infor(filename)
    vertices_num = Configuration.VERTICES

    cost_matrix = np.full((vertices_num+1, vertices_num+1), -1)
    demand_matrix = np.full((vertices_num+1, vertices_num+1), -1)
    adjacent_table = {}
    with open(filename, 'r') as f:
        count = 0
        for line in f.readlines():
            count += 1
            if count >= 10:
                s = line.strip().split()
                if s[0] != 'END':
                    i, j, cost, demand = int(s[0]), int(s[1]), int(s[2]), int(s[3])
                    cost_matrix[i, j], demand_matrix[i, j] = cost, demand
                    cost_matrix[j, i], demand_matrix[j, i] = cost, demand
                    if i in adjacent_table.keys():
                        adjacent_table[i][j] = cost
                    else: adjacent_table[i] = {j: cost}
                    if j in adjacent_table.keys():
                        adjacent_table[j][i] = cost
                    else: adjacent_table[j] = {i: cost}

    return {"adjacent_matrix": cost_matrix, "adjacent_table": adjacent_table, "demand_matrix":demand_matrix,"basic_infor":basic_infor}


def get_basic_infor(filename):

    with open(filename, 'r') as f:
        i = 0
        for line in f.readlines():
            i = i+1
            if i == 1:
                Configuration.NAME = line.strip().split(':')[1].strip()
            if i == 2:
                Configuration.VERTICES = int(line.strip().split(':')[1].strip())
            if i == 3:
                Configuration.DEPOT = int(line.strip().split(':')[1].strip())
            if i == 4:
                Configuration.REQUIRED_EDGES = int(line.strip().split(':')[1].strip())
            if i == 5:
                Configuration.NON_REQUIRED_EDGES = int(line.strip().split(':')[1].strip())
            if i == 6:
                Configuration.VEHICLES = int(line.strip().split(':')[1].strip())
            if i == 7:
                Configuration.CAPACITY = int(line.strip().split(':')[1].strip())
            if i == 8:
                Configuration.TOTAL_COST_OF_REQUIRED_EDGES = int(line.strip().split(':')[1].strip())
            if i == 9:
                break
    return {'CAPACITY':Configuration.CAPACITY, 'REQUIRED_EDGES':Configuration.REQUIRED_EDGES,'DEPOT':Configuration.DEPOT}


def print_result(S, graph, paint=False):

    dijkstra_matrix = graph['dijkstra_matrix']
    adjacent_matrix = graph['adjacent_matrix']
    depot = graph['basic_infor']['DEPOT']
    cost = 0
    routes = []
    for route in S:
        if len(route) == 0:
            continue
        start = Configuration.DEPOT
        routes.append(0)
        routes += route
        for task in route:
            cost += dijkstra_matrix[start][task[0]]
            cost += adjacent_matrix[task[0]][task[1]]
            start = task[1]
        cost += dijkstra_matrix[start][depot]
        routes.append(0)
    r = ','.join((str(item).replace(" ","") for item in routes))
    if paint:
        print("The final best feasible solution is: ")
        print("s",r)
        print("q",cost)
    return cost


def get_cost(S, graph):
    dijkstra_matrix = graph['dijkstra_matrix']
    adjacent_matrix = graph['adjacent_matrix']
    depot = graph['basic_infor']['DEPOT']
    cost = 0
    for route in S:
        start = depot
        for task in route:
            cost += dijkstra_matrix[start][task[0]]
            cost += adjacent_matrix[task[0]][task[1]]
            start = task[1]
        cost += dijkstra_matrix[start][depot]
    return cost


def get_violation(S,graph):
    demand_matrix = graph['demand_matrix']
    total_violate = 0
    capacity = graph['basic_infor']['CAPACITY']
    for route in S:
        load = 0
        for task in route:
            load += demand_matrix[task[0]][task[1]]
        route_violate = max(0, load - capacity)
        total_violate += route_violate
    return total_violate


def evaluation_solution(s, graph, lmd):
    """
    evaluation the score of a possible solution
    :param s: a solution
    :param graph:
    :param lmd:
    :return:
    """
    dijkstra_matrix = graph['dijkstra_matrix']
    adjacent_matrix = graph['adjacent_matrix']
    demand_matrix = graph['demand_matrix']
    capacity = graph['basic_infor']['CAPACITY']
    depot = graph['basic_infor']['DEPOT']
    total_violate = 0
    cost = 0
    for route in s:
        if len(route) == 0:
            continue
        start = depot
        route_load = 0
        for task in route:
            route_load += demand_matrix[task[0]][task[1]]
            cost += dijkstra_matrix[start][task[0]]
            cost += adjacent_matrix[task[0]][task[1]]
            start = task[1]
        route_violate = max(0, route_load - capacity)
        cost += dijkstra_matrix[start][depot]
        total_violate += route_violate
    return cost+(lmd*total_violate)



def generate_two_random(start, end):
    a = random.randint(start, end)
    b = random.randint(start, end)
    while a == b:
        b = random.randint(start, end)
    return a, b


def generate_one_random(start, end, flt=False):
    if flt:
        return random.random()
    else:
        return random.randint(start, end)


def is_clone(s, S):
    for p in S:
        if operator.eq(p, s):
            return True
    return False


def split_route(r):
    i = generate_one_random(0,len(r)-1)
    r_l, r_r = r[:i], r[i:]
    return r_l, r_r


def get_initial_lmd(TC_best, S, graph):
    """
    Because current don't know what is Q in paper, so by personal judging,
    set it to the Capacity of vehicle.
    :param TC_best: the best feasible solution so far
    :param S: the solution used to local search
    :param graph: ...
    :return: non return , just update the lambda
    """
    capacity = graph['basic_infor']['CAPACITY']
    TV = get_violation(S, graph)
    TC = get_cost(S, graph)
    lmd = (TC_best/capacity)*((TC_best/TC)+(TV/capacity)+1)
    return lmd


def is_feasible(s, graph):
    required_edge = graph['basic_infor']['REQUIRED_EDGES']
    finished_edge = 0
    for route in s:
        finished_edge += len(route)
    return True if finished_edge == required_edge else False


