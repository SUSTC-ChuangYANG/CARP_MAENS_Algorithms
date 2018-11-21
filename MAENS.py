import copy
import heapq
import time
from path_scanning import path_scanning
from config import Configuration
from multiprocessing import Pool
from utils import generate_one_random, generate_two_random,\
                  is_clone, evaluation_solution,get_cost, \
                  get_violation, print_result, get_initial_lmd,\
                  split_route, is_feasible


best_feasible_solution = {"cost": 0, "solution": []}
pop_t = []
Graph = None


def init_population(graph, tasks, print_initial=False):
    current_population = []
    while len(current_population) < 2000:
        ntrail = 0
        end = False
        while 1:
            individual = path_scanning(graph=graph, tasks=tasks)
            ntrail += 1
            if not is_clone(individual, current_population):
                break
            if ntrail == Configuration.ubtrial:  # generate a duplicated individual，count until get upper bound
                end = True
                break
        if end:
            break
        heapq.heappush(current_population, (evaluation_solution(individual, graph, lmd=0), individual))
    result = []
    # select p_size best individuals as initial population
    if print_initial: print("Finish initial population, only print top 3.")
    for i in range(Configuration.psize):
        item = heapq.heappop(current_population)
        if print_initial and i < 3: 
            print(item)
        result.append(item[1])
    return result


def crossover(s1, s2, graph):
    depot = graph['basic_infor']['DEPOT']
    dijkstra_matrix = graph['dijkstra_matrix']
    adjacent_matrix = graph['adjacent_matrix']
    # selected two route
    r1 = generate_one_random(0,len(s1)-1)
    r2 = generate_one_random(0,len(s2)-1)
    s_x = copy.deepcopy(s1)
    r_1, r_2 = s_x[r1].copy(), s2[r2].copy()
    r_11, r_12 = split_route(r_1)
    r_21, r_22 = split_route(r_2)
    # remove duplicated
    r_new = r_11.copy()
    for item in r_22:
        if item in set(r_11) or (item[1],item[0]) in set(r_11):
            continue
        else:
            r_new.append(item)
            if item not in set(r_12) or (item[1],item[0]) not in set(r_12):
                for route in s_x:
                    if item not in set(route) and (item[1],item[0]) not in set(route):
                        continue
                    elif item in set(route):
                        route.remove(item)
                        break
                    else:
                        route.remove((item[1], item[0]))
                        break
                    
    # recover the missed
    s = set(r_new)
    missed = [item for item in r_12 if item not in s and (item[1], item[0]) not in s]
    for task in missed:
        min_cost, position = float('inf'), 0
        for i in range(len(r_new)+1):
            addition_cost = adjacent_matrix[task[0]][task[1]]
            if i == 0:
                addition_cost += dijkstra_matrix[depot][task[0]]
                addition_cost += dijkstra_matrix[task[1]][r_new[i][0]]
            elif i == len(r_new):
                addition_cost += dijkstra_matrix[r_new[i-1][1]][task[0]]
                addition_cost += dijkstra_matrix[task[1]][depot]
            else:
                addition_cost += dijkstra_matrix[r_new[i - 1][1]][task[0]]
                addition_cost += dijkstra_matrix[task[1]][r_new[i][0]]
            if addition_cost <= min_cost:
                min_cost, position = addition_cost, i
        r_new.insert(position, task)
    s_x[r1] = r_new
    s_x = [item for item in s_x if item != []]
    return s_x


def local_search(s_x, graph, lmd):
    better_one_found = True
    s_ls = copy.deepcopy(s_x)
    feasible_cnt = 0
    infeasible_cnt = 0
    last_iteration_condition = "feasible"
    while better_one_found:
        s = best_tradition_operator(s_ls, graph, lmd)
        new_solution = ms_operators(s, graph, lmd)
        if new_solution is None:
            break
        else:
            s_ls = new_solution

            # update lambda
            if is_feasible(s_ls, graph):
                if last_iteration_condition == "feasible":
                    feasible_cnt += 1
                    if feasible_cnt == 5:
                        lmd = lmd / 2
                        feasible_cnt = 0
                else:
                    feasible_cnt = 1
                last_iteration_condition = "feasible"

            else:
                if last_iteration_condition == "infeasible":
                    infeasible_cnt += 1
                    if infeasible_cnt == 5:
                        lmd = lmd * 2
                        infeasible_cnt = 0
                else:
                    infeasible_cnt = 1
                last_iteration_condition = "infeasible"

    return s_ls, lmd


def best_tradition_operator(s_x, graph, lmd):
    s1, c1 = single_insertion(s_x, graph, lmd)
    s2, c2 = double_insertion(s_x, graph, lmd)
    s3, c3 = swap(s_x, graph, lmd)
    max_value = max(c1, c2, c3)
    if max_value == c1:
        return s1
    elif max_value == c2:
        return s2
    else:
        return s3


def single_insertion(s, graph, lmd):
    s_t = copy.deepcopy(s)
    min_cost = float('inf')
    pos = (0, 0, 0, 0)
    for i, route in enumerate(s_t):
        for j in range(len(route)):
            task = route[j]
            del route[j]
            for x, r in enumerate(s_t):
                for y in range(len(r)+1):
                    s_t[x].insert(y, task)
                    cost = evaluation_solution(s_t, graph, lmd)
                    if cost <= min_cost:
                        min_cost, pos = cost, (i, j, x, y)
                    del s_t[x][y]
            route.insert(j, task)
    i, j, x, y = pos
    task = s_t[i][j]
    del s_t[i][j]
    s_t[x].insert(y, task)
    if len(s_t[i]) == 0:
        del s_t[i]
    return s_t, min_cost


def double_insertion(s, graph, lmd):
    s_t = copy.deepcopy(s)
    min_cost = float('inf')
    pos = (0, 0, 0, 0)
    for i, route in enumerate(s_t):
        if len(route) >= 2:
            for j in range(len(route)-1):
                double_tasks = route[j:j+2]
                del route[j]
                del route[j]
                for x, r in enumerate(s_t):
                    for y in range(len(r)+1):
                            s_t[x].insert(y, double_tasks[1])
                            s_t[x].insert(y, double_tasks[0])
                            cost = evaluation_solution(s_t, graph, lmd)
                            if cost <= min_cost:
                                min_cost, pos = cost, (i, j, x, y)
                            del s_t[x][y]
                            del s_t[x][y]

                route.insert(j, double_tasks[1])
                route.insert(j, double_tasks[0])
    i, j, x, y = pos
    double_tasks = s_t[i][j:j+2]
    del s_t[i][j]
    del s_t[i][j]
    s_t[x].insert(y, double_tasks[1])
    s_t[x].insert(y, double_tasks[0])
    if len(s_t[i]) == 0:
        del s_t[i]
    return s_t, min_cost


def swap(s, graph, lmd):
    s_t = copy.deepcopy(s)
    min_cost = float('inf')
    pos = (0, 0, 0, 0)
    for i, route in enumerate(s_t):
        for j, task in enumerate(route):
            for x, route2 in enumerate(s_t):
                for y, task2 in enumerate(route2):
                    if x > i or (x == i and y >= j):
                        tmp = s_t[x][y]
                        s_t[x][y] = s_t[i][j]
                        s_t[i][j] = tmp
                        cost = evaluation_solution(s_t, graph, lmd)
                        if cost <= min_cost:
                            min_cost, pos = cost, (i, j, x, y)
                        s_t[i][j] = s_t[x][y]
                        s_t[x][y] = tmp
    i, j, x, y = pos
    tmp = s_t[x][y]
    s_t[x][y] = s_t[i][j]
    s_t[i][j] = tmp
    return s_t, min_cost


def ms_operators(s_x, graph, lmd):
    generated = set()
    traversed = False
    for i in range(100):
        # find two routes from current solutions
        a, b = generate_two_random(0, len(s_x)-1)
        cnt = 0
        while (a, b) in generated:
            cnt += 1
            if cnt > 200:
                traversed = True
                break
            a, b = generate_two_random(0, len(s_x) - 1)
        if traversed or (a, b) in generated:
            continue

        generated.add((a, b))
        parted_tasks = s_x[a] + s_x[b]
        reverse_tasks = [(item[1], item[0]) for item in parted_tasks]
        parted_tasks += reverse_tasks
        # before cost
        before_cost = evaluation_solution([s_x[a], s_x[b]], graph, lmd)
        # after cost
        s, cost = ms_operator(parted_tasks, graph, lmd)
        # update current solution
        if cost < before_cost:
            current_solutions = copy.deepcopy(s_x)
            current_solutions[a] = []
            current_solutions[b] = []
            current_solutions += s
            current_solutions = [item for item in current_solutions if item != []]
            return current_solutions
    return None


def ms_operator(parted_tasks, graph, lmd):
    best = None
    min_cost = float('inf')
    for i in range(1, 6):
        s = path_scanning(graph, parted_tasks)
        # TODO Ulusoy's splitting procedure
        c = evaluation_solution(s, graph, lmd)
        if c <= min_cost:
            min_cost = c
            best = s
    return best, min_cost


def crossover_plus_local_search(pop, graph, time_limit, start, best_feasible_cost):
    """
    ONLY USED FOR MULTIPROCESSING !!!
    :param pop: the population
    :param graph: see MAENS function for detail
    :param time_limit: skip...
    :param start: we also check time left when start a cross_over and local search
                  cause this is the most time-consuming part. so as soon as time left
                  less than 3, stop this immediately, return None, None, None
    :param best_feasible_cost: in paper, we need to get a initial lambda before start
                              a local search.
    :return:different with paper, we return one more parameters "lmd"(lambda) here,
            because we need to use it to evaluation a solution in callback function.
    """
    run_time = time.time() - start
    if time_limit - run_time < 3:
        return None, None, None
    # initial
    a, b = generate_two_random(0, len(pop) - 1)
    s_x = crossover(pop[a], pop[b], graph)
    r = generate_one_random(0, 1, flt=True)
    # get initial lambda
    lmd = get_initial_lmd(best_feasible_cost, s_x, graph)
    if r < 0.2:

        s_ls, lmd = local_search(s_x, graph, lmd)
    else:
        s_ls = None

    return s_x, s_ls, lmd


def update_pop_t(ret):
    """
    This is a callback function of a process,
    purpose1: update pop_t, i.e, store the new generated offspring in pop_t
    purpose2: update best_feasible_solution, cause we need to return a best
              feasible solution when time out.
    :param ret: s_x, s_ls, lmd = ret[0], ret[1], ret[2]
    """
    global pop_t
    global Graph
    s_x, s_ls, lmd = ret[0], ret[1], ret[2]
    if s_x is None and s_ls is None:  # only when the time is coming up, the process will directly stop and return None
        return
    if s_ls is None:  # just do crossover, not do local search
        if not is_clone(s_x, pop_t):
            heapq.heappush(pop_t, (evaluation_solution(s_x, Graph, lmd), s_x))
    else:   # do both cross over and local search
        if not is_clone(s_ls, pop_t):
            heapq.heappush(pop_t, (evaluation_solution(s_ls, Graph, lmd), s_ls))
        elif not is_clone(s_x, pop_t):
            heapq.heappush(pop_t, (evaluation_solution(s_x, Graph, lmd), s_x))

    global best_feasible_solution
    # pop_t[0] means the heap top--> (cost,solution)
    # pop_t[0][0] means the heap top cost,
    # pop_t[0][1] means the heap top solution
    if is_feasible(pop_t[0][1],Graph) and pop_t[0][0] < best_feasible_solution['cost']:
        best_feasible_solution["cost"] = pop_t[0][0]
        best_feasible_solution['solution'] = pop_t[0][1]


def MAENS(start, graph, tasks, time_limit, mul_process=False):
    """
    :param start: the start run time of this program
    :param time_limit: the run time limit of MAENS algorithms
    :param graph: is the graph read from data set, include following attributes:
            {"adjacent_matrix": the adjacent matrix presentation of the graph,
             "adjacent_table": the adjacent table presentation of the graph,
             "demand_matrix": demand_matrix,
             "dijkstra_matrix": dijkstra_matrix[i][j] store the minimum distance between node i and node j
             "basic_infor": store the basic information of this graph, e.g., depot, capacity, required edges and so on,
            }
    :param tasks: an edge tuple list, store all the task need to be finish,
                  notice, if (i, j) in tasks, then (j, i) in it also.
    :param mul_process: run MAENS on multiprocessing or not.
    :return: a feasible solution feasible within time limits.
            Example:
            --------
            s 0,(1,2),(2,3),(3,5),0,0,(1,4),(4,2),(2,9),(4,3),(5,6),0,0,(1,12),(12,6),(6,7),(7,12),0,0,(1,10),(10,9),
            (9,11),(11,5),(5,12),0,0,(1,7),(7,8),(8,10),(10,11),(11,8),0
            q 316
    """
    # initialization multiprocessing setting
    if mul_process:
        pool = Pool(processes=16)
        global Graph
        global pop_t
        global best_feasible_solution
        Graph = graph

    # generate a initial population by path_scanning
    # TODO support more initial population algorithms, e.g, Floyd Algorithm
    pop = init_population(graph, tasks, print_initial=True)
    stop_criterion = 0
    while stop_criterion != Configuration.generation_cnt:
        # initial an iteration
        pop_t = []
        for p in pop:  # set an intermediate population pop_t = pop notice, pop_t is a heap and pop is a list
            heapq.heappush(pop_t, (evaluation_solution(p, graph, 0), p))
        for p in pop:
            if is_feasible(p, graph):
                best_feasible_solution['cost'] = get_cost(p, graph)
                best_feasible_solution['solution'] = p
                break
        # multi processing code
        if mul_process:
            results = []
            for i in range(Configuration.opsize):
                result = pool.apply_async(func=crossover_plus_local_search,
                                          args=(pop, graph, time_limit, start, best_feasible_solution['cost']),
                                          callback=update_pop_t)
                results.append(result)
            # wait an iteration to finish
            for r in results:
                r.wait()
            if time_limit - (time.time() - start) < 3:
                pool.close()
                pool.join()
                print_result(best_feasible_solution['solution'], graph, paint=True)
                return 0
        # single processing code
        else:
            for i in range(Configuration.opsize):
                a, b = generate_two_random(0, len(pop)-1)
                s_x = crossover(pop[a], pop[b], graph)
                r = generate_one_random(0, 1, flt=True)
                lmd = get_initial_lmd(best_feasible_solution['cost'], s_x, graph)
                if r < Configuration.p_ls:
                    s_ls, lmd = local_search(s_x, graph, lmd)
                    if not is_clone(s_ls, pop_t):
                        heapq.heappush(pop_t, (evaluation_solution(s_ls, graph, lmd), s_ls))
                    elif not is_clone(s_x, pop_t):
                        heapq.heappush(pop_t, (evaluation_solution(s_x, graph,lmd), s_x))
                elif not is_clone(s_x, pop_t):
                    heapq.heappush(pop_t, (evaluation_solution(s_x, graph, lmd), s_x))
        # update pop with pop_t
        for i in range(len(pop)):
            item = heapq.heappop(pop_t)
            pop[i] = [task for task in item[1] if task != []]
        run_time = (time.time() - start)
        print("offspring %d generate current run_time is: %f" % (stop_criterion+1,run_time))
        print("current best feasible solution is:", best_feasible_solution )
        stop_criterion += 1
    print_result(best_feasible_solution['solution'], graph, paint=True)
    return 0


