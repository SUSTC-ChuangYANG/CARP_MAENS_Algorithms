from config import Configuration
import random


def path_scanning(graph, tasks):
    free = tasks.copy()
    demand_matrix = graph['demand_matrix']
    dijkstra_matrix = graph['dijkstra_matrix']
    S = []
    while len(free) > 0:
        route = []
        last_end = Configuration.DEPOT
        load = 0
        while load < Configuration.CAPACITY:
            candidates = []
            min_distance = float('inf')
            for task in free:
                distance = dijkstra_matrix[last_end][task[0]]
                if distance < min_distance:
                    candidates.clear()
                    candidates.append(task)
                    min_distance = distance
                elif distance == min_distance:
                    candidates.append(task)

            candidate_count = len(candidates)
            #print("totally candidate:", candidate_count)
            if candidate_count == 0:   # capacity is nearly full, stop this route
                break
            elif candidate_count == 1: # directly select one from one
                selected = candidates[0]
            else:
                r = random.randint(0, candidate_count-1)
                selected = candidates[r]
                #selected = rule(load,candidates,graph,rule_code)
            if demand_matrix[selected[0]][selected[1]]+load > Configuration.CAPACITY:
               break
            free.remove(selected)  # remove current task and reverse task
            free.remove((selected[1], selected[0]))
            route.append(selected)  # add route
            last_end = selected[1]
            load += demand_matrix[selected[0], selected[1]]
        S.append(route)

    return S
