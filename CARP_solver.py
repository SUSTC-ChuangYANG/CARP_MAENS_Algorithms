import sys
from utils import file_to_graph
from djiktra import get_dijkstra_matrix
import numpy as np
from MAENS import MAENS
import time
if __name__ == "__main__":
    file_name = sys.argv[1]
    time_limit = int(sys.argv[3])
    seed = int(sys.argv[5])
    start = time.time()
    # file_name = "val1A.dat"
    # time_limit = 120
    graph = file_to_graph(file_name)
    get_dijkstra_matrix(graph)
    idx = np.where((graph['demand_matrix'] != -1) & (graph['demand_matrix'] != 0))
    tasks = list(zip(idx[0], idx[1]))  # all direction are stored

    MAENS(start=start,graph=graph,tasks=tasks,time_limit=time_limit,mul_process=True)
    # plan = path_scanning(graph=graph, tasks=tasks, rule_code=1)
    # print_result(plan, graph)
