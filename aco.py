import json
import numpy as np
import pandas as pd
import time
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from envs.deliveryNetwork import DeliveryNetwork
from vrpconfig import Config


class Graph:
    def __init__(self, env, config: Config):
        self.cfg = config
        self.env = DeliveryNetwork(self.cfg)
        #Connect this to ENV!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!          
        self.adjacency_map = self.create_adjacency_map()
        #Connect this to ENV!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        self.pheromone_map = self.create_pheromone_map()
        self.demand_map = {}
        self.demand_map[self.cfg.DEPOT_ID] = 0
        for i in self.env.get_delivery().keys():
            self.demand_map[i] = self.env.get_delivery()[i]['vol']
            
    def create_adjacency_map(self) -> Dict[int, Dict[int, float]]:
        adjacency_map = {}
        nodes = list(sorted(self.env.delivery_info.keys()))
        nodes.insert(0,0)
        for index_1, node_1 in enumerate(nodes):
            for node_2 in nodes[index_1+1:]:
                adjacency_map.setdefault(node_1, {})
                adjacency_map.setdefault(node_2, {})
                adjacency_map[node_1][node_2] = self.env.distance_matrix[node_1][node_2]
                adjacency_map[node_2][node_1] = self.env.distance_matrix[node_2][node_1]
        return adjacency_map

    def create_pheromone_map(self) -> Dict[int, Dict[int, float]]:
        pheromone_map = {}
        nodes = list(sorted(self.env.delivery_info.keys()))
        nodes.insert(0,0)
        for index_1, node_1 in enumerate(nodes):
            for node_2 in nodes[index_1+1:]:
                pheromone_init = 1
                pheromone_map.setdefault(node_1, {})
                pheromone_map.setdefault(node_2, {})
                pheromone_map[node_1][node_2] = pheromone_init
                pheromone_map[node_2][node_1] = pheromone_init

        return pheromone_map

    def update_pheromone_map(self, solutions: list):
        
        # apply evaporation to all pheromones
        nodes = list(sorted(self.pheromone_map.keys()))
        for index_1, node_1 in enumerate(nodes):
            for node_2 in nodes[index_1 + 1:]:
                new_value = max(round((1 - self.cfg.RHO) * self.pheromone_map[node_1][node_2], 2), 1e-10)  # * Q / avg_cost
                self.pheromone_map[node_1][node_2] = new_value

        for solution in solutions:
            pheromone_increase = 1 / solution.cost
            for route in solution.routes:
                edges = [(route[index], route[index + 1]) for index in range(0, len(route) - 1)]
                for edge in edges:
                    self.pheromone_map[edge[0]][edge[1]] += pheromone_increase

class Ant:
    def __init__(self, graph: Graph, config: Config):
        self.graph = graph
        self.cfg = config
        self.env = DeliveryNetwork(self.cfg)
        self.time_conv_to_cost = self.cfg['conv_time_to_cost']
        self.capacity = [self.env.get_vehicles()[i]['capacity'] for i in range(0, self.env.n_vehicles)]
        self.tour_time = [0 for i in range(0, self.env.n_vehicles)]
        self.vehicle = 0
        self.reset_state()

    def get_available_nodes(self, current_node):
        allowed_by_time_max = [node for node in self.nodes_left if (self.tour_time[self.vehicle]+self.graph.adjacency_map[current_node][node]+self.env.delivery_info.get(node)['crowd_cost']) <= self.env.delivery_info.get(node)['time_window_max']]
        allowed_by_capacity_time_max_min = [node for node in self.nodes_left if self.capacity[self.vehicle] >= self.graph.demand_map[node]]
        return allowed_by_capacity_time_max_min

    def select_first_node(self):
        available_nodes = self.get_available_nodes()
        return np.random.choice(available_nodes)

    def select_next_delivery(self, current_node):
        
        available_nodes = self.get_available_nodes(current_node)
        #print("AVA:", available_cities)
        
        if not available_nodes:
            return None
                        
        scores = []
        for node in available_nodes:
            scr_val = pow(self.graph.pheromone_map[current_node][node], self.cfg.ALPHA)*pow(1 / self.graph.adjacency_map[current_node][node], self.cfg.BETA)
            scores.append(scr_val)

        denominator = sum(scores)
        probabilities = [score / denominator for score in scores]
                
        next_delivery = np.random.choice(available_nodes, p=probabilities)
        
        return next_delivery

    def move_to_delivery(self, current_node, next_delivery):
        self.routes[self.vehicle].append(next_delivery)
        if next_delivery != self.cfg.DEPOT_ID:
            self.nodes_left.remove(next_delivery)
            crowd_cost = self.env.delivery_info.get(next_delivery)['crowd_cost']
        else:
            crowd_cost = 0
        self.capacity[self.vehicle] -= self.graph.demand_map[next_delivery]
        self.tour_time[self.vehicle] += self.graph.adjacency_map[current_node][next_delivery]+crowd_cost
        self.total_path_cost[self.vehicle] += (self.graph.adjacency_map[current_node][next_delivery])

    def start_new_route(self, k):
        self.routes.append([self.cfg.DEPOT_ID])
        # Randomly Select First Node
        first_city = self.select_first_node()
        self.move_to_delivery(self.cfg.DEPOT_ID, first_city)

    def find_solution(self):
        
        for vehicle in range(0, self.env.n_vehicles):
            self.vehicle = vehicle
            self.routes.append([self.cfg.DEPOT_ID])
            current_node = self.routes[self.vehicle][-1]
            available_nodes = self.get_available_nodes(current_node)
            first_delivery = np.random.choice(available_nodes)
            self.move_to_delivery(self.cfg.DEPOT_ID, first_delivery)
            self.total_path_cost[self.vehicle] += self.env.get_vehicles()[self.vehicle]['cost']
        while self.nodes_left:
            print("PRINT", self.routes)
            for vehicle in range(0, self.env.n_vehicles):
                self.vehicle = vehicle
                current_node = self.routes[self.vehicle][-1]
                if len(self.routes[self.vehicle])>2 and current_node==self.cfg.DEPOT_ID:
                    continue
                else:
                    next_delivery = self.select_next_delivery(current_node)
                    #print("NEXT", next_delivery)
                    if (next_delivery is None):
                        self.move_to_delivery(current_node, self.cfg.DEPOT_ID)
                    else:
                        self.move_to_delivery(current_node, next_delivery)

        for vehicle in range(0, self.env.n_vehicles):
            self.vehicle = vehicle
            if self.routes[self.vehicle][-1] != self.cfg.DEPOT_ID:
                self.move_to_delivery(self.routes[self.vehicle][-1], self.cfg.DEPOT_ID)     
        return Solution(self.routes,
                sum(self.total_path_cost))

    def reset_state(self):
        self.capacity = [self.env.get_vehicles()[i]['capacity'] for i in range(0, self.env.n_vehicles)]
        self.nodes_left = set(self.graph.adjacency_map.keys())
        self.nodes_left.remove(self.cfg.DEPOT_ID)
        self.routes = []
        self.tour_time = [0 for i in range(0, self.env.n_vehicles)]
        self.total_path_cost = [0 for i in range(0, self.env.n_vehicles)]

@dataclass
class Solution:
    routes: List[int]
    cost: float

def get_route_cost(route, graph: Graph):
    total_cost = 0
    for i in range(0, len(route) - 1):
        total_cost += round(graph.adjacency_map[route[i]][route[i + 1]], 5)
    return total_cost


def get_route_cost_opt(route, graph: Graph, DEPOT_ID):
    # Assumes route middle without starting and ending depot
    # Add depot transport cost
    depot_costs = round(graph.adjacency_map[DEPOT_ID][route[0]], 5) + round(graph.adjacency_map[route[-1]][DEPOT_ID], 5)

    return depot_costs + get_route_cost(route, graph)


def two_opt(route, i, j) -> List[int]:
    """
    Perform two opt swap
    >>> two_opt([1, 2, 3, 4, 5, 6], 1, 3)
    [1, 4, 3, 2, 5, 6]
    """
    return route[:i] + route[i:j + 1][::-1] + route[j + 1:]


def get_better_two_opt_swap(route, graph, DEPOT_ID) -> Optional[List[int]]:
    num_eligible_nodes_to_swap = len(route)
    route_cost = get_route_cost_opt(route, graph, DEPOT_ID)
    for i in range(0, num_eligible_nodes_to_swap - 1):
        for k in range(i + 1, num_eligible_nodes_to_swap):
            new_route = two_opt(route, i, k)
            new_cost = get_route_cost_opt(new_route, graph, DEPOT_ID)
            if new_cost < route_cost:
                return new_route 
    return None


def get_optimal_route_intraswap(route, graph, DEPOT_ID):
    best_route = route

    while True:
        improved_route = get_better_two_opt_swap(best_route, graph, DEPOT_ID)
        if improved_route:
            best_route = improved_route
        else:
            break
    return best_route


def apply_two_opt(initial_solution, graph, DEPOT_ID):
    best_routes = [
        [DEPOT_ID] + get_optimal_route_intraswap(route[1:-1], graph, DEPOT_ID) + [DEPOT_ID]
        for route in initial_solution.routes
    ]

    return Solution(best_routes,
                    sum([get_route_cost(route, graph) for route in best_routes]))


def run(cfg: Config, verbose: bool = True) -> Tuple[Solution, List[Solution]]:
    """
    :param cfg: Config
    :param verbose: Should it print output
    :return: Best solution and list of all solution in every iteration
    """

    graph = Graph(DeliveryNetwork(Config), cfg)
    ants = [Ant(graph, cfg) for i in range(0, cfg.NUM_ANTS)]

    best_solution = None

    all_solutions = []

    candidates = []
    for i in range(1, cfg.NUM_ITERATIONS + 1):
        for ant in ants:
            ant.reset_state()
        solutions = []
        for ant in ants:
            an_sol = ant.find_solution()
            solutions.append(an_sol)


        candidate_best_solution = min(solutions, key=lambda solution: solution.cost)
        if cfg.USE_2_OPT_STRATEGY:
            candidate_best_solution = apply_two_opt(candidate_best_solution, graph, cfg.DEPOT_ID)

        candidates.append(candidate_best_solution)
        if not best_solution or candidate_best_solution.cost < best_solution.cost:
            best_solution = candidate_best_solution

        if verbose:
            print("Best Solution in Iteration {}/{} = {}".format(i, cfg.NUM_ITERATIONS, best_solution.cost))
            all_solutions.append(best_solution.cost)
        graph.update_pheromone_map(solutions)

    if verbose:
        print("---")
        print("Final Best Solution:")
        print("---")
        print("Best Solution Cost: \n", best_solution.cost)
        print("Best Solution Routes: \n", best_solution.routes)
        
    return all_solutions, best_solution