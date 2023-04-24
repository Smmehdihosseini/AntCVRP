import math
import json
import numpy as np
from scipy import spatial
import matplotlib.pyplot as plt

class DeliveryNetwork():
    def __init__(self, settings):
        
        super(DeliveryNetwork, self).__init__()
        self.settings = settings

        # Tech Paramenters
        self.conv_time_to_cost = settings.conv_time_to_cost

        # Basic cardinalities:
        self.n_deliveries = settings.n_deliveries
        self.n_vehicles = settings.n_vehicles

        # DELIVERY DEFINITION
        self.delivery_info = {}
        points = [[0,0]]

        if data_csv:
            file1 = open(settings.deliver_info_path, 'r')
            self.delivery_info = json.load(file1)

            temp_del_info = {}
            for k in self.delivery_info:
                v = self.delivery_info[k]
                temp_del_info[int(k)] = v
            self.delivery_info = temp_del_info

            file1.close()
            '''
            lines = file1.readlines()
            for i, line in enumerate(lines):
                if i == 0:
                    continue
                tmp = line.strip().split(',')
                print("Line{}: {}".format(i, tmp))
                self.delivery_info[int(tmp[0])] = {
                    'id': int(tmp[0]),
                    'lat': float(tmp[1]),
                    'lng': float(tmp[2]),
                    'crowdsourced': 0,
                    'vol': float(tmp[3]),
                    'crowd_cost': float(tmp[4]),
                    'p_failed': float(tmp[5]),
                    'time_window_min': float(tmp[6]),
                    'time_window_max': float(tmp[7]),
                }
                points.append([float(tmp[1]), float(tmp[2])])
            '''
        else:
            mean = [0, 0]
            cov = [[1, 0], [0, 1]]
            x, y = np.random.multivariate_normal(mean, cov, self.n_deliveries).T
            self.__initialize_stochastic()
            items_vols = self.generate_vols(self.n_deliveries)            
            for i in range(self.n_deliveries):
                points.append([x[i], y[i]])
                
                time_window_min = (1 + np.random.uniform()) * math.sqrt(x[i]**2 + y[i]**2)
                time_window_max = time_window_min + 2 + 3 * np.random.uniform() 

                self.delivery_info[i + 1] = {
                    'id': i + 1,
                    'lat': x[i],
                    'lng': y[i],
                    'crowdsourced': 0,
                    'vol': items_vols[i],
                    'crowd_cost': self.compute_delivery_costs(items_vols[i]),
                    'p_failed': 0.5,
                    'time_window_min': time_window_min,
                    'time_window_max': time_window_max,
                }

        # load the distance matrix from the csv file
        self.distance_matrix = np.loadtxt(open(settings.distance_matrix_path, "rb"), delimiter=",")

        # load the position of the depot
        self.depot = settings.depot

        # NB: do not assume that the is symetric!
        '''
        if settings['distance_function'] == 'euclidian':
            self.distance_matrix = spatial.distance_matrix(points, points)
        ''' 
        
        # VEHICLE DEFINITION
        self.vehicles = []
        for i in range(self.n_vehicles):
            self.vehicles.append(
                {
                    'capacity': settings.vols_vehicles[i],
                    'cost': settings.costs_vehicles[i]
                }
            )

    def prepare_crowdsourcing_scenario(self):
        self.__fail_crowdship = []
        for _, ele in self.delivery_info.items():
            if np.random.uniform() < ele['p_failed']:
                self.__fail_crowdship.append(ele['id'])

    def run_crowdsourcing(self, delivery_to_crowdship):
        id_remaining_deliveries = [key for key in self.delivery_info]
        tot_crowd_cost = 0
        # RE INITIALIZE
        for key, ele in self.delivery_info.items():
            ele['crowdsourced'] = 0
        # UPTADE ACCORDING TO SIMULATION
        for i in delivery_to_crowdship:
            if self.delivery_info[i]['id'] not in self.__fail_crowdship:
                id_remaining_deliveries.remove(i)
                tot_crowd_cost += self.delivery_info[i]['crowd_cost']
                self.delivery_info[i]['crowdsourced'] = 1
        remaining_deliveries = {}
        for i in id_remaining_deliveries:
            remaining_deliveries[i] = self.delivery_info[i]
        return remaining_deliveries, tot_crowd_cost

    def get_delivery(self):
        return self.delivery_info

    def get_vehicles(self):
        return self.vehicles

    def __initialize_stochastic(self):
        funct_cost_dict = {
            'constant': lambda x: self.settings.funct_cost_dict['K']*x,
        }
        self.compute_delivery_costs = funct_cost_dict[self.settings.funct_cost_dict['name']]

        vol_distr_dict = {
            'uniform': lambda x: np.around(
                np.random.uniform(
                    low=self.settings['vol_distr']['min_vol_bins'],
                    high=self.settings['vol_distr']['max_vol_bins'],
                    size=x
                )
            ),
        }
        self.generate_vols = vol_distr_dict[self.settings['vol_distr']['name']]

    def evaluate_VRP(self, VRP_solution):
        # USAGE COST
        usage_cost = 0
        for k in range(self.n_vehicles):
            if len(VRP_solution[k]) > 0:
                usage_cost += self.vehicles[k]['cost']
        # TOUR COST and CHECK TIME WINDOWS
        travel_cost = 0
        for k in range(self.n_vehicles):
            travel_time = 0
            tour_time = 0
            for i in range(1, len(VRP_solution[k])-1):
                tour_time += self.distance_matrix[
                    VRP_solution[k][i - 1],
                    VRP_solution[k][i],
                ]
                travel_time += self.distance_matrix[
                    VRP_solution[k][i - 1],
                    VRP_solution[k][i],
                ]
                tour_time = max(
                    tour_time, self.delivery_info[VRP_solution[k][i]]['time_window_min']
                )
                if tour_time > self.delivery_info[VRP_solution[k][i]]['time_window_max']:
                    print(tour_time)
                    raise Exception('Too Late for Delivery: ', VRP_solution[k][i])
            travel_cost += self.conv_time_to_cost * tour_time

        # CHECK VOLUME
        for k in range(self.n_vehicles):
            tot_vol_used = 0
            for i in range(1, len(VRP_solution[k]) - 1):
                tot_vol_used += self.delivery_info[VRP_solution[k][i]]['vol']

            if tot_vol_used > self.vehicles[k]['capacity']:
                raise Exception(f"Capacity Bound Violeted {tot_vol_used}>{self.vehicles[k]['capacity']}")

        return usage_cost + travel_cost

    def render(self):
        plt.figure()
        plt.scatter(0, 0, c='green', marker='s')
        for key, ele in self.delivery_info.items():
            plt.scatter(ele['lat'], ele['lng'], c='blue' if ele['crowdsourced'] else 'red')
        plt.show()
    
    def render_tour(self, remaining_deliveries, VRP_solution):
        # PLOT DATA VRP
        for k in range(self.n_vehicles):
            print(f"** Vehicle {k} **")
            tour_time = 0
            for i in range(1, len(VRP_solution[k]) - 1 ):
                tour_time += self.distance_matrix[
                    VRP_solution[k][i - 1],
                    VRP_solution[k][i],
                ]
                delivery = self.delivery_info[VRP_solution[k][i]]
                tour_time_after_waiting = max(
                    tour_time,
                    delivery['time_window_min']
                )
                print(VRP_solution[k][i]-1)
                print(f"node: {delivery['id']}  arrival time: {tour_time:.2f}  [ {delivery['time_window_min']}-{delivery['time_window_max']} ] ")
                tour_time = tour_time_after_waiting
            print(f"** **")

        fig = plt.figure()
        # PRINT DELIVERY
        plt.scatter(self.depot[0], self.depot[1], c='green', marker='s')
        for ele_id, ele in self.delivery_info.items():
            plt.scatter(ele['lat'], ele['lng'], c='red' if ele_id in remaining_deliveries else 'blue')
            plt.text(ele['lat'], ele['lng'], ele['id'], fontdict=dict(color='black', alpha=0.5, size=16))
        self._add_tour(VRP_solution)
        # if you want to save the picture:
        # fig.savefig('./results/comparison.png', dpi=200) 
        # if you want to show the graph
        plt.show()


    def _add_tour(self, VRP_solution):
        dict_vehicle_char = [
            ('red', '--'),
            ('blue', '.'),
            ('green', '-'),
            ('black', '.-'),
            ('yellow', '.-'),
            ('gray', '.-'),
            ('purple', '.-'),
            ('darkcyan', '.-'),
            ('plum', '.-'),
            ('maroon', '.-')
        ]
        for k in range(self.n_vehicles):
            if len(VRP_solution[k]) == 0:
                continue
            plt.plot(
                [self.depot[0], self.delivery_info[VRP_solution[k][1]]['lat']],
                [self.depot[1], self.delivery_info[VRP_solution[k][1]]['lng']],
                color=dict_vehicle_char[k][0]
            )
            for i in range(1, len(VRP_solution[k])-2):
                plt.plot(
                    [self.delivery_info[VRP_solution[k][i]]['lat'], self.delivery_info[VRP_solution[k][i + 1]]['lat']],
                    [self.delivery_info[VRP_solution[k][i]]['lng'], self.delivery_info[VRP_solution[k][i + 1]]['lng']],
                    color=dict_vehicle_char[k][0]
                )
            plt.plot(
                [self.delivery_info[VRP_solution[k][-2]]['lat'], self.depot[0]],
                [self.delivery_info[VRP_solution[k][-2]]['lng'], self.depot[1]],
                color=dict_vehicle_char[k][0]
            )