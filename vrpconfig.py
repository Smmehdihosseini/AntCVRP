import json
from dataclasses import dataclass

@dataclass
class Config:
    ''' Config Dataclass for Ant Colony Optimzation (ACO)
    Args:
        NUM_ANTS (int): Number of Ants in Ant Colony Optimization
        ANT_CAPACITY (int): Amount of resources that an individual ant can carry or manipulate during the optimization process
        NUM_ITERATIONS (int): Number of Iterations
        DEPOT_ID (int): Depot ID
        ALPHA (float): Pheromone Importance parameter in ACO
        BETA (float): Inverse distance heuristic importance parameter in ACO
        RHO (float): Pheromone Evaporation coefficient in ACO

    '''
    try:
        json_rc = open("./envs/vrpconfig.json", 'r')
        config_json = json.load(json_rc)
        json_rc.close()

        NUM_ANTS = config_json["num_ants"]
        ANT_CAPACITY = config_json["ant_capacity"]
        NUM_ITERATIONS = config_json["num_iterations"]
        DEPOT_ID = config_json["depot_id"]
        ALPHA = config_json["alpha"]  # Pheromone importance
        BETA = config_json["beta"]  # inverse distance heuristic importance
        RHO = config_json["rho"]  # pheromone evaporation coefficient
        SETTING_PATH = config_json["setting_path"]
        DELIVERY_INFO_PATH = config_json["deliver_info_path"]
        DISTANCE_MATRIX_PATH = config_json["distance_matrix_path"]

    except:
        raise Exception("Json Config file not found in envs directory")