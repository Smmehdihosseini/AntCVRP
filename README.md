# VRP Solution with Ant Colony Optimization

This repository contains a solution for the Vehicle Routing Problem (VRP) using Ant Colony Optimization (ACO) algorithm. The VRP is a classic optimization problem that aims to minimize the total distance traveled by a set of vehicles in order to serve a set of customers, subject to a set of constraints. This problem has practical applications in logistics, transportation, and distribution.

## Solution Overview

Our solution uses the ACO algorithm to find a set of routes for the vehicles to follow, such that the total distance traveled is minimized. The ACO algorithm is inspired by the foraging behavior of ants, where the ants deposit pheromones to mark the paths that lead to food. In the ACO algorithm, artificial ants also deposit pheromones on the paths that they traverse, and the pheromone levels are used to guide the search for the optimal solution.

The solution is implemented in Python, using the NumPy library for efficient matrix operations. The code is modular and well-documented, and can be easily extended or modified to suit specific requirements.

## How to Use

To use the VRP solution, follow these steps:

1. Install Python 3.x and NumPy library on your system.
2. Clone this repository to your local machine.
3. In the repository directory, install the packages using `pip install -r requirements.txt` (virtual environment is recommended)
3. Check out the repository files and script structures:
- `envs`: This folder contains the json config file. First, change the config parameters (including the ACO parameters according to your desire).
- `envs/data`: Copy and paste your VRP data into this folder (There is no data folder here, you should add it yourself). There must be two files at least, first the `delivery_info.json` file and also the `distance_matrix.csv` (distances between nodes) file. Pay attention to the typo of the filenames.
4. Run `python main.py` to get the ACO results. You can obsereve the optimal route(s) and the correspoding cost to it. (You can also write additonal codes to this file in order to get further results with all solutions.

Example of `delivery_info.json` item:

```javascript
{
  "1": {
    "crowd_cost": 5.501245440387771,
    "crowdsourced": 0,
    "id": 1,
    "lat": 45.07134426639958,
    "lng": 7.693087346614441,
    "p_failed": 0.7883287888960132,
    "time_window_max": 9,
    "time_window_min": 6,
    "vol": 0.882859998043791
  }
 }
```

## Contributions

Contributions are welcome! If you find a bug, have a suggestion, or want to contribute code, please submit an issue or a pull request.
