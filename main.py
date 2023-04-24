
import numpy as np
import pandas as pd
import time
import random
from vrpconfig import Config
from typing import Dict, List, Optional, Tuple
import aco


if __name__ == '__main__':

    start_time = time.time()
    config = Config()
    all_solutions, best_solutions = aco.run(config, True)
    end_time = time.time()
    time_elapsed = (end_time - start_time)

    print(f"Running VRP ACO Finished in {time_elapsed}")
