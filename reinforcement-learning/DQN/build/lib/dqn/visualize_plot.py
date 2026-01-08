from typing import Dict, List
import os
import pandas as pd

from dqn.visualize import Plotter


def collect_training_logs(log_dir: str) -> Dict[str, List[pd.DataFrame]]:
    """ 
    Obtain pandas frames from progress.csv files in the given directory 
    """
    
    return [pd.read_csv(os.path.join(log_dir, folder, "progress.csv"))
                        for folder in os.listdir(log_dir)
                        if os.path.exists(os.path.join(log_dir, folder, "progress.csv"))]
    


df_dict = {"gamma-0.90": collect_training_logs(os.path.join("logs", "vanilla-dqn-gamma-0.90")),
           "gamma-0.99": collect_training_logs(os.path.join("logs", "vanilla-dqn-gamma-0.99"))}
plotter = Plotter(df_dict)
plotter()
