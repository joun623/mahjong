import numpy as np
import matplotlib.pyplot as plt
import pickle
from read_mjlog import Mjlog, MjlogContents
 
pickle_path = "pickle/result.pickle"

with open(pickle_path, mode='rb') as result_pickle:
    mjlog_data = pickle.load(result_pickle)

    plt.rcParams['font.family'] = 'IPAPGothic' 
    print(mjlog_data)
    for name, player_data in mjlog_data.items():
        print(player_data)
        play_num = player_data["match_num"]
        
        if play_num > 500:
            plt.plot(player_data["rate"], label=name)

    # plt.legend(loc="upper top")
    plt.show()
