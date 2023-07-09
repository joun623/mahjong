import os
import numpy as np
from data import tile_variety, hai, ROUND_NAME, REACH_ID
import copy 
import pickle
from config import TRAIN_PATH, TEST_PATH, TEST_SITUATION_PATH
from util import reduce_hai_sanma, convert_sanma_id, num_list2one_hot, get_now_discard

def make_data(train_data_ratio=0.8):
    """
        データ準備
        return nd_array
    """
    data_path_list = [TRAIN_PATH, TEST_PATH, TEST_SITUATION_PATH]
    if all([os.path.exists(path) for path in data_path_list]):
        # ピクルから読み込み
        return load_pickel()

    else:
        # 局面データから読み込み
        state, select_state, situation_list = get_state()

        train_data, test_data = split_numarray(state, train_data_ratio)
        train_label, test_label = split_numarray(select_state, train_data_ratio) 
        train_situation, test_situation = split_numarray(situation_list, train_data_ratio) 
        save_pickle(train_data, train_label, test_data, test_label, test_situation)

        return train_data, train_label, test_data, test_label, test_situation

def load_pickel():
    """
        ニューラルネットへの入力データをロード
    """
    with open(TRAIN_PATH, mode='rb') as train_data_pickle:
        train_data, train_label = pickle.load(train_data_pickle)

    with open(TEST_PATH, mode='rb') as test_data_pickle:
        test_data, test_label = pickle.load(test_data_pickle)

    with open(TEST_SITUATION_PATH, mode='rb') as test_situation_pickle:
        test_situation = pickle.load(test_situation_pickle)


    return train_data, train_label, test_data, test_label, test_situation

def save_pickle(train_data, train_label, test_data, test_label, test_situation):
    """ 
        ニューラルネットへの入力データをセーブ
    """
    with open(TRAIN_PATH, mode='wb') as train_data_pickle:
        pickle.dump((train_data, train_label), train_data_pickle) 

    with open(TEST_PATH, mode='wb') as test_data_pickle:
        pickle.dump((test_data, test_label), test_data_pickle) 

    with open(TEST_SITUATION_PATH, mode='wb') as test_situation_pickle:
        pickle.dump(test_situation, test_situation_pickle) 


def get_state():
    """
        モデルの入力データを作る
    """
    situation_list = read_hand()
    state_list = []
    select_state_list = []
    match_list = []
    print("対局を読み込みます")
    for i, one_match in enumerate(situation_list):
        print(i,end=" / ")
        print(len(situation_list))
        for situation in one_match:
            if (situation["name"][situation["who_id"]]["name"] == "思考中..." and int(situation["haihu_id"][:4]) > 2015) or (situation["name"][situation["who_id"]]["name"] in ["adii", "ad11", "臼鯨", "9割ゼンツ", "９割ゼンツ"] and sum(situation["hand"][situation["who_id"]][1]) and int(situation["haihu_id"][:4]) > 2015): 

                    match_list.append(situation)
                    state_list.append(situation2state(situation))

                    # 選んだ牌の読み込み
                    select_id = convert_sanma_id(situation["select_id"] >> 2)

                    # 選んだ牌
                    select_state_list.append(select_id)

    state = np.array(state_list, dtype=np.float32)  # 説明変数
    select_state = np.array(select_state_list, dtype=np.int32)  # 目的変数
    return (state, select_state, match_list)

def read_hand():
    """
        事前に作成した局面のピクルを読み込む
    """
    pickle_path = "pickle/situation.pickle"
    with open(pickle_path, mode='rb') as situation_list_pickle:
        situation_list = pickle.load(situation_list_pickle)

    return situation_list



def situation2state(situation_dict):
    """
        局面のリストからモデルの入力データを作る
    """
    hand_id = 0  # 手牌のid 
    expose_id = 1  # 副露メンツのid 
    north_id = 2  # 北抜きのid

    my_id = situation_dict["who_id"]
    other_id = [0, 1, 2]
    other_id = list(set(other_id) - set([my_id]))
    if other_id == [0,2]:
        other_id = [2,0]

    # 自プレイヤー手牌の取得
    hand = situation_dict["hand"][situation_dict["who_id"]][hand_id]
    hand = reduce_hai_sanma(hand)
    # hand_state = [hand]

    hand_state = num_list2one_hot(hand)

    # 全プレイヤーの副露牌の取得
    meld_state = []
    tmp_meld_data = [reduce_hai_sanma(who_situation[expose_id]) for who_situation in situation_dict["hand"]]
    # meld_data = meld_data[0:3] # さんま
    meld_data = []
    meld_data.append(copy.deepcopy(tmp_meld_data[my_id]))
    meld_data.append(copy.deepcopy(tmp_meld_data[other_id[0]]))
    meld_data.append(copy.deepcopy(tmp_meld_data[other_id[1]]))

    """
    for md in meld_data:
        meld_state.append(num_list2one_hot(md))

    """
    for md in meld_data:
        meld_state.append([md])


   # 全プレイヤーの北抜の取得
    north_state = []
    tmp_north_data = [[sum(who_situation[north_id][hai.index("北"): hai.index("北") + 4])] for who_situation in situation_dict["hand"]]
    # meld_data = meld_data[0:3] # さんま
    north_data = []
    north_data.append(tmp_north_data[my_id])
    north_data.append(tmp_north_data[other_id[0]])
    north_data.append(tmp_north_data[other_id[1]])

    for nd in north_data:
        north_state.append(num_list2one_hot(nd))


    # 全プレイヤーの捨て牌の取得
    discard_state = []
    tmp_situation = []
    tmp_situation.append(situation_dict["discard"][my_id])
    tmp_situation.append(situation_dict["discard"][other_id[0]])
    tmp_situation.append(situation_dict["discard"][other_id[1]])


    for who_discard in tmp_situation:
        # tmp_who_discard = copy.deepcopy(who_discard)
        tmp_who_discard = get_now_discard(who_discard, situation_dict["discard_num"])
        for _ in range(25 - len(tmp_who_discard)):
            tmp_who_discard.append([False for _ in range(141)])
        who_discard_list = []
        for discard in tmp_who_discard:
            tmp_discard = []
            # print(discard)
            for i in range(34):
                tmp_discard.append(any(discard[i: i+4]))
            for i in range(5):
                tmp_discard.append(discard[-(i+1)])
            # print(reduce_hai_sanma(tmp_discard[:]))
            who_discard_list.append(reduce_hai_sanma(copy.deepcopy(tmp_discard)))
        discard_state.append(copy.deepcopy(who_discard_list))


    # データ結合
    state = copy.deepcopy(hand_state)
    for ms in meld_state[:]: 
        state += ms

    for ns in north_state[:]: 
        state += ns


    for ds in discard_state[:]: 
        state += ds


    # ドラ
    dora_list = [False for _ in range(27)]
    dora_list[convert_sanma_id(situation_dict["dora"])] = True
    state.append(dora_list)

    # 現状のツモ回数
    # tsumo_num_list = [False for _ in range(56)]
    # tsumo_num_list[situation_dict["tsumo_num"]] = True
    # state.append(tsumo_num_list)
    state.append([situation_dict["tsumo_num"]])

    other_lag = [lag and (not who_have[my_id]) for lag, who_have in zip(situation_dict["lag"][0], situation_dict["lag"][1]) ]
    # ラグ牌
    state.append(reduce_hai_sanma(other_lag))

    # 東一局〜東三局のどれか
    round_data = [False for _ in range(3)]
    round_data[(situation_dict["who_id"] - situation_dict["round_id"]) % 3] = True
    state.append(round_data)

    # ツモ牌
    tsumo_id = [False for _ in range(27)]
    if situation_dict["tsumo_id"] != 1000:
        tsumo_id[convert_sanma_id(situation_dict["tsumo_id"] >> 2)] = True
    state.append(tsumo_id)

    # １次元にresizeして返却
    return [flatten for inner in state for flatten in inner]


def split_numarray(np_array, split=0.8):
    """
        訓練データ、テストデータを作る
    """
    if isinstance(np_array, np.ndarray):
        len_array = np_array.shape[0]
    else:
        len_array = len(np_array)
        
    train_length = int(split * len_array)
    train = np_array[:train_length]
    test = np_array[train_length:]

    return train, test


