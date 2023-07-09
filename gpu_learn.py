import pickle
import copy 
import os
import sys
import numpy as np
import random
import chainer
from chainer import Chain, Variable, optimizers, serializers, cuda
import chainer.functions as F
import chainer.links as L
from data import tile_variety, hai, ROUND_NAME, REACH_ID
from util import one_hot_id2tile
import cupy as cp

FIRST_LAYER_NUM= 1000
SECOND_LAYER_NUM = 500
OUTPUT_LAYER_NUM = 27
EPOCH_NUM = 20000
BATCH_NUM =2048 
TRAIN_DATA_RATIO = 0.8
MODEL_PATH = "model/mj_model.npz"
STATE_PATH = "model/mj_state.npz"
TRAIN_PATH = "pickle/train_data.pickle"
TEST_PATH = "pickle/test_data.pickle"
TEST_SITUATION_PATH = "pickle/test_data_situation.pickle"

class NeuralNet(chainer.Chain):
    """
        ニューラルネットのモデル構築
        参考:http://pao2.hatenablog.com/entry/2018/07/26/203813
    """
    def __init__(self, n_units, select_unit, n_out):
        super().__init__(
            l1 = L.Linear(None, n_units),
            l2 = L.Linear(n_units, n_units),
            l3 = L.Linear(n_units, select_unit),
            l4 = L.Linear(select_unit, select_unit),
            l5 = L.Linear(select_unit, select_unit),
            l6 = L.Linear(select_unit, n_out)
        )

    def __call__(self, x):
        h1 = F.dropout(F.relu(self.l1(x)), ratio=0.3)
        h2 = F.dropout(F.relu(self.l2(h1)), ratio=0.3)
        h3 = F.dropout(F.relu(self.l3(h2)), ratio=0.3)
        h4 = F.dropout(F.relu(self.l4(h3)), ratio=0.3)
        h5 = F.dropout(F.relu(self.l5(h4)), ratio=0.3)

        return self.l6(h5)

def learn_hands(learn_type):
    """
        局面データからディープラーニングを行う
        learn_type = "train" or "retrain"
    """
    is_retrain = learn_type == "retrain"

    # データ準備
    train_data, train_label, test_data, test_label, test_situation = make_data(TRAIN_DATA_RATIO) 

    # モデル準備
    model = NeuralNet(FIRST_LAYER_NUM, SECOND_LAYER_NUM, OUTPUT_LAYER_NUM)
    model.compute_accuracy = True 
    optimizer = optimizers.Adam()
    optimizer.setup(model)

    optimizer.add_hook(chainer.optimizer.WeightDecay(0.0005))

    gpu_device = 0
    cuda.get_device(gpu_device).use()
    model.to_gpu(gpu_device)
    xp = cuda.cupy

    train_data = xp.asarray(train_data)
    train_label = xp.asarray(train_label)
    test_data = xp.asarray(test_data)
    test_label = xp.asarray(test_label)
    if all([os.path.exists(arg) for arg in [MODEL_PATH, STATE_PATH]]) and is_retrain:
        serializers.load_npz(MODEL_PATH, model)
        serializers.load_npz(STATE_PATH, optimizer)

    loss_log = []

    # モデル学習
    for epoch in range(EPOCH_NUM):
        batch_x, batch_y = make_minibatch(train_data, train_label, BATCH_NUM)
        train_data_variable = Variable(batch_x) 
        train_label_variable = Variable(batch_y)
        model.cleargrads()
        prod_label = model(train_data_variable)
        loss = F.softmax_cross_entropy(prod_label, train_label_variable)
        loss.backward()
        optimizer.update()

        """
        loss_log.append(loss.data)
        y = F.softmax(prod_label)
        prediction_label = np.argmax(y.data, 1)
        train_acc = float(np.sum(prediction_label == batch_y) / prediction_label.size)

        test_data_variable = Variable(test_data)    
        test_y = model(test_data_variable)
        test_y = F.softmax(test_y)
        test_pred_label = np.argmax(test_y.data, 1)
        test_acc = float(np.sum(test_pred_label == test_label) / test_label.size)
        """
        train_acc, t_pred = calc_accuracy(model, batch_x, batch_y)
        test_acc, v_pred = calc_accuracy(model, test_data, test_label)
        print("epoch: " + str(epoch) + " loss:", end=" ") 
        print(loss.data, end="  train_acc: ")
        print(train_acc, end="  test_acc: ")
        print(test_acc)
        if epoch != 0 and epoch % 500 == 0:
            serializers.save_npz(MODEL_PATH, model)
            serializers.save_npz(STATE_PATH, optimizer)

    serializers.save_npz(MODEL_PATH, model)
    serializers.save_npz(STATE_PATH, optimizer)


    check_test_data(test_data, test_label, test_situation, model)

def calc_accuracy(model, data, label):
    data_variable = Variable(data)    
    y = model(data_variable)
    y = F.softmax(y)
    pred_label = np.argmax(y.data, 1)
    acc = float(np.sum(pred_label == label) / pred_label.size)

    return acc, pred_label

def check_test_data(test_data, test_label, test_situation, model):
    # テストデータを用いてモデルを確認
    tyuntyan = ["②", "③", "④", "⑤", "⑥", "⑦", "⑧", "2", "3", "4", "5", "6", "7", "8"]
    no_green = ["②", "③", "④", "⑤", "⑥", "⑦", "⑧", "3", "5", "7"]
    green = ["2", "4", "6", "8"]

    """
    test_data_variable = Variable(test_data)    
    y = model(test_data_variable)
    y = F.softmax(y)
    pred_label = np.argmax(y.data, 1)
    acc = float(np.sum(pred_label == test_label) / test_label.size)
    """
    acc, pred_label = calc_accuracy(model, test_data, test_label)
    sort_value = np.sort(y.data, axis=1)[:,::-1]
    sort_args = np.argsort(y.data, axis=1)[:,::-1]
    green_acc, tyuntyan_acc, jihai_acc = calc_detail_accracy(pred_label, test_label, test_situation)

    """
    about_correct_num = 0
    about_correct_num2 = 0
    about_correct_num3 = 0


    for p, t, s in zip(pred_label, test_label, test_situation):
        if p != t:
            if (sanma_hai[p] in no_green and sanma_hai[t] in no_green) or (sanma_hai[p] in green and sanma_hai[t] in green) :
                about_correct_num += 1
                about_correct_num2 += 1
                about_correct_num3 += 1

            elif sanma_hai[p] in tyuntyan and sanma_hai[t] in tyuntyan:
                about_correct_num2 += 1
                about_correct_num3 += 1

            elif (sanma_hai[p] in tile_variety["kaze"] and sanma_hai[t] in tile_variety["kaze"]) or (sanma_hai[p] in tile_variety["sangen"] and sanma_hai[t] in tile_variety["sangen"] ) or (sanma_hai[p] in tile_variety["routou"] and sanma_hai[t] in tile_variety["routou"] )  :
                about_correct_num3 += 1

            else:
                pass
                print("pred-> ",end="")
                print(sanma_hai[p], end=", ")
                print("correct-> ",end="")
                print(sanma_hai[t])

        else:
            about_correct_num += 1
            about_correct_num2 += 1
            about_correct_num3 += 1
    """
    data_num = test_label.size
    print("test_data size = " + str(data_num))

    print(acc)
    print(green_acc / data_num)
    print(tyuntyan_acc / data_num)
    print(jihai_acc / data_num)

    for p, t, s, val, arg in zip(pred_label, test_label, test_situation, sort_value, sort_args):
        # if p != t:
        if not (sanma_hai[p] in no_green and sanma_hai[t] in no_green):
             if echo_situation(s, p, val, arg):
                 break

def calc_detail_accracy(self, pred_label, correct_label, situation):
    about_correct_num = 0
    about_correct_num2 = 0
    about_correct_num3 = 0


    for p, t, s in zip(pred_label, cprrect_label, situation):
        if p != t:
            if (sanma_hai[p] in no_green and sanma_hai[t] in no_green) or (sanma_hai[p] in green and sanma_hai[t] in green) :
                about_correct_num += 1
                about_correct_num2 += 1
                about_correct_num3 += 1

            elif sanma_hai[p] in tyuntyan and sanma_hai[t] in tyuntyan:
                about_correct_num2 += 1
                about_correct_num3 += 1

            elif (sanma_hai[p] in tile_variety["kaze"] and sanma_hai[t] in tile_variety["kaze"]) or (sanma_hai[p] in tile_variety["sangen"] and sanma_hai[t] in tile_variety["sangen"] ) or (sanma_hai[p] in tile_variety["routou"] and sanma_hai[t] in tile_variety["routou"] )  :
                about_correct_num3 += 1

            else:
                pass
                """
                print("pred-> ",end="")
                print(sanma_hai[p], end=", ")
                print("correct-> ",end="")
                print(sanma_hai[t])
                """

        else:
            about_correct_num += 1
            about_correct_num2 += 1
            about_correct_num3 += 1


        return about_correct_num, about_correct_num2, about_correct_num3Jo
        
def echo_situation(situation, pred_id, val, arg):
    house_name = ["東家","南家","西家"]
    print(situation["haihu_id"])
    print("ROUND_NAME: " + ROUND_NAME[situation["round_id"]])
    print("hero = " + situation["name"][situation["who_id"]]["name"])
    print("dora = " + hai[situation["dora"]])

    print("------------------")
    for i in range(3):
        print (house_name[(i - situation["round_id"]) % 3])
        print("[" + situation["name"][i]["name"] + "]")
        if i == situation["who_id"]:
            # print ("pred -> " + sanma_hai[pred_id], end=", correct ->")
            for ar, vl in zip(arg[:3], val[:3]):
                print(sanma_hai[ar] + " : " + str(round(vl * 100, 2)), end="%,  ")
            print()

        hand_str = hai_num_list2tilename_list(situation["hand"][i][0])

        if situation["tsumo_id"] != 1000 and i == situation["who_id"]:
            hand_str.remove(hai[situation["tsumo_id"] >> 2])
            print("hand [" + " ".join(hand_str) + "   tsumo:" + hai[situation["tsumo_id"] >> 2], end="] ")
        else:
            print("hand [" + " ".join(hand_str), end="] ")

        if sum([d[REACH_ID] for d in situation["discard"][i]]): 
            print("リーチ!")
        elif sum(situation["hand"][i][1]):
            print("鳴き!")
        else :
            print("")

        print ("correct -> " + hai[situation["select_id"] >> 2])
        print("expose [" + " ".join(hai_num_list2tilename_list(situation["hand"][i][1])) + "]")
        discard_list = get_discard_list(get_now_discard(situation["discard"][i], situation["discard_num"]))

        for i in range(int(len(discard_list) / 6)):
            print(" ".join(discard_list[i*6 : i*6+6]))
        if len(discard_list) % 6:
            print(" ".join(discard_list[-(len(discard_list) % 6):]))


        print()
    print("------------------")
    is_next = input("go next ? ->")

    return is_next in ["no", "false", "n", "f", "end", "quit", "exit"]

def make_data(train_data_ratio=0.8):
    # データ準備
    data_path_list = [TRAIN_PATH, TEST_PATH, TEST_SITUATION_PATH]
    if all([os.path.exists(path) for path in data_path_list]):
        with open(TRAIN_PATH, mode='rb') as train_data_pickle:
            train_data, train_label = pickle.load(train_data_pickle)

        with open(TEST_PATH, mode='rb') as test_data_pickle:
            test_data, test_label = pickle.load(test_data_pickle)

        with open(TEST_SITUATION_PATH, mode='rb') as test_situation_pickle:
            test_situation = pickle.load(test_situation_pickle)


    else:
        state, select_state, situation_list = get_state()
        train_data_ratio = 0.8

        train_data, test_data = split_numarray(state, train_data_ratio)
        train_label, test_label = split_numarray(select_state, train_data_ratio) 
        train_situation, test_situation = split_numarray(situation_list, train_data_ratio) 
        with open(TRAIN_PATH, mode='wb') as train_data_pickle:
            pickle.dump((train_data, train_label), train_data_pickle) 


        with open(TEST_PATH, mode='wb') as test_data_pickle:
            pickle.dump((test_data, test_label), test_data_pickle) 

        with open(TEST_SITUATION_PATH, mode='wb') as test_situation_pickle:
            pickle.dump(test_situation, test_situation_pickle) 

    return train_data, train_label, test_data, test_label, test_situation

def hai_num_list2tilename_list(hand, is_yonma=False):
    # tile_variety_num = 27 + int(is_yonma) * 7
    # tile_variety_num = 27 + int(is_yonma) * 7
    hand_list = []

    for i in range(len(hand)):
        hand_list += [hai[i] for _ in range(hand[i])]

    return hand_list

def get_discard_list(discard_list):
    discard_strs = []
    for d in discard_list:
        discard_strs.append(one_hot_id2tile(d)) 

    return discard_strs
    
def reduce_hai_sanma(hands):
    # 四麻 -> サンマ
    return hands[0:1] + hands[8:]

def situation2state(situation_dict):
    """
        局面のリストからモデルの入力データを作る
    """
    hand_id = 0  # 手牌のid 
    expose_id = 1  # 副露メンツのid 

    my_id = situation_dict["who_id"]
    other_id = [0, 1, 2]
    other_id = list(set(other_id) - set([my_id]))
    if other_id == [0,2]:
        other_id = [2,0]

    # 自プレイヤー手牌の取得
    hand = situation_dict["hand"][situation_dict["who_id"]][hand_id]
    hand = reduce_hai_sanma(hand)
    hand_state = num_list2one_hot(hand)

    # 全プレイヤーの副露牌の取得
    meld_state = []
    tmp_meld_data = [reduce_hai_sanma(who_situation[expose_id]) for who_situation in situation_dict["hand"]]
    # meld_data = meld_data[0:3] # さんま
    meld_data = []
    meld_data.append(tmp_meld_data[my_id])
    meld_data.append(tmp_meld_data[other_id[0]])
    meld_data.append(tmp_meld_data[other_id[1]])

    for md in meld_data:
        meld_state.append(num_list2one_hot(md))

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
            tmp_who_discard.append([False for _ in range(140)])
        who_discard_list = []
        for discard in tmp_who_discard:
            tmp_discard = []
            # print(discard)
            for i in range(34):
                tmp_discard.append(any(discard[i: i+4]))
            for i in range(4):
                tmp_discard.append(discard[-(i+2)])
            # print(reduce_hai_sanma(tmp_discard[:]))
            who_discard_list.append(reduce_hai_sanma(tmp_discard[:]))
        discard_state.append(who_discard_list[:])

    """
    discard_state = []
    discard_data = [reduce_hai_sanma(who_discard) for who_discard in situation_dict["discard"]]
    discard_data = discard_data[0:3] # さんま
    for dd in discard_data:
        discard_state.append(num_list2one_hot(dd))
    """

    # データ結合
    state = hand_state[:]
    for ms in meld_state[:]: 
        state += ms

    for ds in discard_state[:]: 
        state += ds


    dora_list = [False for _ in range(27)]
    dora_list[convert_sanma_id(situation_dict["dora"])] = True
    state.append(dora_list)
    state.append([situation_dict["tsumo_num"]])
    state.append(reduce_hai_sanma(situation_dict["lag"]))
    round_data = [False for _ in range(3)]
    round_data[(situation_dict["who_id"] - situation_dict["round_id"]) % 3] = True
    state.append(round_data)
    tsumo_id = [False for _ in range(27)]
    if situation_dict["tsumo_id"] != 1000:
        tsumo_id[convert_sanma_id(situation_dict["tsumo_id"] >> 2)] = True

    state.append(tsumo_id)

    return [flatten for inner in state for flatten in inner]

def get_now_discard(discard, num):
    now_discard = []
    for d in discard:
        if d[-1] < num:
            now_discard.append(d)
        else:
            break

    return now_discard

def num_list2one_hot(num_list):
    """
        ex.
        num_list = [0, 2, 1]
        return [[True, False, False, False], [False, False, True, False], [False, True, False, False]]
    """
    one_hot = []
    hai_num = 4  # 1種類の牌の枚数
    for _ in range(len(num_list)):
        one_hot.append([False for _ in range(hai_num + 1)]) # 0 ~ hai_numの計(hai_num+1)のリスト
    for i, num in enumerate(num_list):
        one_hot[i][num] = True 

    return one_hot

def convert_sanma_id(tile_id):
    """
        牌のIDをヨンマ->サンマに
    """
    if tile_id == 0:
        return 0
    else :
        return tile_id - 7

def read_hand():
    """
        事前に作成した局面のピクルを読み込む
    """
    # pickle_path = "../../pickle/test_situation.pickle"
    pickle_path = "pickle/situation.pickle"
    # situation_list_pickle
    # test_situation_pickle
    with open(pickle_path, mode='rb') as situation_list_pickle:
        situation_list = pickle.load(situation_list_pickle)

    return situation_list


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
            if situation["name"][situation["who_id"]]["name"] == "思考中...":
                match_list.append(situation)
                # 手牌の読み込み
                #self_hand = situation["hand"][situation["who_id"]][hand_id]
                one_match_state_list = situation2state(situation)
                state_list.append(situation2state(situation))

                # 選んだ牌の読み込み
                select_id = convert_sanma_id(situation["select_id"] >> 2)
                
                # 選んだ牌
                select_state_list.append(select_id)
            
    state = cp.array(state_list, dtype=cp.float32)  # 説明変数
    select_state = cp.array(select_state_list, dtype=cp.int32)  # 目的変数
    return (state, select_state, match_list)


def make_test_pickle():
    # 小さめの局面リストを作る
    with open("pickle/situation.pickle", mode='rb') as situation_list_pickle:
        situation_list = pickle.load(situation_list_pickle)

    with open("pickle/test_situation.pickle", mode='wb') as test_situation_pickle:
        pickle.dump(situation_list[0], test_situation_pickle) 

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

def make_minibatch(X, y, data_num):
    """
        ミニバッチを作成
    """
    if len(X) < data_num:
        print("長さが足りません")
        return (X, y)

    else:
        data_len_list = list(range(X.shape[0]))
        data_alive_list = random.sample(data_len_list, data_num)
        data_delete_list = list(set(data_len_list) - set(data_alive_list))
        drop_out_X = cp.delete(X, data_delete_list, 0)
        drop_out_y = cp.delete(y, data_delete_list, 0)

        return drop_out_X, drop_out_y

sanma_hai = ["一", "九", #萬子
       "①", "②", "③", "④", "⑤", "⑥", "⑦", "⑧", "⑨", #筒子
       "1", "2", "3", "4", "5", "6", "7", "8", "9", #索子
      "東", "南", "西", "北", "白", "發", "中"]

def prepare_check_data():
    path_list = [MODEL_PATH, STATE_PATH, TRAIN_PATH, TEST_PATH, TEST_SITUATION_PATH]
    if not all([os.path.exists(path) for path in path_list]):
        print("チェックするためのデータが存在しません。")
        sys.exit()

    with open(TEST_PATH, mode='rb') as test_data_pickle:
        test_tuple = pickle.load(test_data_pickle)

    with open(TEST_SITUATION_PATH, mode='rb') as test_situation_pickle:
        test_situation = pickle.load(test_situation_pickle)


    test_data = test_tuple[0]
    test_label = test_tuple[1]
    model = neuralnet(FIRST_LAYER_NUM, SECOND_LAYER_NUM, OUTPUT_LAYER_NUM)
    model.compute_accuracy = True 
    optimizer = optimizers.Adam()
    optimizer.setup(model)


    gpu_device = 0
    cuda.get_device(gpu_device).use()
    model.to_gpu(gpu_device)
    xp = cuda.cupy 
    optimizer.add_hook(chainer.optimizer.WeightDecay(0.0005))

    test_data = xp.asarray(test_data)
    test_label = xp.asarray(test_label)
    serializers.load_npz(MODEL_PATH, model)
    serializers.load_npz(STATE_PATH, optimizer)

    return test_data, test_label, test_situation, model, state

if __name__ == "__main__":
    if len(sys.argv) <= 1:
        learn_hands("train")
    elif sys.argv[1] in ["train", "retrain"]:
        learn_hands(sys.argv[1])
    elif sys.argv[1] == "check":
        test_data, test_label, test_situation, model, state = prepare_check_data()
        check_test_data(test_data, test_label, test_situation, model)
    else:
        learn_hands("train")
