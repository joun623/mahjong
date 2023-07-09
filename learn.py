import sys
import numpy as np
import random
import csv
from data import tile_variety, hai, sanma_hai, ROUND_NAME, REACH_ID, MELD_ID, HAND_DISCARD_ID
from util import one_hot_id2tile, get_now_discard, hai_num_list2tilename_list, get_discard_list
from neural_network import NeuralNet, prepare_model, calc_accuracy, save_model_optimizer
from make_learn_data import make_data 
from config import EPOCH_NUM, BATCH_NUM, TRAIN_DATA_RATIO
from echo_gui import display 


def main():
    if len(sys.argv) <= 1:
        learn_hands("train")
    elif sys.argv[1] in ["train", "retrain"]:
        learn_hands(sys.argv[1])
    elif sys.argv[1] == "check":
        train_data, train_label, test_data, test_label, test_situation = make_data()
        model, optimizer = prepare_model()
        check_test_data(test_data, test_label, test_situation, model)
        # test_data, test_label, test_situation, model, state = prepare_check_data()
    else:
        learn_hands("train")

def learn_hands(learn_type):
    """
        局面データからディープラーニングを行う
        learn_type = "train" or "retrain"
    """
    is_retrain = learn_type == "retrain"

    # データ準備
    train_data, train_label, test_data, test_label, test_situation = make_data(TRAIN_DATA_RATIO) 

    model, optimizer = prepare_model(is_retrain)
    loss_log = []
    train_acc_list = []
    test_acc_list = []

    record_value = []
    print ("train_data_size = " + str(train_label.size))

    # モデル学習
    for epoch in range(EPOCH_NUM):
        batch_x, batch_y = make_minibatch(train_data, train_label, BATCH_NUM)
        model.cleargrads()
        pred_label = model(batch_x)
        loss = model.backward(pred_label, batch_y, optimizer)
        

        if epoch != 0 and epoch % 1000 == 0:
            # 誤差の計測
            train_acc, t_pred, y = calc_accuracy(model, batch_x, batch_y)
            test_acc, v_pred, y = calc_accuracy(model, test_data, test_label)
            print("epoch: " + str(epoch) + " loss:", end=" ") 
            print(loss.data, end="  train_acc: ")
            print(train_acc, end="  test_acc: ")
            print(test_acc)
            record_value.append([epoch, loss.data, train_acc, test_acc])
            with open("data.csv", "w") as f:
                writer = csv.writer(f, lineterminator='\n')
                #writer.writerows([loss_log, train_acc_list, test_acc_list])
                writer.writerows(record_value)
        else:
            print("epoch: " + str(epoch) + " loss:" + str(loss.data)) 

        if epoch != 0 and epoch % 2500 == 0:
            # モデルの保存
            save_model_optimizer(model, optimizer)


    check_test_data(test_data, test_label, test_situation, model)

def check_test_data(test_data, test_label, test_situation, model):
    # テストデータを用いてモデルを確認
    acc, pred_label, y = calc_accuracy(model, test_data, test_label)
    sort_value = np.sort(y.data, axis=1)[:,::-1]
    sort_args = np.argsort(y.data, axis=1)[:,::-1]
    # green_acc, tyuntyan_acc, jihai_acc = calc_detail_accracy(pred_label, test_label, test_situation, sort_args, sort_value)
    correct_num, another_collect_num, candidate_prob_sum = calc_another_choice(pred_label, test_label, sort_args, sort_value, 0.01)

    data_num = test_label.size
    print("test_data size = " + str(data_num))
    print("test_data size = " + str(test_data.shape[0]))

    print(acc)
    """
    print(green_acc / data_num)
    print(tyuntyan_acc / data_num)
    print(jihai_acc / data_num)
    """
    print("hand alculation acc:" + str(correct_num / data_num))
    print("another include acc:" + str(another_collect_num/ data_num))
    print("candidate_prob_sum:" + str(candidate_prob_sum / data_num))

    display(test_situation, sort_value, sort_args)
    for p, t, s, val, arg in zip(pred_label, test_label, test_situation, sort_value, sort_args):
        # if p != t:
        if not (sanma_hai[p] in tile_variety["no_green"] and sanma_hai[t] in tile_variety["no_green"]):
            pass
            # display(s)
            # if echo_situation(s, p, val, arg):
            #    break

def calc_another_choice(pred_label, correct_label, sort_args, sort_value, THRESHOLD_PROB=0.3):
    correct_num = 0
    another_collect_num = 0
    candidate_prob_sum = 0.0

    test_data_num = correct_label.size

    for pred, correct, args, values in zip(pred_label, correct_label, sort_args, sort_value):

        correct_index = np.where(args == pred)
        if pred == correct:
            correct_num += 1

        elif values[correct_index] > THRESHOLD_PROB:
            another_collect_num += 1

        candidate_prob_sum += values[correct_index]

    return correct_num, another_collect_num, candidate_prob_sum

def calc_detail_accracy(pred_label, correct_label, situation, sort_args, sort_value):
    # 詳細な制度を計測
    about_correct_num = 0
    about_correct_num2 = 0
    about_correct_num3 = 0

    url_list = [s["haihu_id"] for s in situation]
    year_list = {"2013": 0, "2014": 0, "2015": 0, "2016": 0, "2017": 0, "2018": 0}
    correct_list = {"2013": 0, "2014": 0, "2015": 0, "2016": 0, "2017": 0, "2018": 0}
    for ul in url_list:
        year_list[ul[:4]] += 1
    

    for p, t, s in zip(pred_label, correct_label, situation):
        if p != t:
            if (sanma_hai[p] in tile_variety["no_green"] and sanma_hai[t] in ["no_green"]) or (sanma_hai[p] in tile_variety["green"] and sanma_hai[t] in tile_variety["green"]) :
                about_correct_num += 1
                about_correct_num2 += 1
                about_correct_num3 += 1

            elif sanma_hai[p] in tile_variety["tyuntyan"] and sanma_hai[t] in tile_variety["tyuntyan"]:
                about_correct_num2 += 1
                about_correct_num3 += 1

            # elif (sanma_hai[p] in tile_variety["kaze"] and sanma_hai[t] in tile_variety["kaze"]) or (sanma_hai[p] in tile_variety["sangen"] and sanma_hai[t] in tile_variety["sangen"] ) or (sanma_hai[p] in tile_variety["routou"] and sanma_hai[t] in tile_variety["routou"] )  :
            elif s["discard_num"] > 47 and all([sum(s_one[1]) == 0 for s_one in s["hand"]]): # and sanma_hai[p] in tile_variety["yaotyu"] and sanma_hai[t] in tile_variety["yaotyu"]:
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
            correct_list[s["haihu_id"][:4]] += 1

    print(year_list)
    print (correct_list)
    return about_correct_num, about_correct_num2, about_correct_num3
        
def echo_situation(situation, pred_id, val, arg):
    house_name = ["東家","南家","西家"]
    print(situation["haihu_id"])
    print("ROUND_NAME: " + ROUND_NAME[situation["round_id"]])
    print("hero = " + situation["name"][situation["who_id"]]["name"])
    print("dora = " + hai[situation["dora"]])

    print("------------------")
    for i in range(3):
        print (house_name[(i - situation["round_id"]) % 3])  # 東or南or西家表示
        print("[" + situation["name"][i]["name"] + "]")  # 名前

        if i == situation["who_id"]:
            # もし打牌選択者なら
            for ar, vl in zip(arg[:3], val[:3]):
                # 打牌候補トップ3を表示
                print(sanma_hai[ar] + " : " + str(round(vl * 100, 2)), end="%,  ")
            print()

        hand_str = hai_num_list2tilename_list(situation["hand"][i][0])

        if situation["tsumo_id"] != 1000 and i == situation["who_id"]:
            # ツモがなされている場合
            hand_str.remove(hai[situation["tsumo_id"] >> 2])
            print("hand [" + " ".join(hand_str) + "   tsumo:" + hai[situation["tsumo_id"] >> 2], end="] ")
        else:
            # ツモがなされていない場合（鳴きの後の打牌）
            print("hand [" + " ".join(hand_str), end="] ")

        if sum([d[REACH_ID] for d in situation["discard"][i]]): 
            print("リーチ!")
        elif sum(situation["hand"][i][1]):
            print("鳴き!")
        else :
            print("")

        print ("correct -> " + hai[situation["select_id"] >> 2])
        print("expose [" + " ".join(hai_num_list2tilename_list(situation["hand"][i][1])) + "]")
        discard_list = get_now_discard(situation["discard"][i], situation["discard_num"])
        discard_tile_list = get_discard_list(discard_list)
        meld_list = [str(int(d[HAND_DISCARD_ID])) for d in discard_list] 

        for i in range(int(len(discard_tile_list) / 6)):
            print(" ".join(discard_tile_list[i*6 : i*6+6]))
        if len(discard_tile_list) % 6:
            print(" ".join(discard_tile_list[-(len(discard_list) % 6):]))

        for i in range(int(len(meld_list) / 6)):
            print(" ".join(meld_list[i*6 : i*6+6]))
        if len(discard_list) % 6:
            print(" ".join(meld_list[-(len(discard_list) % 6):]))


        print()
    print("------------------")
    is_next = input("go next ? ->")

    return is_next in ["no", "false", "n", "f", "end", "quit", "exit"]


def make_test_pickle():
    # 小さめの局面リストを作る
    with open("pickle/situation.pickle", mode='rb') as situation_list_pickle:
        situation_list = pickle.load(situation_list_pickle)

    with open("pickle/test_situation.pickle", mode='wb') as test_situation_pickle:
        pickle.dump(situation_list[0], test_situation_pickle) 

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
        drop_out_X = np.delete(X, data_delete_list, 0)
        drop_out_y = np.delete(y, data_delete_list, 0)

        return drop_out_X, drop_out_y

if __name__ == "__main__":
    main()
    
