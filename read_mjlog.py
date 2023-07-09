from match_data import MatchData
import os
import re
import pickle
import copy
import math
from util import extract_in_doublequates, url_decode, int2bit
from data import DAN, MATCH_TYPE, YAKU, YAKUMAN, HANDS_INDEX, EXPOSED_MELD_INDEX, hai

class MjlogContents:
    '''
        一戦のデータを読み込む
        !! 現在log_testを読み込んでいる
    '''
    def __init__(self):
        self.basic_data = {}
        self.user = []    
        self.all_match_data = []
        self.type = []
    
    def read_mjlog(self, remove_bracket_elements, file_name):
        match_delimiter = "INIT"
        split_elements = []
        tmp_elements = [] 

        # 基本データ(basic_data)と対局データ(match_data)に分ける

        for elem in remove_bracket_elements:
            if elem.startswith(match_delimiter):
               split_elements.append(tmp_elements) 
               tmp_elements = []
            tmp_elements.append(elem)

        if len(tmp_elements) > 0 and tmp_elements[0].startswith(match_delimiter) and tmp_elements[-1].endswith("mjloggm"):
            split_elements.append(tmp_elements[:-1])
        # 基本データ 
        self.read_basic_mjlog(split_elements[0])
        if "四麻" in self.type:
            return

        # 対局データ
        self.read_match_data(split_elements[1:], file_name)

    def read_basic_mjlog(self, basic_elements):
        # 基本データを読み込み
        # 今後対局者名等しっかり分ける
        for elem in basic_elements:
            split_space = elem.split(" ")
            self.basic_data[split_space[0]] = " ".join(split_space[1:])

        self.read_user()
        self.read_type()

    def read_user(self):
        '''
            user情報を取得
        '''
        self.user = []
        un_data = [extract_in_doublequates(un) for un in self.basic_data["UN"].split(" ")]
        user_name = [url_decode(name) for name in un_data[0:3]]
        dan_data = [DAN[int(dan)] for dan in un_data[4].split(",")]
        rate_data = [float(rate) for rate in un_data[5].split(",")]
        sex_data = un_data[6].split(",")

        for name, dan, rate, sex in zip(user_name, dan_data, rate_data, sex_data):
            tmp_user = {
                "name": name,
                "dan": dan,
                "rate": rate,
                "sex": sex
                }

            self.user.append(tmp_user)
    
    def read_type(self):
        """
            対局情報を取得
        """
        type_str = self.basic_data["GO"].split(" ")[0]
        type_num = int(extract_in_doublequates(type_str))
        bit_type = int2bit(type_num, len(MATCH_TYPE))

        for i, bit in enumerate(bit_type):
            self.type.append(MATCH_TYPE[i][int(bit)])

        # print(self.type)

    def read_match_data(self, match_elements, file_name):
        # 対局データを読み込み
        discards = []

        for one_match in match_elements:
            # 一局ごとの処理
                self.all_match_data.append(MatchData(one_match, self.user, file_name))

class Mjlog:
    def __init__(self):
        self.contents_list = []
        self.result_data = {}
        self.match_num = {}
        self.win_num = {}
        self.discard_tile_num = {}
        self.table_rate = [1500]
        # self.record_result()

    def read_mjlog_files(self, xml_pass):
        files = os.listdir(xml_pass)
        files.sort()
        len_files = len(files)
        for i, file_name in enumerate(files):
            print(str(i) + "/" + str(len_files), end=", ")
            print(file_name)
            with open(xml_pass + file_name)  as f:
                mjlog_str = f.read()
                mjlog_elements = re.findall("<.+?>", mjlog_str)

                # かっこの除きかたが雑
                remove_bracket_elements = [elem[1:-1] for elem in mjlog_elements]
                contents = MjlogContents()
                contents.read_mjlog(remove_bracket_elements, file_name)
                if "三麻" in contents.type:
                    self.contents_list.append(contents)

    def record_result(self):
        all_match_num = 0
        all_win_num = 0
        pinzu_data_list = []
        souzu_data_list = []

        print ("content_len = ")
        print (len(self.contents_list))
        for contents in self.contents_list:
            # 各試合
            for data in contents.all_match_data: 
                if data.result["is_disconnect_end"]:
                    continue

                # 各局
                all_match_num += 1
                meld_tile_num = sum([one_hand[EXPOSED_MELD_INDEX].count(True) for one_hand in data.hands])

                for u in data.user:
                    # 各ユーザごとの処理（初期化も含めて）
                    name = u["name"] 
                    if name not in self.result_data:
                        self.result_data[name] = {} 
                        self.result_data[name]["match_num"] = 0 
                        self.result_data[name]["yakuman_num"] = 0
                        self.result_data[name]["meld_tile_num"] = 0
                        self.result_data[name]["rate"] = [1500]
                    self.result_data[name]["match_num"] += 1
                    self.result_data[name]["yakuman_num"] += sum([w["is_win"] for w in data.result["win_data"]])
                    self.result_data[name]["meld_tile_num"] += meld_tile_num 

                for win_data in data.result["win_data"]:
                    # 和了に関する処理
                    all_win_num += win_data["is_win"]
                    if win_data["is_win"]:
                        # 和了が発生した時

                        # 和了者、放銃者のデータ追加
                        win_name = data.user[win_data["who_id"]]["name"]
                        if "win_num" not in self.result_data[win_name]:
                            self.result_data[win_name]["win_num"] = 0
                        self.result_data[win_name]["win_num"] += 1

                        # 放銃者のデータ追加 
                        from_name = data.user[win_data["from_who_id"]]["name"]
                        if "discard_tile_num" not in self.result_data[from_name]:
                            # ダブルででた時2回計上してしまう
                            self.result_data[from_name]["discard_tile_num"] = 0
                        self.result_data[from_name]["discard_tile_num"] += 1

                        # 和了役の追加
                        if "yaku" not in self.result_data[win_name]:
                            self.result_data[win_name]["yaku"] = []

                        self.result_data[win_name]["yaku"].append(win_data["yaku_list"])

                        if len(set([hai[i >> 2] for h_m in data.hands[win_data["who_id"]][0:2] for i, h in enumerate(h_m) if h]) & set(["②", "③", "④", "⑤", "⑥", "⑦", "⑧"])):
                            print(data.hands[win_data["who_id"]][0] )
                            pinzu_data_list.append(win_data["yaku_list"])
                        if len(set([hai[i >> 2] for h_m in data.hands[win_data["who_id"]][0:2] for i, h in enumerate(h_m) if h]) & set(["2", "3", "4", "5", "6", "7", "8"])):
                            souzu_data_list.append(win_data["yaku_list"])

                # レート処理
                rate_ave = sum([self.result_data[u["name"]]["rate"][-1] for u in data.user]) / 3 #len(data.user)
                three_user_match_num = [self.result_data[u["name"]]["match_num"] for u in data.user]
                for u in data.user:
                    # K = 50 
                    K = 10 
                    """
                    user_match_num = self.result_data[u["name"]]["match_num"] 
                    if user_match_num < 200:
                        K = K * 5 * (1 - user_match_num * 0.001)
                    elif any(num < 100 for num in three_user_match_num):
                        K = K * 0.2   
                    """

                    self.result_data[u["name"]]["rate"].append(self.result_data[u["name"]]["rate"][-1] + K * (int(data.result["win_data"][0]["is_win"]) - (1 / (1 + math.pow(10, (self.table_rate[-1] - rate_ave) / 400)))))

                K = 10
                self.table_rate.append(self.table_rate[-1] + K * (int(not data.result["win_data"][0]["is_win"]) - (1 / (1 + math.pow(10, (rate_ave - self.table_rate[-1]) / 400 )))))
                # self.result_data[u["name"]]["rate"].append(self.result_data[u["name"]]["rate"][-1] + K * (int(data.result["win_data"][0]["is_win"]) - ((1 / 10000) * (rate_ave - 1500) + (1 / 3))))
                
                """
                rate_ave = sum([self.result_data[u["name"]]["rate"][-1] for u in data.user]) / 3 #len(data.user)
                three_user_match_num = [self.result_data[u["name"]]["match_num"] for u in data.user]
                for u in data.user:
                    # K = 50 
                    K = 10 
                    user_match_num = self.result_data[u["name"]]["match_num"] 
                    if user_match_num < 200:
                        K = K * 5 * (1 - user_match_num * 0.001)
                    elif any(num < 100 for num in three_user_match_num):
                        K = K * 0.2   
                    
                    self.result_data[u["name"]]["rate"].append(self.result_data[u["name"]]["rate"][-1] + K * (int(data.result["win_data"][0]["is_win"]) - ((1 / 10000) * (rate_ave - 1500) + (1 / 3))))
                """



        yakuman_name = YAKU + ["数え役満"]
        pinzu_yaku_list = [0 for _ in range(len(yakuman_name))]
        souzu_yaku_list = [0 for _ in range(len(yakuman_name))]
        for yaku in pinzu_data_list:
            if len(yaku) < 4:
                for y in yaku:
                    pinzu_yaku_list[y] += 1
            else:
                pinzu_yaku_list[-1] += 1

        for yaku in souzu_data_list:
            if len(yaku) < 4:
                for y in yaku:
                    souzu_yaku_list[y] += 1
            else:
                print(yaku)
                souzu_yaku_list[-1] += 1

        pinzu_yaku_dict = {}
        souzu_yaku_dict = {}

        for name, pinzu_yaku, souzu_yaku in zip(yakuman_name, pinzu_yaku_list, souzu_yaku_list):
            pinzu_yaku_dict[name] = pinzu_yaku
            souzu_yaku_dict[name] = souzu_yaku

        pinzu_sort_yaku_list = sorted(pinzu_yaku_dict.items(), key=lambda x: -x[1])
        souzu_sort_yaku_list = sorted(souzu_yaku_dict.items(), key=lambda x: -x[1])

        print("pinzu-zu")
        print(pinzu_sort_yaku_list)
        print(souzu_sort_yaku_list)
        for yaku_data in pinzu_sort_yaku_list:
            if yaku_data[0] in YAKUMAN:
                if yaku_data[1] > 0:
                    print("{}: {}({:.1f}%)".format(yaku_data[0], yaku_data[1],float(yaku_data[1] / sum(pinzu_yaku_dict.values()) * 100)), end=", ")
        print("")

        print("sozu-zu")
        for yaku_data in souzu_sort_yaku_list:
            if yaku_data[0] in YAKUMAN:
                if yaku_data[1] > 0:
                    print("{}: {}({:.1f}%)".format(yaku_data[0], yaku_data[1],float(yaku_data[1] / sum(souzu_yaku_dict.values()) * 100)), end=", ")
        print("")


        
        for name, player_data in self.result_data.items():
            # 各種データ表示
           play_num = player_data["match_num"]
           
           if "win_num" in player_data:
               win_rate = float(player_data["win_num"] / play_num)
           else:
               win_rate = -1

           if "discard_tile_num" in player_data:
               discard_rate = float(player_data["discard_tile_num"] / play_num)
           else:
               discard_rate = -1

           if "yakuman_num" in player_data:
               yakuman_rate = float(player_data["yakuman_num"] / play_num)
           else:
               yakuman_rate = -1 

           if "meld_tile_num" in player_data:
               average_meld_tile = float(player_data["meld_tile_num"] / play_num)
           else:
               average_meld_tile = -1 



           if play_num >= 100:
                print("-----------------")
                print(name)
                print("対局数 %d " % play_num)
                if "win_num" in player_data:
                    print("和了率 {:.4f} ".format(win_rate))

                if "discard_tile_num" in player_data:
                   
                    print("放銃率 {:.4f}".format(float(player_data["discard_tile_num"] / play_num)))
                if "yakuman_num" in player_data:
                    print("役満率 {:.4f}".format(float(player_data["yakuman_num"] / play_num)))

                if "meld_tile_num"  in player_data:
                    print("晒牌数平均 {:.4f}".format(float(player_data["meld_tile_num"] / play_num)))

                print("rate: ")
                print( player_data["rate"])


                if "yaku" in player_data:
                    yakuman_name = YAKU + ["数え役満"]
                    yaku_list = [0 for _ in range(len(yakuman_name))]
                    for yaku in player_data["yaku"]:
                        for y in yaku:
                            yaku_list[y] += 1

                    yaku_dict = {}
                    
                    for name, yaku in zip(yakuman_name, yaku_list):
                        yaku_dict[name] = yaku 

                    sort_yaku_list = sorted(yaku_dict.items(), key=lambda x: -x[1])

                    for yaku_data in sort_yaku_list:
                        if yaku_data[0] in YAKUMAN:
                            if yaku_data[1] > 0:
                                print("{}: {}({:.1f}%)".format(yaku_data[0], yaku_data[1],float(yaku_data[1] / len(player_data["yaku"]) * 100)), end=", ")
        print("table_rate: " + str(self.table_rate))

if __name__ == "__main__":
    pickle_path = 'pickle/mjlog.pickle'
    """
    if os.path.exists(pickle_path):
        with open(pickle_path, mode='rb') as mjlog_data_pickle:
            mjlog_data = pickle.load(mjlog_data_pickle)
    """ 
    # else :
    xml_pass = "./log_only2/xml/"
    mjlog_data = Mjlog()
    mjlog_data.read_mjlog_files(xml_pass) 
    situation_list = []

    mjlog_data.record_result()
    for contest_data in mjlog_data.contents_list:
        for all_match_data in contest_data.all_match_data:
            situation_list.append(all_match_data.situation)

    with open("pickle/situation.pickle", mode='wb') as situation_list_pickle:
        pickle.dump(situation_list, situation_list_pickle)

    with open("pickle/result.pickle", mode='wb') as result_pickle:
        pickle.dump(mjlog_data.result_data, result_pickle)


    with open(pickle_path, mode='wb') as mjlog_data_pickle:
        pickle.dump(mjlog_data, mjlog_data_pickle)

