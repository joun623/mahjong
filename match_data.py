import copy
from itertools import zip_longest
from util import id2tile, str2dct, one_hot_id2tile, one_hot_ids2tiles, extract_in_doublequates, extract_doublequates_toIntlist, calc_bit, calc_tiles_num, one_hot_ids2ids, next_dora, tilename2index, one_hot_id2id
from judge_tenpai import is_tenpai_thirteen_orphans, is_tenpai_all_honors, is_tenpai_all_terminals, is_tenpai_four_wind, is_tenpai_three_big_dragons 
from data import hai, HANDS_INDEX, EXPOSED_MELD_INDEX, EXPOSED_NORTH_INDEX,   YAKUMAN, ROUND_NAME, YAKU, HAND_DISCARD_ID, REACH_ID, MELD_ID, NORTH_ID,DISCARD_NUM_ID 
from meld import get_meld_data


class MatchData:
    ''' 
        １局の対局データを保存
    '''
    def __init__(self, match_elements, user, file_name):
        self.init_dct = {}  # <INIT ~>
        self.dealt_tiles = [] # 配牌
        self.hands = [] # 進行手牌
        self.situation = []
        self.discards = [[], [], [], []]  # 捨て牌 
        self.user = user # userの名前
        self.reach_flag = [False for _ in range(len(user))] # リーチしているかどうか
        self.result = {} # あがりか流局か
        self.result["win_data"] = []
        self.result["tsumo_num"] = 0
        self.result["discard_num"] = 0
        self.result["haihu_id"] = file_name
        self.result["lag"] = [[False for _ in range(34)], [[False, False, False] for _ in range(34)]]
        self.flag = []
        self.result["is_disconnect_end"] = False

        for _ in range(3):
            self.flag.append({"is_reach": False, "is_meld":False, "is_north": False, "tsumo_id":1000})
        init_data = match_elements[0]
        discards_data = match_elements[1:]

        self.read_init_data(init_data)
        self.read_discards_data(discards_data)


    def read_init_data(self, init_data):
        # <INIT ~>の部分を読み込む
        # init_data = 'INIT seed=~~ ten=~~ oya=~~ ... '>
        split_space = init_data.split(" ")

        for elem in split_space:
            if elem.startswith("INIT"):
                continue

            elem_name = elem.split("=")[0]
            in_double_quates = extract_doublequates_toIntlist(elem)

            self.init_dct[elem_name] = in_double_quates

        seed_data = self.init_dct["seed"]
        self.init_dct["round_name"] = seed_data[0]
        self.init_dct["honba"] = seed_data[1]
        self.init_dct["kyotaku"] = seed_data[2]
        self.init_dct["pip"] = seed_data[3:5]
        self.init_dct["dora_tile"] = seed_data[5]
        self.init_dct["point"] = [point * 100 for point in self.init_dct["ten"]]

        # self.echo_init_data()
        self.read_dealt_tiles()
        
    def echo_init_data(self):
        """
            初期データの表示
        """
        print(str(self.init_dct["round_name"]) + ", " + str(self.init_dct["honba"]) + "本場, 供託" + str(self.init_dct["kyotaku"]))

        for u, point in zip(self.user, self.init_dct["point"]):
            print (u["name"] + ": " + str(point) + "点", end=" ")

        print("")
        print("ドラ表示牌：" + id2tile(self.init_dct["dora_tile"]))

    def read_discards_data(self, discards_data):
        '''
            捨て牌を読み込む
        '''
        for discard_id in discards_data:
            if "=" in discard_id:
                if discard_id.startswith("N"):
                    self.call_meld(discard_id)
                elif discard_id.startswith("REACH"):
                    self.call_reach(discard_id)
                elif discard_id.startswith("AGARI"):
                    self.call_winning(discard_id)
                elif discard_id.startswith("RYUUKYOKU"):
                    self.call_draw(discard_id)
                else:
                    print(discard_id)
            else:
                self.advance_tile(discard_id) 

        if not  (discards_data[-1].startswith("RYUUKYOKU") or  discards_data[-1].startswith("AGARI")):

            self.result["is_disconnect_end"] = True 

    def read_dealt_tiles(self):
        '''
            配牌を読み込む
        '''
        for i in range(4):
            tmp_dealt_tiles = [False for _ in range(136)]
            dealt_tiles_list = self.init_dct["hai" + str(i)]

            for tile_id in dealt_tiles_list:
                tmp_dealt_tiles[tile_id] = True

            self.dealt_tiles.append(copy.deepcopy(tmp_dealt_tiles))
            self.hands.append([copy.deepcopy(tmp_dealt_tiles), [False for _ in range(136)], [False for _ in range(136)]])

    def advance_tile(self,discard_id):
        '''
            ツモ、打牌の読み込み
            discard_id = "T123/"
            T : mark
            123 : tile_id
        '''
        who_id = {"T": 0, "U": 1, "V": 2, "W": 3,
                  "D": 0, "E": 1, "F": 2, "G": 3}

        tsumo_mark = ["T", "U", "V", "W"]
        dahai_mark = ["D", "E", "F", "G"]

        mark = discard_id[0] 
        tile_id = int(discard_id[1:-1])
        
        if mark in tsumo_mark:
            self.tsumo(tile_id, who_id[mark])
        elif mark in dahai_mark:
            self.discard(tile_id, who_id[mark])


    def tsumo(self, discard_id, who_id):
        '''
            ツモ読み込み
        '''
        self.hands[who_id][HANDS_INDEX][discard_id] = True
        self.result["tsumo_num"] += 1
        self.flag[who_id]["tsumo_id"] = discard_id

    
    def register_result(self, discard_id, who_id):
        situation_dct = {}
        situation_dct["hand"] = []
        for u in self.hands:
            u_hand = []
            for h in u:
                u_hand.append(calc_tiles_num(h,hai))
            situation_dct["hand"].append(copy.deepcopy(u_hand))

        """
        situation_dct["discard"] = []
        for u in self.discards:
            tmp_discard = [False for _ in range(136)]
            for one_discard in u[:136]:
                tmp_discard = [t or o for t, o in zip(copy.deepcopy(tmp_discard), one_discard)]
            situation_dct["discard"].append(calc_tiles_num(tmp_discard, hai) )
        """
        situation_dct["who_id"] = who_id
        situation_dct["round_id"] = self.init_dct["round_name"]
        situation_dct["name"] = self.user
        situation_dct["dora"] = next_dora(self.init_dct["dora_tile"] >> 2)
        situation_dct["tsumo_num"] = self.result["tsumo_num"]
        situation_dct["select_id"] = discard_id
        # if "reach_flag" not in situation_dct:
        #    situation_dct["reach_flag"] = [False for _ in range(3)]
        situation_dct["reach_flag"] = self.reach_flag
        situation_dct["lag"] = copy.deepcopy(self.result["lag"])
        situation_dct["discard"] = self.discards[:]
        situation_dct["discard_num"] = self.result["discard_num"]
        situation_dct["haihu_id"] = self.result["haihu_id"]
        situation_dct["tsumo_id"] = self.flag[who_id]["tsumo_id"]
        self.situation.append(situation_dct)

    def call_meld(self, discard_id):
        '''
            鳴きの処理
        '''
        meld_data = get_meld_data(discard_id)
        tile_sequence = meld_data["meld_tile_id"] % 4
        first_tile_id = meld_data["meld_tile_id"] - tile_sequence

        for i, is_unused in enumerate(meld_data["unused_tile"]):
            if not is_unused:
                self.hands[meld_data["from_who_id"]][HANDS_INDEX][first_tile_id + i] = False

                self.hands[meld_data["from_who_id"]][EXPOSED_MELD_INDEX + int(meld_data["is_north"])][first_tile_id + i] = True

        if meld_data["who_discard_id"] != meld_data["from_who_id"]:
            # ポンや明カンで他者からハイをとった場合
            self.discards[meld_data["who_discard_id"]][-1][MELD_ID] = True
        elif meld_data["is_north"]:
             # 北抜きの場合
            self.flag[meld_data["from_who_id"]]["is_north"] = True
        else:
            # 暗カン等で鳴きが発生した場合
            self.flag[meld_data["from_who_id"]]["is_meld"] = True

        # self.echo_hands(self.hands, meld_data["from_who_id"])

    def call_reach(self, discard_id):
        split_space = discard_id.split(" ")
        who_reach = int(extract_in_doublequates(split_space[1]))
        self.reach_flag[who_reach] = True
        self.flag[who_reach]["is_reach"] = True


    def call_winning(self, discard_id):
        win_data = {}
        discard_dct = str2dct(discard_id)
        who_id = int(discard_dct["who"])
        from_who_id = int(discard_dct["fromWho"])
        ten_type = int(discard_dct["ten"].split(",")[2])

        if ten_type == 5:
            # print ("和了者=" + self.user[who_id]["name"])
            # print ("放銃者=" + self.user[from_who_id]["name"])
            is_win = True

            if "yakuman" in discard_dct:
                yaku_list = [int(y) for y in discard_dct["yakuman"].split(",")]
            else:
                yaku_list = [int(y) for y in discard_dct["yaku"].split(',')] + [len(YAKU)]
        else :
            is_win = False 
            yaku_list = [int(y) for y in discard_dct["yaku"].split(',')]

        
        win_data["is_win"] = is_win
        win_data["who_id"] = who_id
        win_data["from_who_id"] = from_who_id
        win_data["yaku_list"] = yaku_list
        self.result["win_data"].append(win_data)

    def call_draw(self, discard_id):
        self.result["win_data"].append({"is_win": False} )

    def discard(self, discard_id, who_id):
        '''
            打牌読み込み
        '''
        self.register_result(discard_id, who_id)
        self.hands[who_id][HANDS_INDEX][discard_id] = False 
        tmp_discard_tile = [False for _ in range(136 + 5)]
        tmp_discard_tile[discard_id] = True
        tmp_discard_tile[REACH_ID] = self.flag[who_id]["is_reach"]
        tmp_discard_tile[MELD_ID] = self.flag[who_id]["is_meld"]
        tmp_discard_tile[NORTH_ID] = self.flag[who_id]["is_north"]
        tmp_discard_tile[HAND_DISCARD_ID] = self.flag[who_id]["tsumo_id"] == discard_id
        tmp_discard_tile[DISCARD_NUM_ID] = self.result["discard_num"]

        self.discards[who_id].append(tmp_discard_tile)

        self.flag[who_id]["is_reach"] = False
        self.flag[who_id]["is_meld"] = False
        self.flag[who_id]["is_north"] = False
        self.flag[who_id]["tsumo_id"] = 1000 
        self.result["discard_num"] += 1
        # ラグの有無
        other_player = [0, 1, 2]
        other_player = list(set(other_player) - set([who_id]))
        for i in other_player:
            if calc_tiles_num(self.hands[i][HANDS_INDEX], hai)[discard_id >> 2] >= 2:
                self.result["lag"][0][discard_id >> 2] = True
                self.result["lag"][1][discard_id >> 2][i] = True
        # self.echo_hands(self.hands, who_id)

    def show_tile_advance(self):
        pass
        
    def echo_hands(self, hands, who_id):
        print("", end="[")
        print(" ".join(one_hot_ids2tiles(self.hands[who_id][HANDS_INDEX])), end="] [")
        print(" ".join(one_hot_ids2tiles(self.hands[who_id][EXPOSED_MELD_INDEX])), end="] [ ")
        print(" ".join(one_hot_ids2tiles(self.hands[who_id][EXPOSED_NORTH_INDEX])), end="] -> ")
        print(is_tenpai_thirteen_orphans(self.hands[who_id]))

    def echo_board(self):
        print("vvvvvvvvvvvvvvvvvvvvvvvvvvvvv")
        
        for i in range(len(self.user)):
            user_name = self.user[i]["name"]
            if self.reach_flag[i]:
                user_name = user_name + ": リーチ中"
            print(self.user[i]["name"])

            is_tenpais = [
                is_tenpai_three_big_dragons(self.hands[i]),
                is_tenpai_all_honors(self.hands[i]),
                is_tenpai_all_terminals(self.hands[i]),
                is_tenpai_thirteen_orphans(self.hands[i]),
                is_tenpai_four_wind(self.hands[i])
            ]
            if any(is_tenpais):
                print("!!! テンパイ: ",end="")
                for yaku in [YAKUMAN[index] for index, is_tenpai in enumerate(is_tenpais) if is_tenpai]:
                    print(yaku,end=", ")
                print(" !!!")

            print ("---手牌---")
            print(one_hot_ids2tiles(self.hands[i][HANDS_INDEX]),end="")
            print(one_hot_ids2tiles(self.hands[i][EXPOSED_MELD_INDEX]),end="")
            print(one_hot_ids2tiles(self.hands[i][EXPOSED_NORTH_INDEX]))

            print ("---捨牌---")
            for discard_row in self.split_six(self.discards[i]):
                print([one_hot_id2tile(d) for d in discard_row])
            print()
        print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        
    def split_six(self, list_discard):
        split_list = []
        row = int(len(list_discard) / 6)

        for i in range(row):
            split_list.append(list_discard[i: i + 6])

        if len(list_discard) % 6 != 0:
            split_list.append(list_discard[row * 6:])

        return split_list


