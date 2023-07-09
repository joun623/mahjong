import re
from urllib.parse import unquote
from data import hai, HANDS_INDEX, EXPOSED_MELD_INDEX, EXPOSED_NORTH_INDEX

def id2tile(tile_id, is_in_redtile=True):
    """
        tile_id = natural_id
        return 牌の名前
    """
    red_flag = False

    # 赤あり、赤なし(init_typeで見分けられる)
    if is_in_redtile:
        if tile_id in [16,52,88]:
            red_flag = True

    tile_name = hai[tile_id >> 2]

    if red_flag:
        tile_name = "r" + tile_name

    return tile_name

def ids2tiles(tile_id_list, is_in_redtile=True):
    return [id2tile(ids,is_in_redtile) for ids in tile_id_list]

def tilename2index(str_tiles):
    return [hai.index(tile) for tile in str_tiles]

def str2dct(dct_list):
    dct = {}
    split_space = dct_list.split(" ")
    for d in split_space:
        if "=" in d:
            split_equal = d.split("=")
            dct[split_equal[0]] = extract_in_doublequates(split_equal[1])

    return dct

def join_hands_rotated(hands):
    return [hand or rotated for hand, rotated in zip(hands[HANDS_INDEX], hands[EXPOSED_MELD_INDEX])]

def one_hot_id2tile(one_hot_id, is_in_redtile=True):
    for index, is_tile in enumerate(one_hot_id):
        if is_tile:
            return id2tile(index)

def one_hot_ids2tiles(one_hot_ids, is_in_redtile=True):
    tile_id_list = [index for index, is_tile in enumerate(one_hot_ids) if is_tile] 
    return ids2tiles(tile_id_list, is_in_redtile)

def one_hot_id2id(one_hot_id):
    for i, is_tile in enumerate(one_hot_id):
        if is_tile: 
            return i

def one_hot_ids2ids(one_hot_ids):
     return [index for index, is_tile in enumerate(one_hot_ids) if is_tile] 


def extract_in_doublequates(search_str):
    '''
       search_str = "hoge "0,0,0,2,2,112" hoge"↲
       return "0,0,0,2,2,112"
    '''
    # "hoge" -> hoge↲
    in_double_quates = re.findall('".+?"', search_str)

    # ""が存在しない場合は、空の文字列を返す
    if in_double_quates :
        exclusion_doublequates = in_double_quates[0].replace('"', '')
        # exclusion_doublequates = [int(str_elem) for str_elem in exclusion_doublequates.split(",")]↲
    else :
        exclusion_doublequates = "" 

    return exclusion_doublequates



def strs2ints(str_list):
    '''
        str_list = ["0", "0", "0", "2", "2", "112"]
        return [0, 0, 0, 2, 2, 112]
    '''

    if str_list == ['']:
        return []

    else: 
        return [int(s) for s in str_list]



def extract_doublequates_toIntlist(search_str):
    extract_doublequates_str = extract_in_doublequates(search_str)
    value_list = extract_doublequates_str.split(",")
    return strs2ints(value_list)

def url_decode(s_quate):
    return unquote(s_quate)

def difference_list(src_list, diff_list):
    return list(set(src_list) - set(diff_list))

def calc_bit(bit_list):
    bit_num = 0
    for i in range(len(bit_list)):
       bit_num += pow(2, i) * bit_list[i]

    return bit_num 
 
def int2bit(int_num, bit_length):
    return [bool(int_num & pow(2, i)) for i in range(bit_length)]

def next_dora(dora_id, is_sanma=True):
    if hai[dora_id] == "一" and is_sanma:
        return hai.index("九")

    elif hai[dora_id] == "九" :
        return hai.index("一")

    elif hai[dora_id] == "⑨" :
        return hai.index("①")

    elif hai[dora_id] == "9" :
        return hai.index("1")

    elif hai[dora_id] == "北" :
        return hai.index("東")

    elif hai[dora_id] == "中" :
        return hai.index("白")

    return dora_id + 1

def calc_tile_num(hands, tile_type):
    # ある牌(tile_type)の枚数を数える

    # tile_typeが牌の名前で入力された時は、idに変換
    if type(tile_type) is str:
       tile_id = hai.index(tile_type) 
    else :
       tile_id = tile_type

    tile_first_id = tile_id * 4
    one_tiles = hands[tile_first_id: tile_first_id + 4] 
    tile_num = 0

    for is_exist_tile in one_tiles:
            tile_num += is_exist_tile 

    return tile_num


def calc_tiles_num(hands, tile_types):
    # 複数牌の枚数を数える
    # tile_types = [tile_id(0~33)]
    # tile_types = ["中", "發"]
    sum_tiles = []

    for tile_type in tile_types:
        sum_tiles.append(calc_tile_num(hands, tile_type)) 

    return sum_tiles

def convert_sanma_id(tile_id):
    """
        牌のIDをヨンマ->サンマに
    """
    if tile_id == 0:
        return 0
    else :
        return tile_id - 7

def reduce_hai_sanma(hands):
    # 四麻 -> サンマ
    return hands[0:1] + hands[8:]

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

def get_now_discard(discard, num):
    now_discard = []
    for d in discard:
        if d[-1] < num:
            now_discard.append(d)
        else:
            break

    return now_discard

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


