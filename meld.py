import math
from data import hai
from util import id2tile, extract_in_doublequates, calc_bit, int2bit

def get_meld_data(discard_id):
    # 鳴きの動作
    # discard_id = "N who=0 m=12141"

    meld_data = {}
    split_space = discard_id.split(" ")
    meld_data["from_who_id"] = int(extract_in_doublequates(split_space[1]))
    meld_data["is_north"] = False
    meld_value = int(extract_in_doublequates(split_space[2]))
    bit_meld_value = int2bit(meld_value, 16)

    if bit_meld_value[2]: 
        print("chow")

    elif bit_meld_value[3]:
        meld_data.update(pong_extendpon(bit_meld_value, False))

    elif bit_meld_value[4]:
        meld_data.update(pong_extendpon(bit_meld_value, True))

    elif bit_meld_value[5]:
        meld_data.update(meld_north(bit_meld_value))

    else:
        meld_data.update(kong(bit_meld_value))


    meld_data["who_discard_id"] = (meld_data["from_who_id"] + meld_data["who_discard_id"]) % 4 


    return meld_data

def pong_extendpon(bit_meld_value, is_extendpon):
    meld_data = {}
    who_discard_id = calc_bit(bit_meld_value[0:2])
    meld_data["who_discard_id"] = who_discard_id

    unused_tile = calc_bit(bit_meld_value[5:7])
    unused_tile_list = [False for _ in range(4)]
    unused_tile_list[unused_tile] = True
    meld_data["unused_tile"] = unused_tile_list

    meld_tile_data = calc_bit(bit_meld_value[9: 16])

    meld_tile_id = math.floor(meld_tile_data / 3)  # 牌の種類
    discard_tile_id  = meld_tile_data % 3  # 同種4牌のうちどの牌を鳴いたか
    if unused_tile <= discard_tile_id:
        discard_tile_id += 1

    meld_data["meld_tile_id"] = meld_tile_id * 4 + discard_tile_id 

    if is_extendpon:
        meld_data["unused_tile"][unused_tile] = False

    return meld_data

def meld_north(bit_meld_value):
    meld_data = {}
    tile_id = calc_bit(bit_meld_value[8:16])
    meld_data["who_discard_id"] = 0  # 自分
    unused_list = [True for _ in range(4)]
    unused_list[tile_id % 4] = False
    meld_data["unused_tile"] = unused_list
    meld_data["meld_tile_id"] = tile_id
    meld_data["is_north"] = True

    return meld_data

def kong(bit_meld_value):
    meld_data = {}
    tile_id = calc_bit(bit_meld_value[8:16])
    meld_data["who_discard_id"] = calc_bit(bit_meld_value[0:2])
    unused_list = [False for _ in range(4)]
    meld_data["unused_tile"] = unused_list
    meld_data["meld_tile_id"] = tile_id

    return meld_data
