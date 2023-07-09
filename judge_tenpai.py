from data import hai, tile_variety 
from util import join_hands_rotated, ids2tiles, tilename2index, difference_list, calc_tiles_num 


def is_tenpai_thirteen_orphans(hands):
    menber = [] 
    menber.extend(tile_variety["routou"])
    menber.extend(tile_variety["zihai"])

    menber_num = calc_tiles_num(join_hands_rotated(hands), menber)
    is_over_single = [n >= 1 for n in menber_num] 

    # print(menber_num)
    if all(is_over_single):
        return True  # 13面待ち
    elif sum(is_over_single) == 12 and any([n == 2 for n in menber_num]):
        return True

    return False

    
def judge_tenpai_add_pon(hands, tile_types, pon_num=4):
    '''
        ポン込みでテンパイしているかどうか(単騎候補牌も含めて)
        tile_types: ポン材の種類
        pon_num: テンパイに必要なポン材の数
    '''
    is_tenpai = 0 
    tile_nums = calc_tiles_num(join_hands_rotated(hands), tile_types)

    num_over_pair = [num >= 2 for num in tile_nums].count(True)  # 対子以上の数
    num_single = [num == 1 for num in tile_nums].count(True)  # 単騎の数

    if num_over_pair > pon_num:
        is_tenpai = 2 
    elif num_over_pair == pon_num and num_single >= 1:
        is_tenpai = 1 

    return is_tenpai

def is_tenpai_all_honors(hands):
    # 字一色のテンパイ判定
    relate_type_name = ["東", "南", "西", "北", "白", "發", "中"]
    return bool(judge_tenpai_add_pon(hands, tilename2index(relate_type_name)))

def is_tenpai_all_terminals(hands):
    # 清老頭のテンパイ判定
    relate_type_name = ["一", "九", "①", "⑨", "1", "9"]
    return bool(judge_tenpai_add_pon(hands, tilename2index(relate_type_name)))

def is_tenpai_four_wind(hands):
    # 四喜和のテンパイ判定
    relate_type_name = ["東", "南", "西", "北"]
    is_tenpai = judge_tenpai_add_pon(hands, tilename2index(relate_type_name), 3)

    if is_tenpai == 1: 
        if not is_setorpair_medium(join_hands_rotated(hands), "kaze"):
            is_tenpai = 0

    return bool(is_tenpai)

def is_tenpai_three_big_dragons(hands):
    # 大三元のテンパイ判定

    doragons_num = calc_tiles_num(join_hands_rotated(hands), tile_variety["sangen"])
    return  all([dn >= 2 for dn in doragons_num]) and \
        is_setorpair_medium(join_hands_rotated(hands), "sangen")


def is_setorpair_medium(hands, exclusion_type):
    # １メンツ、あるいは１対子があるかどうか判定
    # exclusion_type = 除外するメンツ("kaze" or "sangen" or ...)
    target_tile = difference_list(hai, tile_variety[exclusion_type])
    target_tile_ids = tilename2index(target_tile) 

    if any([num >= 2 for num in calc_tiles_num(hands, target_tile_ids)]):
        return True
    elif is_consecutive_tile(hands):
        return True

    return False
    
def is_consecutive_tile(hands, is_yonma=False):
    # 順子が一つでもあるかどうか

    consective_num = 3

    simples = ["pinzu", "souzu"]
    if is_yonma:
        simples.append("manzu")

    for smp in simples: 
        ids = tilename2index(tile_variety[smp])
        is_exist_tiles = [num > 0 for num in calc_tiles_num(hands, ids) ] 
        for i in range(len(is_exist_tiles) - consective_num): 
            if all(is_exist_tiles[i: i + consective_num]):
                return True

    return False

