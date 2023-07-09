hai = ["一", "二", "三", "四", "五", "六", "七", "八", "九", #萬子
       "①", "②", "③", "④", "⑤", "⑥", "⑦", "⑧", "⑨", #筒子
       "1", "2", "3", "4", "5", "6", "7", "8", "9", #索子
      "東", "南", "西", "北", "白", "發", "中"]

sanma_hai = ["一", "九", #萬子
       "①", "②", "③", "④", "⑤", "⑥", "⑦", "⑧", "⑨", #筒子
       "1", "2", "3", "4", "5", "6", "7", "8", "9", #索子
      "東", "南", "西", "北", "白", "發", "中"]

tile_variety = { 
    "manzu": ["一", "二", "三", "四", "五", "六", "七", "八", "九"],
    "pinzu": ["①", "②", "③", "④", "⑤", "⑥", "⑦", "⑧", "⑨"],
    "souzu": ["1", "2", "3", "4", "5", "6", "7", "8", "9"],
    "zihai": ["東", "南", "西", "北", "白", "發", "中"],
    "kaze": ["東", "南", "西", "北"],
    "sangen": ["白", "發", "中"],
    "routou": ["一", "九", "①", "⑨", "1", "9"],
    "tyuntyan": ["②", "③", "④", "⑤", "⑥", "⑦", "⑧", "2", "3", "4", "5", "6", "7", "8"],
    "no_green": ["②", "③", "④", "⑤", "⑥", "⑦", "⑧", "3", "5", "7"],
    "green": ["2", "4", "6", "8"],
    "yaotyu":["一", "九", "①", "⑨", "1", "9", "東", "南", "西", "北", "白", "發", "中"]
}

DAN = [
    "新人","９級","８級","７級","６級","５級","４級","３級","２級","１級",
    "初段","二段","三段","四段","五段","六段","七段","八段","九段","十段",
    "天鳳","RESERVED..."
]

ROUND_NAME = [
    "東一局","東二局","東三局","東四局",
    "南一局","南二局","南三局","南四局",
    "西一局","西二局","西三局","西四局",
    "北一局","北二局","北三局","北四局",
    ]
MATCH_TYPE = [
    ["対コンピュータ戦", "対人戦" ],
    ["赤アリ", "赤ナシ"],
    ["喰アリ", "喰ナシ"],
    ["東風", "東南"],
    ["四麻", "三麻"],
    [None, "特上"],
    ["5+10秒", "速"],
    [None, "上級"],
    [None, "暗"],
    [None, "祝"],
    [None, "雀荘"],
    [None, "技能"]
]


YAKU=[
    # 一飜(21)
    "門前清自摸和","立直","一発","槍槓","嶺上開花",
    "海底摸月","河底撈魚","平和","断幺九","一盃口",
    "自風 東","自風 南","自風 西","自風 北",
    "場風 東","場風 南","場風 西","場風 北",
    "役牌 白","役牌 發","役牌 中",
    # 二飜(11)
    "両立直","七対子","混全帯幺九","一気通貫","三色同順",
    "三色同刻","三槓子","対々和","三暗刻","小三元","混老頭",
    # 三飜(3)
    "二盃口","純全帯幺九","混一色",
    # 六飜(1)
    "清一色",
    # 満貫(1)
    "人和",
    # 役満(15)
    "天和","地和","大三元","四暗刻","四暗刻単騎","字一色",
    "緑一色","清老頭","九蓮宝燈","純正九蓮宝燈","国士無双",
    "国士無双１３面","大四喜","小四喜","四槓子",
    # 懸賞役(3)
    "ドラ","裏ドラ","赤ドラ"
];

YAKUMAN = ["天和","地和","大三元","四暗刻","四暗刻単騎","字一色",
    "緑一色","清老頭","九蓮宝燈","純正九蓮宝燈","国士無双",
    "国士無双１３面","大四喜","小四喜","四槓子", "数え役満"
    ]
# YAKUMAN=["大三元", "字一色", "清老頭", "国士無双", "四喜和"]

HANDS_INDEX = 0
EXPOSED_MELD_INDEX = 1 
EXPOSED_NORTH_INDEX = 2
HAND_DISCARD_ID = 136 
REACH_ID = 137 
MELD_ID = 138 
NORTH_ID = 139
DISCARD_NUM_ID= 140