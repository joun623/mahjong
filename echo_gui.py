import tkinter as tk
import sys
from PIL import Image, ImageTk, ImageOps
from util import hai_num_list2tilename_list, get_now_discard, get_discard_list
from data import HANDS_INDEX, EXPOSED_MELD_INDEX, EXPOSED_NORTH_INDEX, hai, sanma_hai, HAND_DISCARD_ID, MELD_ID

tile_dir = "./tile/"
tile_path = "-66-90-s.png"
reach_tile_path = "-66-90-3-yoko.png"
MY_TEHAI_X = 300
MY_TEHAI_Y = 800
MY_DISCARD_X = 400
MY_DISCARD_Y = 600
HAI_HEIGHT = 50
HAI_WIDTH = 30

hai2file_name={
    "一": "man1","二": "man2", "三": "man3","四": "man4","五": "man5","六": "man6","七": "man7","八": "man8","九": "man9",
    "①": "pin1","②": "pin2","③": "pin3","④": "pin4","⑤": "pin5","⑥": "pin6","⑦": "pin7","⑧": "pin8","⑨": "pin9",
    "1": "sou1","2": "sou2","3": "sou3","4": "sou4","5": "sou5","6": "sou6","7": "sou7","8": "sou8","9": "sou9",
    "東": "ji1","南": "ji2","西": "ji3","北": "ji4","白": "ji6","發": "ji5","中":"ji7",
    "r⑤": "aka1", "r5": "aka2", "r五": "aka3"
    }

global img
class displayImage(tk.Frame):
    def __init__(self, situations, vals, args, master=None):
        super().__init__(master)
        self.pack()
        self.situations = situations
        self.vals = vals
        self.args = args
        self.num = 0

        self.init_canvas() 
        self.refresh()
        # self.create_widgets()

    def refresh(self):
        self.situation = self.situations[self.num]
        self.val = self.vals[self.num]
        self.arg = self.args[self.num]
        global img 
        img = []
        self.create_situation()
        self.num += 1

    def remove_refresh(self):
        self.num -= 2 
        self.refresh()

    def callback(self, event):
        self.refresh()
        self.update()

    def right_callback(self, event):
        self.remove_refresh()
        self.update()


    def create_situation(self):
        
        discard_path = []

        self.discard = []
        # 手牌表示

        self.create_tehai()
        self.create_meld()
        self.create_all_discard() 
        self.create_candidate()
        self.create_central()
        self.create_header()
        self.canvas.bind("<Button-1>", self.callback)
        self.canvas.bind("<Button-2>", self.right_callback)
        self.canvas.pack()
        # self.button = tk.Button(self, text="表示", relief="groove", bg="white", height=2, width=5)
        # self.button["command"] = self.callback
        # self.button.place(x=5,y=20)
        # self.button.pack(anchor="ne")

    def init_canvas(self):
        # canvasの初期化    
        self.canvas = tk.Canvas(self, bg="green", width=1000, height=1000)

    def create_hai_canvas(self, hai_list, x, y, dx, dy, rotate_angle=0, is_tsumo_discards=[], is_meld_discards=[]):
        if not len(is_tsumo_discards):
            is_tsumo_discards = [False for _ in range(len(hai_list))]
        if not len(is_meld_discards):
            is_meld_discards = [False for _ in range(len(hai_list))]

        global img
        for i, name, is_tsumo, is_meld in zip(range(len(hai_list)), hai_list, is_tsumo_discards, is_meld_discards):
            path = tile_dir + hai2file_name[name] + tile_path
            im = Image.open(path)# .rotate(rotate_angle)
            if rotate_angle == 90:
                im = im.transpose(Image.ROTATE_90) 
            elif rotate_angle == 180: 
                im = im.transpose(Image.ROTATE_180) 
            elif rotate_angle == 270:
                im = im.transpose(Image.ROTATE_270)
            im = im.convert("RGB")

            if is_meld:
                im = ImageOps.invert(im)
            if is_tsumo:
                im = im.point(lambda x: x * 0.5)
                # im = im.convert("L")
            img.append(ImageTk.PhotoImage(im))
            self.canvas.create_image(x + dx * i, y + dy * i ,image=img[-1])

    def create_tehai(self):
        # 手牌表示
        hero_hand_list = hai_num_list2tilename_list(self.situation["hand"][self.situation["who_id"]][HANDS_INDEX])
        if self.situation["tsumo_id"] != 1000:
            hero_hand_list.remove(hai[self.situation["tsumo_id"] >> 2])

        opponent1_hand_list = hai_num_list2tilename_list(self.situation["hand"][(self.situation["who_id"] + 1) % 3][HANDS_INDEX])
        opponent2_hand_list = hai_num_list2tilename_list(self.situation["hand"][(self.situation["who_id"] + 2) % 3][HANDS_INDEX])


        self.create_hai_canvas(hero_hand_list, MY_TEHAI_X, MY_TEHAI_Y, HAI_WIDTH, 0)
        self.create_hai_canvas(opponent1_hand_list, MY_TEHAI_Y, 1000 - MY_TEHAI_X, 0, -HAI_WIDTH, 90)
        self.create_hai_canvas(opponent2_hand_list, 1000 - MY_TEHAI_X, 1000 - MY_TEHAI_Y, -HAI_WIDTH, 0, 180)

        if self.situation["tsumo_id"] != 1000:
            self.create_hai_canvas([hai[self.situation["tsumo_id"] >> 2]], MY_TEHAI_X + len(hero_hand_list) * HAI_WIDTH + 10, MY_TEHAI_Y, HAI_WIDTH, 0)

    def create_meld(self):
        # 手牌表示

        hero_hand_list = hai_num_list2tilename_list(self.situation["hand"][self.situation["who_id"]][EXPOSED_MELD_INDEX])
        opponent1_hand_list = hai_num_list2tilename_list(self.situation["hand"][(self.situation["who_id"] + 1) % 3][EXPOSED_MELD_INDEX])
        opponent2_hand_list = hai_num_list2tilename_list(self.situation["hand"][(self.situation["who_id"] + 2) % 3][EXPOSED_MELD_INDEX])

        self.create_hai_canvas(hero_hand_list, MY_TEHAI_X + 350, MY_TEHAI_Y, HAI_WIDTH, 0)
        self.create_hai_canvas(opponent1_hand_list, MY_TEHAI_Y , 650 - MY_TEHAI_X, 0, -HAI_WIDTH, 90)
        self.create_hai_canvas(opponent2_hand_list, 650 - MY_TEHAI_X, 1000 - MY_TEHAI_Y, -HAI_WIDTH, 0, 180)


    def create_all_discard(self):
        # 捨て牌表示
        hero_discard_list = get_now_discard(self.situation["discard"][self.situation["who_id"]], self.situation["discard_num"])
        
        oppnent1_discard_list = get_now_discard(self.situation["discard"][(self.situation["who_id"] + 1) % 3], self.situation["discard_num"])

        oppnent2_discard_list = get_now_discard(self.situation["discard"][(self.situation["who_id"] + 2) % 3], self.situation["discard_num"])

        self.create_discard(hero_discard_list, MY_DISCARD_X, MY_DISCARD_Y, (HAI_WIDTH,0), (0, HAI_HEIGHT), 0)
        self.create_discard(oppnent1_discard_list, MY_DISCARD_Y, 1000- MY_DISCARD_X, (0, -HAI_WIDTH), (HAI_HEIGHT, 0), 90)
        self.create_discard(oppnent2_discard_list, 1000 - MY_DISCARD_X, 1000 - MY_DISCARD_Y, (-HAI_WIDTH, 0), (0, -HAI_HEIGHT), 180)

    def create_discard(self, discard_data, x, y, dx_tuple, dy_tuple, rotate_angle):
        discard_list = get_discard_list(discard_data)
        is_tsumo_discards = [dd[HAND_DISCARD_ID] for dd in discard_data ]
        is_meld_discards = [dd[MELD_ID] for dd in discard_data ]
        discard_length = len(discard_list)
        row_num = int(len(discard_list) / 6)
        for i in range(row_num):
            self.create_hai_canvas(discard_list[i * 6: (i + 1) * 6], x + i * dy_tuple[0], y + i * dy_tuple[1] , dx_tuple[0], dx_tuple[1], rotate_angle, is_tsumo_discards[i * 6: (i + 1) * 6], is_meld_discards[i * 6: (i + 1) * 6])

        if discard_length % 6:
            self.create_hai_canvas(discard_list[-(discard_length % 6):], x + row_num * dy_tuple[0], y + row_num * dy_tuple[1], dx_tuple[0], dx_tuple[1], rotate_angle, is_tsumo_discards[-(discard_length % 6):], is_meld_discards[-(discard_length % 6):])

    def create_candidate(self):
        self.create_hai_canvas([sanma_hai[a] for a in self.arg[0:3]], MY_TEHAI_X - 200, MY_TEHAI_Y - 100, HAI_WIDTH, 0)

        self.create_hai_canvas([hai[self.situation["select_id"] >> 2]], MY_TEHAI_X - 180, MY_TEHAI_Y , HAI_WIDTH, 0)

        self.label = tk.Label(text=", ".join([str(int(v * 1000) / 10) + "%" for v in self.val[0:3]]))
        self.label.place(x = 70, y = 730)
        self.label_correct = tk.Label(text="correct")
        self.label_correct.place(x = 50, y = 800)

    def create_central(self):
        house_name = ["東家","南家","西家"]
        self.label_correct = tk.Label(text="dora")
        self.label_correct.place(x = 460, y = 470)

        self.create_hai_canvas([hai[self.situation["dora"]]], 480, 510, 0, 0, 0)

        self.label_correct = tk.Label(text="残:" + str(55 - self.situation["tsumo_num"]))
        self.label_correct.place(x = 400, y = 430)

        self.label_correct = tk.Label(text=house_name[self.situation["who_id"] % 3])
        self.label_correct.place(x = 480, y = 550)
        self.label_correct = tk.Label(text=house_name[(self.situation["who_id"]  + 1) % 3])
        self.label_correct.place(x = 530, y = 480)
        self.label_correct = tk.Label(text=house_name[(self.situation["who_id"] + 2) % 3])
        self.label_correct.place(x = 480, y = 430)

    def create_header(self):
        self.EditBox = tk.Entry(text=self.situation["haihu_id"])
        self.EditBox.delete(0, tk.END)
        self.EditBox.insert(tk.END, "http://tenhou.net/0/?log=" + self.situation["haihu_id"])
        self.EditBox.place(x = 0, y = 0)
        self.label = tk.Label(text="lag: ")
        self.label.place(x = 0, y = 50)
        lag_list = [is_lag and (not who_have[self.situation["who_id"]]) for is_lag, who_have in zip(self.situation["lag"][0], self.situation["lag"][1])]
        self.create_hai_canvas([hai[i] for i, lag in enumerate(lag_list) if lag], 50, 50, HAI_WIDTH, 0)

def display(situation, vals, args):
    root = tk.Tk()
    d = displayImage(situation, vals, args, master=root) 
    d.mainloop()


if __name__ == '__main__':
    root = tk.Tk()
    d = displayImage(master=root) 
    d.mainloop()
