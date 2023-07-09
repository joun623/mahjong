import tkinter as tk
from PIL import Image, ImageTk

# CLOCK = 'tiles.jpeg'
tile_dir = "./tile/"
tile_path = "-66-90-l.png"
reach_tile_path = "-66-90-l-yoko.png"

class Frame(tk.Frame):
    def __init__(self, tile_name, master=None):
        tk.Frame.__init__(self, master)

        image_path = tile_dir + tile_name + tile_path
        image = Image.open(image_path)

        self.img = ImageTk.PhotoImage(image)

        il = tk.Label(self, image=self.img)
        il.pack()


if __name__ == '__main__':
    tile_name = input()
    f = Frame(tile_name)
    f.pack()
    f.mainloop()

