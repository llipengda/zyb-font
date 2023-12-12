import os

import gui

src_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.dirname(src_path)
os.chdir(root_path)

if __name__ == '__main__':
    gui.paint.run()
