import os

import deeplearning


if __name__ == "__main__":
    if not os.path.exists("out/model.pth"):
        deeplearning.train_and_test()
    
    predict = deeplearning.Pridict()
    for file in os.listdir("draw"):
        res = predict(f"draw/{file}")
        print(f"{file} is {res}")

# import gui

# src_path = os.path.dirname(os.path.abspath(__file__))
# root_path = os.path.dirname(src_path)
# os.chdir(root_path)

# if __name__ == '__main__':
#     gui.paint.run()
