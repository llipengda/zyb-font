import os

import deeplearning


if __name__ == "__main__":
    if not os.path.exists("out/model.pth"):
        deeplearning.train_and_test()
    
    predict = deeplearning.Pridict()
    for file in os.listdir("draw"):
        res = predict(f"draw/{file}")
        print(f"{file} is {res}")
