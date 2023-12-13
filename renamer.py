import os

if __name__ == "__main__":
    with open("data/HWDB/test/zf_gnt_test.txt", "r", encoding='UTF-8') as txt:
        lines = [x.replace("\n", "").split(",") for x in txt]

        for line in lines :
            parts = line[0].rsplit("\t",maxsplit=1) # 1290-c/563.png	兼
            path = "data/HWDB/test/"+parts[0] # 1290-c/563.png
            [dir,fullname] = parts[0].rsplit("/",maxsplit=1) # 1290-c 563.png
            name = fullname.rsplit(".",maxsplit=1)[0] # 563
            if(os.path.exists(path)):
                os.rename(path, "data/HWDB/test/"+dir + "/" +name+"-"+ parts[1] + ".png")
