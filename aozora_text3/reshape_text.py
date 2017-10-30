"""
ファイルの中身を再形成
不要なかっこや改行を除く
"""


import re
import os
import sys


class File():
    def __init__(self):
        self.getlines = []
        self.filelist = []


    def get_files_indir(self):
        dir_name = "./"
        files = os.listdir(dir_name)
        try:
            for file in files:
                if (".txt" in file) and ("re_" in file):
                    self.filelist.append(file)

        except :
            print("reshape_text error occard" + file)
            sys.exit(0)


    def checkline(self,line):
        flag = 0
        # 半角
        if (("-" in line ) == True): flag += 1
        if (("[" in line ) == True): flag += 1
        if (("]" in line ) == True): flag += 1
        if (("(" in line ) == True): flag += 1
        if ((")" in line ) == True): flag += 1

        # 全角
        if (("【" in line ) == True): flag += 1
        if (("】" in line ) == True): flag += 1
        if (("［" in line ) == True): flag += 1
        if (("］" in line ) == True): flag += 1
        # if (("》" in line ) == True): flag += 1
        # if (("《" in line ) == True): flag += 1
        if (("（" in line ) == True): flag += 1
        if (("）" in line ) == True): flag += 1
        if (("｜" in line ) == True): flag += 1
        if (("ルビ" in line ) == True): flag += 1



        # 改行のみも除く
        if ((" \n" == line ) == True): flag += 1
        if (("\n" == line ) == True): flag += 1


        if (("＊" == line ) == True): flag += 1
        return flag


    def delword(self,line):
        line = re.sub(r'.《.*.》', "", line)
        line = re.sub(r'.《.*.》', "", line)
        line = re.sub(r'\u3000', "", line)
        return line


    def readfile(self,fname):
        project_dir = os.path.dirname(os.path.abspath(__file__))
        self.getlines = []
        with open(project_dir + '/' + fname,'r') as file:
            print("open "+fname)
            line = file.readline()
            while line:
                line = file.readline()
                if ("底本" in line) == True: break
                if (self.checkline(line) == 0) :
                    result = self.delword(line)
                    self.getlines.append(result)



    def writefile(self,fname):
        project_dir = os.path.dirname(os.path.abspath(__file__))
        save_dir = "/"
        fname = "re_"+fname
        with open(project_dir + save_dir + fname,'w') as file:
            print("save "+ project_dir+save_dir+fname)
            for line in self.getlines:
                file.writelines(line)


def test():
    st="あいうえ《かきくけこ》あああ\u3000"
    result = re.sub(r'.《.*.》', "", st)
    result = re.sub(r'\u3000', "", result)
    print(result)

    # new_st = st.replace('《*》','')
    # print(new_st)

def main():
    myfile = File()
    myfile.get_files_indir()

    for fname in myfile.filelist :
        myfile.readfile(fname)
        myfile.writefile(fname)

if __name__ == "__main__" :
    main()