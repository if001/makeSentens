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
        
        # if (("[" in line ) == True): flag += 1
        # if (("]" in line ) == True): flag += 1
        # if (("(" in line ) == True): flag += 1
        # if ((")" in line ) == True): flag += 1

        # 全角
        if ((" 〕" in line ) == True): flag += 1
        if ((" 〔" in line ) == True): flag += 1
        if (("【" in line ) == True): flag += 1
        if (("】" in line ) == True): flag += 1 
        if (("［" in line ) == True): flag += 1
        if (("］" in line ) == True): flag += 1
        # if (("》" in line ) == True): flag += 1
        # if (("《" in line ) == True): flag += 1
        # if (("（" in line ) == True): flag += 1
        # if (("）" in line ) == True): flag += 1
        if (("｜" in line ) == True): flag += 1
        if (("ルビ" in line ) == True): flag += 1
        if (("（例）" in line ) == True): flag += 1
        if (("（ 例 ）" in line ) == True): flag += 1
        if (("［ ＃］" in line ) == True): flag += 1
        if ("底本" in line) == True: flag += 1
        # 改行のみも除く
        if ((" \n" == line ) == True): flag += 1
        if (("\n" == line ) == True): flag += 1

        # その他
        if (("＊" in line ) == True): flag += 1
        if (("http:" in line ) == True): flag += 1
        if (("青空文庫" in line ) == True): flag += 1
        if (("入力 ：" in line ) ): flag += 1
        if (("校正 ：" in line ) ): flag += 1
        if (("公開" in line ) and ("年" in line ) and ("月" in line) ): flag += 1
        if (("修正" in line ) and ("年" in line ) and ("月" in line) ): flag += 1
        if (("発行" in line ) and ("年" in line ) and ("月" in line) ): flag += 1
        
        return flag

    
    def rm_between(self,line):
        line = re.sub(r'.《.+?.》', "", line)
        # line = re.sub(r'.(.+?.)', "", line)
        line = re.sub(r'.（.+?.）', "", line)
        line = re.sub(r'.［.+?.］', "", line)
        line = re.sub(r'.[.+?.]', "", line)

        return line

    def delword(self,line):
        line = re.sub(r'【', "", line)
        line = re.sub(r'】', "", line)
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
                if (self.checkline(line) == 0) :
                    line = self.rm_between(line)
                    line = self.delword(line)
                    self.getlines.append(line)



    def writefile(self,fname):
        project_dir = os.path.dirname(os.path.abspath(__file__))
        save_dir = "/"
        fname = "re_"+fname
        with open(project_dir + save_dir + fname,'w') as file:
            print("save "+ project_dir+save_dir+fname)
            for line in self.getlines:
                file.writelines(line)


def test():
    f = File()
    st="ところが 、 その 予想 が がらっと 外れ 、 意外 や 、 題 を 聴け ば 「 水棲 人 」 。 私 も 、 ちょっと 暫 《 しば ら 》 く は 聴き ちがい で は ない か と 思っ た ほど だ 。 "
    st = f.rm_between(st)
    #st = re.sub(r'.《.+?.》', "", st)
    print(st)
    # new_st = st.replace('《*》','')
    # print(new_st)

def main():
    myfile = File()
    myfile.get_files_indir()

    for fname in myfile.filelist :
        myfile.readfile(fname)
        myfile.writefile(fname)

if __name__ == "__main__" :
    #test()
    main()
