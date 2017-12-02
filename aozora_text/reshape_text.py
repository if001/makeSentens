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
        if (("   \n" == line ) == True): flag += 1
        if (("  \n" == line ) == True): flag += 1
        if ((" \n" == line ) == True): flag += 1
        if (("\n" == line ) == True): flag += 1
        if ((" 。 " == line ) == True): flag += 1

        # その他
        if (("※" in line ) == True): flag += 1
        if ((" ＊ " in line ) == True): flag += 1
        if (("＊" in line ) == True): flag += 1
        if (("http:" in line ) == True): flag += 1
        if (("青空文庫" in line ) == True): flag += 1
        if (("入力 、 校正 、 制作" in line ) ): flag += 1
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


    def del_word(self,line):
        line = re.sub(r'\n', "", line)
        line = re.sub(r'「', "", line)
        line = re.sub(r'」', "", line)
        line = re.sub(r'【', "", line)
        line = re.sub(r'】', "", line)
        line = re.sub(r'\u3000', "", line)
        return line


    def add_token(self,line):
        # line = re.sub(r'。', "。 BOS", line)
        line = "BOS" + line
        return line


    def add_end(self,line):
        line += "。"
        return line


    def readfile(self,fname):
        project_dir = os.path.dirname(os.path.abspath(__file__))
        self.getlines = []

        with open(project_dir + '/' + fname,'r') as file:
            print("open "+fname)
            lines = file.read()
            lines = lines.split("。")
            for line in lines:
                if (self.checkline(line) == 0):
                    line = self.rm_between(line)
                    line = self.del_word(line)
                    line = self.add_token(line)
                    line = self.add_end(line)
                    self.getlines.append(line)


    def writefile(self, fname):
        project_dir = os.path.dirname(os.path.abspath(__file__))
        save_dir = "/"
        fname = "re_"+fname

        with open(project_dir + save_dir + fname, 'w') as file:
            print("save " + project_dir+save_dir+fname)
            for line in self.getlines:
                file.write(line+"\n")


def test():
    myfile = File()
    fname = "re_oshieto_tabisuru_otoko.txt"
    myfile.readfile(fname)
    myfile.writefile(fname)


def main():
    myfile = File()
    myfile.get_files_indir()

    for fname in myfile.filelist:
        myfile.readfile(fname)
        myfile.writefile(fname)

if __name__ == "__main__" :
    # test()
    main()
