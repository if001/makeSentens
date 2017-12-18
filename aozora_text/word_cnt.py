import sys
import re
from itertools import chain

def readfile_to_sentens(fname):
    try:
        wordlists = []
        with open(fname,'r') as file:
            for line in file :
                wordlists.append(line.split(" "))
        return wordlists
    except:
        print(fname+" dose not exist")
        sys.exit(0)


def word_cnt(fname):
    wordlist = readfile_to_sentens(fname)
    print(fname)
    print("文章数:",len(wordlist))
    wordlist = list(chain.from_iterable(wordlist))
    print("非ユニーク単語数:",len(wordlist))
    wordlist = list(set(wordlist))
    print("ユニーク単語数:",len(wordlist))
    print("--------")

def main():
    fname = "./files/files_all_rnp.txt"
    word_cnt(fname)

    fname = "./files/files_all_ymn.txt"
    word_cnt(fname)

    fname = "./files/files_all_osk.txt"
    word_cnt(fname)

    fname = "./files/files_all_ogr.txt"
    word_cnt(fname)

    fname = "./files/files_all_umn.txt"
    word_cnt(fname)




    
if __name__ == "__main__":
    main()
