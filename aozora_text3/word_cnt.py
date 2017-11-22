import sys
import re

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




def main():
    from itertools import chain
    wordlist = readfile_to_sentens("./files/files_all_ogr.txt")
    #wordlist = readfile_to_sentens("./files/files_all_umn.txt")
    #wordlist = readfile_to_sentens("./files/files_all_rnp.txt")
    print("文章数:",len(wordlist))
    wordlist = list(chain.from_iterable(wordlist))
    print("非ユニーク単語数:",len(wordlist))
    wordlist = list(set(wordlist))
    print("ユニーク単語数:",len(wordlist))

if __name__ == "__main__":
    main()
