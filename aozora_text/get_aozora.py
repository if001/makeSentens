"""
This program get aozora_bunko novels.
To do it, you set author_id as option.

example,
$ python3 get_aozora.py -id <author_id>

"""

import urllib.request
import os
import sys

def get_body(url):
    try:
        response = urllib.request.urlopen(url)
        print(">>>> get body : ",url)
        return response.read().decode('utf-8')
    except:
        print(" ----- html error cood",url)
        return ""


def get_cards(body):
    __cards = []
    for value in body.split("\n"):
        if ('<li>' in value) and ('.html' in value) :
            __cards.append(value.split('"')[1].split('/')[-1].split(".")[0].replace("card",""))
    return __cards


def get_zip_code(body,cardId):
    zipcode = []
    for value in body.split("\n"):
        if ('.zip' in value) and ('files' in value) and (cardId in value):
            zipcode.append(value.split('"')[1].split("/")[-1])
    if len(zipcode) != 0: return zipcode[0]
    if len(zipcode) == 0: return ""


def get_novel_body(authorId,cardId):
    url = "http://www.aozora.gr.jp/cards/"+ authorId + "/card" + cardId + ".html"
    return get_body(url)


def zero_padding(authorId):
    # authorIDが6けたなので足りないぶんを0で埋める
    if len(str(authorId)) < 6 :
        zero = ""
        for i in range(6 - len(authorId)):
            zero += "0"
        authorId = zero + authorId
    return authorId


def download(url,savedir,filename):
    print("download : " + filename)
    try:
        urllib.request.urlretrieve(url, savedir+filename)
        print("save ok to "+savedir+filename)
    except:
        print("html error cood",url)
        return ""


def get_path():
    return os.path.dirname(os.path.abspath(__file__))


def main():
    if ("-id" in sys.argv) and (len(sys.argv) == 3 ):
        authorId = sys.argv[2]
    else :
        print("invalid argument.")
        print("you must set '-id' as option")
        print("and set author_id list after 'id'")
        exit(0)

    project_path = get_path()
    #savedir = "/files"
    savedir = "/"
    if os.path.isdir(project_path+savedir):
        print("ok. save " + project_path + savedir + " exist.")
    else:
        print("file no exist.")
        print("make file to ",project_path+savedir)
        os.makedirs(project_path+savedir)

    url = "http://www.aozora.gr.jp/index_pages/person"+authorId+".html"
    body = get_body(url)
    cardIds = get_cards(body)

    authorId = zero_padding(authorId)

    zip_code = []
    for value in cardIds:
        body = get_novel_body(authorId,value)
        zip_code.append(get_zip_code(body,value))


    for value in zip_code:
        print(authorId)
        print(value)
        url = "http://www.aozora.gr.jp/cards/" + authorId + "/files/" + value
        download(url,project_path+savedir,value)

if __name__ == "__main__":
   main()
