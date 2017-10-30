#!/bin/sh
#
#ダウンロードを飛ばしたい場合は、
#python3 get_aozora.pyをスキップさせ、
#<re>のついたファイルを全て削除しておく
#

if [ $# -ne 2 ] ; then
    echo "invalid argument"
    echo "you set author id first"
    echo "you set save file name second"
    exit 1
fi

authorId=$1
savefile=$2

echo "start get_aozora.py"
python3 get_aozora.py -id $1
if [ $? -ne 0 ]; then
    echo "get_aozora.py error"
    exit 1
fi


echo "start unzip_somefile.sh"
./unzip_somefile.sh
if [ $? -ne 0 ]; then
    echo "unzip_somefile.sh error"
    exit 1
fi

echo "start mecab_text.sh"
./mecab_text.sh
if [ $? -ne 0 ]; then
    echo "./mecab_text.sh error"
    exit 1
fi

echo "reshape_text.py "
python3 reshape_text.py
if [ $? -ne 0 ]; then
    echo "reshape_text.py error"
    exit 1
fi

./linking_text.sh $savefile
if [ $? -ne 0 ]; then
    echo "./linking_text.sh error"
    exit 1
fi


mv re_* ./files
mv *.zip ./files
rm *.png
