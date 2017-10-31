#!/bin/sh
# ./all_in_one.sh <authorId> <save file name>
#ダウンロードを飛ばしたい場合は、
#python3 get_aozora.pyをスキップさせ、
#<re>のついたファイルを全て削除しておく

# 江戸川乱歩 1779
# 夢野久作 96
# 大阪圭吉 236
# 小栗虫太郎 125 *
# 海野十三 160 *



if [ $# -ne 2 ] ; then
    echo "invalid argument"
    echo "./all_in_one.sh <authorId> <save file name>"
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

echo "rm png"
rm *.png

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


rm re_* 
mv *.zip ./zip
#rm *.png
