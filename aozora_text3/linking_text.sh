#!/bin/sh
##
#青空文庫から取ってきたテキストファイルを全て連結
##

if [ $# -ne 1 ] ; then
    echo "invalid argument"
    echo "you set save file name"
    exit 1
fi

fname=$1

files=`ls | grep re_re_.*.txt`
cat $files > files_all_${fname}.txt

exit 0
