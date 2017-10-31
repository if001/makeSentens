#!/bin/sh
##
#青空文庫から取ってきたテキストファイルを全て連結
##
error(){
    if [ $? -ne 0 ]; then
	echo "get_aozora.py error"
	exit 1
    fi
}

fname=$1

files=`ls | grep re_re_.*.txt`
cat $files > files_all_${fname}.txt
mv files_all_${fname}.txt ./files
exit 0
