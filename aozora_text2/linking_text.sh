#!/bin/sh
##
#青空文庫から取ってきたテキストファイルを全て連結
##

files=`ls | grep re_re_.*.txt`

cat $files > files_all.txt

exit 0
