#! /bin/sh

files=`ls | grep .*zip`
echo "mecab_text.sh"
for file in $files
do
    echo $file
    unzip -o $file
done

exit 0


