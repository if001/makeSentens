#! /bin/sh

files=`ls | grep .*zip`
echo "unzip.sh"
for file in $files
do
    echo $file
    unzip -o $file
done

exit 0


