#!/bin/sh

./get_aozora.sh
if [ $? -ne 0 ]; then
    echo "./get_aozora.sh error"
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
    echo "./reshape_text.py error"
    exit 1
fi


echo "linking_text.sh"
./linking_text.sh
if [ $? -ne 0 ]; then
    echo "./linking_text.sh error"
    exit 1
fi
