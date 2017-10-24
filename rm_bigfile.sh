#!/bin/sh
TARGETS=(
    "wait/"
    "model/"
    "aozora_text/*.txt"
    "aozora_text2/*.txt"
    "*.png"
)

target=$(printf " %s" "${TARGETS[@]}")
target=${target:1}

git filter-branch -f --index-filter "git rm -r --cached --ignore-unmatch ${target}" -- --all
