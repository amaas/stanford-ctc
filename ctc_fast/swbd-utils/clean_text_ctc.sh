#!/bin/bash

for x in 'train' 'dev' 'eval2000'
do
    text=data/$x/text
    ctctext=data/$x/text_ctc
    cp $text $ctctext
    sed -i 's/_1/ /g' $ctctext
done

