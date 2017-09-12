#! /bin/sh
##
#青空文庫からテキストファイルを取ってくる
##


gettextfile(){
    file=$1
    if [ ! -e $file ]; then
	wget http://www.aozora.gr.jp/cards/001779/files/$file
	unzip -o $file
	rm $file
    fi
}

file=57105_ruby_59617.zip
gettextfile $file

file=57181_ruby_59529.zip
gettextfile $file

file=57240_ruby_60876.zip
gettextfile $file

file=57182_ruby_60006.zip
gettextfile $file

file=56674_ruby_61552.zip
gettextfile $file

file=56645_ruby_58194.zip
gettextfile $file

file=57183_ruby_60007.zip
gettextfile $file

file=57505_ruby_61213.zip
gettextfile $file

file=56673_ruby_58784.zip
gettextfile $file

file=57225_ruby_58894.zip
gettextfile $file

file=57226_ruby_58864.zip
gettextfile $file

file=57227_ruby_58809.zip
gettextfile $file

file=57228_ruby_58697.zip
gettextfile $file

file=56677_ruby_60183.zip
gettextfile $file

file=57343_ruby_59977.zip
gettextfile $file

file=57506_ruby_61212.zip
gettextfile $file

file=57185_ruby_60071.zip
gettextfile $file

file=57405_ruby_59991.zip
gettextfile $file

file=58039_ruby_61534.zip
gettextfile $file

file=56669_ruby_58718.zip
gettextfile $file

file=56646_ruby_58227.zip
gettextfile $file

file=54836_ruby_58195.zip
gettextfile $file

file=56671_ruby_59594.zip
gettextfile $file

file=57186_ruby_60160.zip
gettextfile $file

file=57187_ruby_60161.zip
gettextfile $file

file=56650_ruby_58200.zip
gettextfile $file

file=57108_ruby_60855.zip
gettextfile $file

file=56675_ruby_60091.zip
gettextfile $file

file=57109_ruby_60781.zip
gettextfile $file

file=57229_ruby_61307.zip
gettextfile $file

file=56672_ruby_61189.zip
gettextfile $file

file=56647_ruby_58166.zip
gettextfile $file

file=57190_ruby_58233.zip
gettextfile $file

file=56648_ruby_58198.zip
gettextfile $file

file=57192_ruby_59533.zip
gettextfile $file

file=56651_ruby_58728.zip
gettextfile $file

file=57414_txt_59976.zip
gettextfile $file

file=57193_ruby_59534.zip
gettextfile $file

file=57194_ruby_60070.zip
gettextfile $file

file=57230_ruby_59440.zip
gettextfile $file

file=57165_ruby_60437.zip
gettextfile $file

file=56649_ruby_59454.zip
gettextfile $file

file=57196_ruby_59530.zip
gettextfile $file

file=57197_ruby_58710.zip
gettextfile $file

file=56670_ruby_59514.zip
gettextfile $file


rm ./*.png
exit 0
