# -*- coding: utf-8 -*-

import sys
import MeCab
import re

#文字列を受け取って、漢字をカタカナに変換する。
def get_words_to_katakana(wordline):
   m = MeCab.Tagger("-Ochasen")
   m.parse('')#空でパースする必要がある

   node = m.parseToNode(wordline)

   wordlist = []
   while(node):
      # print(node.feature.split(","))
      word = node.feature.split(",")[-2]

      if (word != "*") and (word != "々"):
         wordlist.append(word)
      node=node.next

   return wordlist

#カタカナをひらがなに変換
def kata_to_hira(wordlist):
   hiragana = ["ぁ","あ","ぃ","い","ぅ","う","ゔ","ぇ","え","ぉ","お","か","が","き","ぎ","く","ぐ","け","げ","こ","ご","さ","ざ","し","じ","す","ず","せ","ぜ","そ","ぞ","た","だ","ち","ぢ","っ","つ","づ","て","で","と","ど","な","に","ぬ","ね","の","は","ば","ぱ","ひ","び","ぴ","ふ","ぶ","ぷ","へ","べ","ぺ","ほ","ぼ","ぽ","ま","み","む","め","も","ゃ","や","ゅ","ゆ","ょ","よ","ら","り","る","れ","ろ","ゎ","わ","ゐ","ゑ","を","ん"]
   katakana = ["ァ","ア","ィ","イ","ゥ","ウ","ヴ","ェ","エ","ォ","オ","カ","ガ","キ","ギ","ク","グ","ケ","ゲ","コ","ゴ","サ","ザ","シ","ジ","ス","ズ","セ","ゼ","ソ","ゾ","タ","ダ","チ","ヂ","ッ","ツ","ヅ","テ","デ","ト","ド","ナ","ニ","ヌ","ネ","ノ","ハ","バ","パ","ヒ","ビ","ピ","フ","ブ","プ","ヘ","ベ","ペ","ホ","ボ","ポ","マ","ミ","ム","メ","モ","ャ","ヤ","ュ","ユ","ョ","ヨ","ラ","リ","ル","レ","ロ","ヮ","ワ","ヰ","ヱ","ヲ","ン"]

   suuji = list("0123456789?")
   zensuuji = list("０１２３４５６７８９？")

   kigou = list("!?()")
   zenkigou = list("！？（）")

   hira_wordlist = "s"
   for words in wordlist:
      for oneword in words:
         if (oneword in katakana):
            hira_wordlist += hiragana[katakana.index(oneword)]
         elif (oneword in zensuuji):
            hira_wordlist += suuji[zensuuji.index(oneword)]
         elif (oneword in zenkigou):
            hira_wordlist += kigou[zenkigou.index(oneword)]
         else :
            hira_wordlist += oneword

   hira_wordlist+="e"
   return [hira_wordlist]

#カタカナをひらがなに変換
def kata_to_hira_list(wordlist):
   hiragana = ["ぁ","あ","ぃ","い","ぅ","う","ゔ","ぇ","え","ぉ","お","か","が","き","ぎ","く","ぐ","け","げ","こ","ご","さ","ざ","し","じ","す","ず","せ","ぜ","そ","ぞ","た","だ","ち","ぢ","っ","つ","づ","て","で","と","ど","な","に","ぬ","ね","の","は","ば","ぱ","ひ","び","ぴ","ふ","ぶ","ぷ","へ","べ","ぺ","ほ","ぼ","ぽ","ま","み","む","め","も","ゃ","や","ゅ","ゆ","ょ","よ","ら","り","る","れ","ろ","ゎ","わ","ゐ","ゑ","を","ん"]
   katakana = ["ァ","ア","ィ","イ","ゥ","ウ","ヴ","ェ","エ","ォ","オ","カ","ガ","キ","ギ","ク","グ","ケ","ゲ","コ","ゴ","サ","ザ","シ","ジ","ス","ズ","セ","ゼ","ソ","ゾ","タ","ダ","チ","ヂ","ッ","ツ","ヅ","テ","デ","ト","ド","ナ","ニ","ヌ","ネ","ノ","ハ","バ","パ","ヒ","ビ","ピ","フ","ブ","プ","ヘ","ベ","ペ","ホ","ボ","ポ","マ","ミ","ム","メ","モ","ャ","ヤ","ュ","ユ","ョ","ヨ","ラ","リ","ル","レ","ロ","ヮ","ワ","ヰ","ヱ","ヲ","ン"]

   suuji = list("0123456789?")
   zensuuji = list("０１２３４５６７８９？")

   kigou = list("!?()")
   zenkigou = list("！？（）")

   hira_wordlist = ["s"]
   for words in wordlist:
      tmp = ""
      for oneword in words:
         if (oneword in katakana):
            tmp += hiragana[katakana.index(oneword)]
         elif (oneword in zensuuji):
            tmp += suuji[zensuuji.index(oneword)]
         elif (oneword in zenkigou):
            tmp += kigou[zenkigou.index(oneword)]
         else :
            tmp += oneword
      hira_wordlist.append(tmp)

   hira_wordlist.append("e")
   return hira_wordlist

#文字列を受け取って、単語ごとに切ったリストを返す
def get_words(wordline):
   m = MeCab.Tagger("-Owakati")
   m.parse('')
   words = m.parse(wordline)
   words_list = []

   tmp=""
   for value in words:
      tmp += value
      if value == " ":
         words_list.append(tmp[:-1])
         tmp = ""

   return words_list

if __name__ == '__main__':
   text = u's今日は良い天気ですね。e'
   # get_words(text)
   print(get_words(text))

   word_list = get_words_to_katakana(text)
   print(word_list)
   hira_word_list = kata_to_hira(word_list)
   print(hira_word_list)
