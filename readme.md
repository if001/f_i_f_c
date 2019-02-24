# fifc
![fifc](https://user-images.githubusercontent.com/15050355/53283888-e1397180-378f-11e9-8096-75b42a2bc0aa.png)

ある1つの文字をフォント画像の特徴量を使いベクトルに変換する  
特徴量ベクトルからある1つの文字への変換も可能  

## fifc_opt.py 
以下の変換が行える  

- char⇄img
- img⇄feature
- feature⇄char

以下の初期化をあらかじめ行う必要がある。  

## font_img
fontファイルから、1文字ごとに画像ファイルを生成する。  
画像のファイルの保存名は、1文字の読みを16進数変換したものを利用する。  

### 初期化
`font_img/font_img_opt.py`のmainを実行  
font_img/以下のimageディレクトリに全フォントの画像が生成される。  

## img_char
画像と1文字を相互に変換  
font_imgで画像が生成されていることが前提となる.  
毎回画像はロードせず、kvs(font_img/img_save_kvs.py)に  
flattenされた画像配列と読み方を保存.  

### 初期化
`img_char/img_char_opt.py`のmainを実行  
画像と読みをkvsに保存。  

## img_feature
画像と特徴量を相互変換  
変換をするためには、autoecoderを学習させておく必要がある。  

### 初期化
`img_feature/train.py`を実行  
学習用にfont_img/image/以下に画像ファイルを準備しておく。  
