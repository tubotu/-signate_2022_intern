# signate_2022_intern
SIGNATE 22卒インターン選考コンペティション（予測モデル部門）優勝(https://signate.jp/competitions/411)

モデリング概要は　https://signate.jp/competitions/402/discussions/120210204

## コード使い方
bert_learning.py 
making_bert_features.py 
preprocessing.py 
learning.py 
predicting.py 

基本は上から順番に実行します。 

### bert_learning.py 
html_contentからtextを抽出したものを使いbertの学習をして、models_bertにモデルを出力します。 

### making_bert_features.py 
上で学習したモデルを用いて、bert_predフォルダ内にスタッキングに使う特徴量(予測確率)を出力します。 

今回は上の二つのモジュールを使って bert-base-uncased(256単語) bert-base-cased(512単語) roberta-base(256単語) roberta-base(512単語) の学習をしました。(uncasedの統一がなされていないのは私のミスですが時間が足りなかったため、そのまま用いました。) 
それぞれのモデルを学習するためにはpythonファイル内のMODEL_NAMEを書き換える必要があります。 
models_bert内にモデルが保存されています。モデル名の末尾が_512.pthとなっているのは512単語の学習をしたモデルです。 
特筆すべきこととしては、512単語学習させるときはメモリ節約のために各バッチサイズの変数設定を4に下げて学習を行いました。(256単語の時は16に設定しています。) 
making_bert_features.pyの最後でbert_predフォルダ内にスタッキングに使う特徴量を保存しています。pthファイル同様、末尾が_512.npyとなっているものは512単語の学習をしたモデルの出力となっています。 

### preprocessing.py 
前処理の関数でクロスバリデーション用に分けたデータのリストをまとめてpreprocessed_data内にpickleファイル形式で保存しています。 
bert_feature_addでbertの特徴量を取り込んでいます。 また、クロスバリデーションの分割情報も保存しています。(cv.pickle) 

### learning.py 
preprocessing.pyで生成したtrainとvalidのデータを用いてlightgbmとニューラルネットワークの学習を行います。 
モデルはmodelsフォルダにpickleファイルまたはh5ファイルとして出力されます。 
pickleファイルはlightgbmのクロスバリデーション時の5つのモデルがリストとしてまとめて保存されています。ニューラルネットワークのモデルはそれぞれモデルごとに保存されています。 

### predicting.py 
preprocessing.pyで生成したtestデータとlearning.pyで学習したモデルを利用してsubmitファイルの作成をします。


