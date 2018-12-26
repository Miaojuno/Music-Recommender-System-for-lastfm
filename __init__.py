
from music_train_c import lstm_train

count=0
all_count=0

for i in range(5,15):
    rawpath=r'F:\Date\lastfm\lastfm-dataset-1K\userid-timestamp-artid-artname-traid-traname.tsv'

    result=lstm_train(id=i,seq_length=30,w2c_length=50,window=7,nb_epoch=1,rawpath=rawpath)

    count1=result.count1    #   top1命中数
    count5 = result.count5  #   top5命中数
    count = result.count    #   top10命中数
    all_count = result.all_count    #   测试集总数
