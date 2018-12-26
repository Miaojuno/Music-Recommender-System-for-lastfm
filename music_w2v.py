
from gensim.models import Word2Vec

def w2v_train(id,size,window,rawpath):
    f=open(rawpath,"r", encoding='UTF-8')
    user='user_'
    id=str(id)
    for i in range(6-len(id)):
        user+='0'
    user+=id

    data = []
    data_line = []
    flag=0 #判断是否查找到所训练用户
    for i,line in enumerate(f.readlines()):
        if line.split("\t")[0]==user:
            data_line.append(line.split("\t")[-1][:-1])
            flag=1
        elif flag==1:
            data.append(data_line)
            break
    f.close()
    w2v_model = Word2Vec(data, sg=1, size=size,  window=window,  min_count=1, hs=0)
    return w2v_model,data

