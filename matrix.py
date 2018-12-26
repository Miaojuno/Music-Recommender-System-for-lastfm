
def createdict(row_data,id):
    dict={}
    user = 'user_'
    id = str(id)
    for i in range(6 - len(id)):
        user += '0'
    user += id
    count=0
    flag = 0  # 判断是否查找到所训练用户
    for line in row_data.readlines():
        if line.split("\t")[0] == user:
            if line.split("\t")[-1][:-1] not in dict:
                dict[line.split("\t")[-1][:-1]]=count
                count+=1
            flag=1
        elif flag == 1:
            break
    return dict


def create_embedding_matrix(dict,w2v_model):
    embedding_matrix=[]
    for it,i in dict.items():
        embedding_matrix.append(w2v_model[it])
    return embedding_matrix

