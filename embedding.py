#coding:utf-8
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import datetime
import random

# f=open("4-1版本1用户关系随机游走序列.txt","r")
# sentences = [
#     ["3","1","3","9","3","4","3","2","3"],
#     ["3","1","3","5","3","8","3","2","3","6","3"],
#     ["3","10","3","6","3","2","3","1","3"],
#     ["3","9","3","1","3"],
#     ["3","1","3"]
# ]


# sentences=[]
# asum=0
# for i in f:
#     #去掉第一个地点编码，后面的为用户编码
#     aaa=i.replace("\n","").split(",")
#     sentences.append(aaa)
#     asum+=1
#     # break
#     ###################
#     # if(asum==10000):
#     #     break
#     ##################
# f.close()



# for i in sentences:
#     print(i)
# raise()



f=open("随机游走序列_100次_50步.txt","r",encoding="utf-8")
sentences=[]
for i in f:
    # print(i)
    # raise()
    a=i.strip().split(",")
    # print(a)
    sentences.append(a)
f.close()
# print(sentences)
sentences=sentences
print(len(sentences))


# raise()


# sentences = [
#     ["3","1","3","9","3","4","3","2","3"],
#     ["3","1","3","5","3","8","3","2","3","6","3"],
#     ["3","10","3","6","3","2","3","1","3"],
#     ["3","9","3","1","3"],
#     ["3","1","3"]
# ]

# sentences=[
#     [1,2,3,4,5,6,7,8,9,],
#     [2,3,4,5,6,7,8,9,10],
#     [3,4,5,6,7,8,9,10,11],
#     [4,5,6,7,8,9,10,11,12],
#     [5,6,7,8,9,10,11,12]
#
# ]

# ###一个嵌入模型的实例
# sentences=[]
# for i in range(100000):
#     lista=[]
#     for j in range(i,i+10):
#         lista.append(str(j))
#     sentences.append(lista)
# print(sentences)

model_with_loss = Word2Vec(
    min_count=0,
    compute_loss=True,
    hs=0,#1是分级softmax，0是负采样
    sg=1,#训练算法，1是skip-gram，0是CBOW
    seed=42,
    size=100,workers=10,iter=1
)


model_with_loss.build_vocab(sentences)  # prepare the model vocabulary

import tqdm
for i in tqdm.tqdm(range(50)):
    random.shuffle(sentences)
    print("计算损失：")
# getting the training loss value
    training_loss = model_with_loss.get_latest_training_loss()
    print(training_loss)
    print("开始培训：")
    model_with_loss.train(sentences=sentences,compute_loss=True,total_examples=model_with_loss.corpus_count,epochs=model_with_loss.iter)



# raise()



# print("开始训练")
# model = Word2Vec(min_count=1,size=100,workers=10)
# model.build_vocab(sentences)  # prepare the model vocabulary
# model.iter=100
#
# #模型迭代一遍所用的时间0:05:46.622112
#
# #打印一些相关参数
# # print(model.iter)
# # print(model.corpus_count)
# # print(model.window)
# # print(model.min_count)
# # print(model.sg)
# # print(model.alpha)
# # print(model.min_alpha)
#
# start = datetime.datetime.now()
# print("开始训练")
# model.train(sentences, total_examples=model.corpus_count, epochs=model.iter)  # train word vectors
# print("结束训练")
# end = datetime.datetime.now()
# print(end - start)


######模型加载
# model=Word2Vec.load("word2vec.model")
#####################

#######模型求相似度
print(model_with_loss.similarity("1", "2"))
print(model_with_loss.similarity("1", "45"))
# print(model.similarity('dog', 'say'))
######################################

######打印模型的嵌入的向量值
# print(model["say"])
####################################

###########模型保存
model_with_loss.save("sum_relation_500.model")  #模型保存
model_with_loss.wv.save("sum_relation_500.kv") #模型的嵌入向量文件保存
######################


# wv = KeyedVectors.load("wordvectors.kv", mmap='r')
# vector = wv['1']  # numpy vector of a word
#
# print(vector)
# '''[-1.06593193e-02  1.01109548e-02 -7.43591134e-03 -1.29635669e-02
#   1.48757352e-02 -6.94910530e-03 -7.58603495e-03  1.86305009e-02
#  -1.39087271e-02  5.99567546e-03 -8.04566836e-04  3.17187817e-03
#   7.25889811e-03  7.61491898e-03  2.50529544e-03  8.85979459e-03
#   4.98266332e-03 -1.38026932e-02  1.21847745e-02  3.91440745e-03
#   1.35238178e-03  2.69043865e-03  1.02743646e-02 -1.24960914e-02
#   6.98336586e-03  1.07352082e-02 -6.44208957e-03  9.34791899e-07
#  -9.54123773e-03  1.67856738e-02  1.25167414e-03 -6.10647677e-03
#   4.99763992e-04  1.24679571e-02 -7.71110365e-03 -2.21710838e-02
#   2.68112519e-03  6.57264935e-03 -4.69124410e-03  7.40040513e-03
#  -5.40207187e-03 -1.17933275e-02  8.06967635e-03  1.02361618e-02
#   1.23274531e-02 -5.51408203e-03 -6.43263198e-03  1.80766056e-03
#  -1.33589818e-03 -3.47106741e-03  7.17991963e-03 -1.34332001e-03
#  -7.33278552e-03  1.35073988e-02 -1.84385236e-02 -1.27615994e-02
#  -1.30736809e-02 -3.46858520e-03  1.79111597e-03  2.82797194e-03
#   7.64258066e-03 -7.70741049e-03  1.67404246e-02  1.52776400e-02]
# '''




#整合用户的朋友关系和用户的签到关系