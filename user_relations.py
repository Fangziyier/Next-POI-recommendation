
# id	checktime	location	jingweidu
# 1	2011-02-12 00:53:18	1	40.75051626137434,-73.9934992790222
# 1	2011-02-10 22:33:06	2	40.645089355976346,-73.7845230102539
#
import numpy as np

# youzoucishu=50
# youzoubushu=20
# raise()
f=open("my-CA-dataset-less10.txt")
f.readline()
location_user={}
user={}
# raise()
# a的形式为：['1', '2011-01-09 15:40:44', '5', '34.098,-118.328395']
for i in f:
    a=i.strip().split("\t")
    try:
        user[a[0]]
    except:
        user[a[0]]=1  #
        continue
    try:
        location_user[int(a[2])]
    except:
        location_user[int(a[2])]=[]
    if(int(a[0]) in location_user[int(a[2])]):
        pass
    else:
        location_user[int(a[2])].append(int(a[0]))
f.close()
print(len(location_user))



# 1	2011-02-10 22:33:06	2	40.645089355976346,-73.7845230102539
user_location_sum={}
f=open("my-CA-dataset-less10.txt")
f.readline()
for i in f:
    a=i.strip().split("\t")
    try:
        user_location_sum[int(a[0])]
    except:
        user_location_sum[int(a[0])]={}
    try:
        user_location_sum[int(a[0])][int(a[2])]
    except:
        user_location_sum[int(a[0])][int(a[2])]=0
    user_location_sum[int(a[0])][int(a[2])]+=1




aaa=0

check_relation=np.zeros((len(user)+1,len(user)+1))
for i in location_user:  #遍历所有地点，i表示一个地点
    for j in location_user[i]:  #遍历当前地点的用户j
        for k in location_user[i]:  #遍历当前地点的用户k
            if(j==k):
                pass
            else:
                min_sum=min([user_location_sum[j][i],user_location_sum[k][i]])
                if(min_sum)>1:
                    check_relation[j][k]+=min_sum
                    aaa+=1
                    # print(j,k)
print(aaa)

# import time
# time.sleep(10000)

# 709,6
# 709,1219
# 709,1220
f=open("new_friendship_ca.txt","r",encoding="utf-8")
friend_relation=np.zeros((len(user)+1,len(user)+1))
for i in f:
    a=i.strip().split(",")
    friend_relation[int(a[0])][int(a[1])]+=1
f.close()

hangmax=check_relation.sum(axis=1)
for i,one_hangmax in zip([_ for _ in range(len(hangmax))],hangmax):
    if(one_hangmax==0):
        pass
    else:
        check_relation[i]=check_relation[i]/one_hangmax
# check_relation=check_relation.cumsum(axis=1)

hangmax=friend_relation.sum(axis=1)
for i,one_hangmax in zip([_ for _ in range(len(hangmax))],hangmax):
    if (one_hangmax == 0):
        pass
    else:
        friend_relation[i] = friend_relation[i] / one_hangmax

# friend_relation=check_relation.cumsum(axis=1)
########## check_relation=check_relation.cumsum(axis=0)
zong_relation=0.5*friend_relation+0.5*check_relation

hangmax=zong_relation.sum(axis=1)
for i,one_hangmax in zip([_ for _ in range(len(hangmax))],hangmax):
    if (one_hangmax == 0):
        pass
    else:
        zong_relation[i] = zong_relation[i] / one_hangmax
zong_relation=zong_relation.cumsum(axis=1)


#生成随机数列
import random
sentences=[]
import tqdm


print("正在生成嵌入数据：")
for i in tqdm.tqdm(range(1,len(zong_relation))):
    # print("i")
    #生成50个的随机游走数列

    for _ in range(500):  #随机游走次数
        # print(_)
        if (np.sum(zong_relation[i]) == 0):  # 当前用户和任何人都没有关系
            print("没有边 "+str(i))
            break  #没有边和用户相邻
        one_list=[]
        one_list.append(i)
        flag=i
        while(len(one_list)!=50):  #50步长
            # print(len(one_list))
            one_random = random.random()  #生成0-1之间的数，不包括0和1
            if(one_random==0):
                import time
                print("yes")
                time.sleep(10)
                raise()

            for j in range(1,len(zong_relation[flag])): #因为元素多个0，所以不用加1
                # if(np.sum(flag)==0.):
                #     raise()
                # print(flag)
                # print("__")
                # print(np.sum(zong_relation[1059]))
                # print(np.sum(zong_relation[flag]))
                # print("zong" + str(zong_relation[flag][j]))
                # print(one_random)
                #
                # print(j)
                if(one_random<=zong_relation[flag][j]):
                    # if(np.sum(zong_relation[j])==0):
                    #     print("yes")
                    flag=j
                    one_list.append(j)
                    break  #找到了，继续找下一个元素
        # print(one_list)
        if(one_list==[]):
            raise()
            continue
        sentences.append(one_list)

fw=open("随机游走序列_500次_50步.txt","w+",encoding="utf-8")
for i in sentences:
    fw.write(",".join([str(_) for _ in i])+str("\n"))
fw.close()










#raise()  #由于生成随机游走序列太长，因此我把两个文件分开了
#
#
#
#
#
# from gensim.models import Word2Vec
# from gensim.models import KeyedVectors
# import datetime
# import random
#
# # f=open("4-1版本1用户关系随机游走序列.txt","r")
# # sentences = [
# #     ["3","1","3","9","3","4","3","2","3"],
# #     ["3","1","3","5","3","8","3","2","3","6","3"],
# #     ["3","10","3","6","3","2","3","1","3"],
# #     ["3","9","3","1","3"],
# #     ["3","1","3"]
# # ]
#
#
# # sentences=[]
# # asum=0
# # for i in f:
# #     #去掉第一个地点编码，后面的为用户编码
# #     aaa=i.replace("\n","").split(",")
# #     sentences.append(aaa)
# #     asum+=1
# #     # break
# #     ###################
# #     # if(asum==10000):
# #     #     break
# #     ##################
# # f.close()
#
#
#
# # for i in sentences:
# #     print(i)
# # raise()
# random.shuffle(sentences)
# print("开始训练")
# model = Word2Vec(min_count=1,size=100,workers=4)
# model.build_vocab(sentences)  # prepare the model vocabulary
# model.iter=1
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
#
#
# ######模型加载
# # model=Word2Vec.load("word2vec.model")
# #####################
#
# #######模型求相似度
# # print(model.similarity("1", "2"))
# # print(model.similarity("1", "3"))
# # print(model.similarity('dog', 'say'))
# ######################################
#
# ######打印模型的嵌入的向量值
# # print(model["say"])
# ####################################
#
# ###########模型保存
# model.save("sum_relation_100.model")  #模型保存
# model.wv.save("sum_relation_100.kv") #模型的嵌入向量文件保存
# ######################


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






