#coding:utf-8
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import utils as nn_utils
import torch.optim as optim

import numpy as np
import heapq
import time
import random
import tqdm
import math
import pandas as pd


#use_cuda=1
use_cuda=0 #还没测试过

#############################设置随机种子，保证模型的可复现性
np.random.seed(2020)
torch.manual_seed(2020)
random.seed(2020)

if torch.cuda.is_available():
    if use_cuda:
        torch.cuda.manual_seed_all(2020)
    else:
        print('WARNING: You have a CUDA device, so you should probably run with --cuda')
############################################################################################


# Parameters
# ==================================================
if(use_cuda==1):
    ftype = torch.cuda.FloatTensor
    ltype = torch.cuda.LongTensor
else:
    ftype=torch.FloatTensor
    ltype=torch.LongTensor









class Short_Term_GRU(nn.Module):
    def __init__(self,user_sum,item_sum,batch_size,hidden_size,item_emb,user_emb,week_emb,check_time_emb):
        super(Short_Term_GRU,self).__init__()
        self.n_items=item_sum
        self.n_user=user_sum
        self.hidden_size=hidden_size
        self.n_layers=1
        self.batch_size=batch_size
        self.item_emb=item_emb
        self.user_emb = user_emb
        self.week_emb=week_emb
        self.check_time_emb=check_time_emb

        self.rnn = nn.GRU(self.hidden_size +33, self.hidden_size, self.n_layers, batch_first=True)


    def forward(self,item_input,week_input,time_input,seq_lengths_input,current_batch):
        # item=Variable(torch.LongTensor([[1,2,3],[4,5,0]]))
        # week=Variable(torch.LongTensor([[0,1,1],[7,7,7]]))
        # time=Variable(torch.LongTensor([[24,0,0],[22,1,1]]))  #batch,sequence,hidden_size
        seq_lengths = torch.LongTensor(seq_lengths_input).type(ltype)
        #print(item_input,"zheshi item_input")
        self.embed_item=self.item_emb(torch.LongTensor(item_input).type(ltype))
        #print(self.embed_item,"this is embedding item")
        self.embed_week=self.week_emb(torch.LongTensor(week_input).type(ltype))
        self.embed_time=self.check_time_emb(torch.LongTensor(time_input).type(ltype))
        input_item_relate=torch.cat([self.embed_item,self.embed_week,self.embed_time],dim=-1) #batch,sequence,3*hidden_size
        # print(input_item_relate.shape)
        # raise()
        pack = nn_utils.rnn.pack_padded_sequence(input_item_relate, seq_lengths, batch_first=True)
        # print(pack)
        # raise()

        # h0 = Variable(torch.zeros(self.n_layers, self.batch_size, self.hidden_size))
        h0 = Variable(torch.zeros(self.n_layers, current_batch, self.hidden_size)).type(ftype)
        out, _ = self.rnn(pack, h0)
        # print(out)
        unpacked=nn_utils.rnn.pad_packed_sequence(out,batch_first=True)
        # print("短期")
        # print(unpacked)
        # print(unpacked[0].shape)
        # raise()
        # time.sleep(10)
        return unpacked, _

        pass

import time
class Current_GRU(nn.Module):
    def __init__(self,user_sum,item_sum,batch_size,hidden_size,item_emb,user_emb,week_emb,check_time_emb):
        super(Current_GRU,self).__init__()
        self.n_items=item_sum
        self.n_user=user_sum
        self.hidden_size=hidden_size
        self.n_layers=1
        self.batch_size=batch_size
        self.item_emb=item_emb
        self.user_emb=user_emb

        self.week_emb=week_emb
        self.check_time_emb=check_time_emb

        self.rnn = nn.GRU(self.hidden_size +33, self.hidden_size, self.n_layers, batch_first=True)

    def forward(self,item_input,week_input,time_input,seq_lengths_input,current_batch):
        # item=Variable(torch.LongTensor([[1,2,3],[4,5,0]]))
        # week=Variable(torch.LongTensor([[0,1,1],[7,7,7]]))
        # time=Variable(torch.LongTensor([[24,0,0],[22,1,1]]))  #batch,sequence,hidden_size
        # seq_lengths = [3, 2]

        item_input = Variable(torch.LongTensor(item_input).type(ltype))
        week_input = Variable(torch.LongTensor(week_input).type(ltype))
        time_input = Variable(torch.LongTensor(time_input).type(ltype))  # batch,sequence,hidden_size
        seq_lengths = [int(session_max_length) for _ in range(current_batch)]  #全局找到的


        self.embed_item=self.item_emb(item_input)
        # print(self.embed_item)
        self.embed_week=self.week_emb(week_input)
        self.embed_time=self.check_time_emb(time_input)
        input_item_relate=torch.cat([self.embed_item,self.embed_week,self.embed_time],dim=-1) #batch,sequence,3*hidden_size
        # print(input_item_relate.shape)
        # raise()
        pack = nn_utils.rnn.pack_padded_sequence(input_item_relate, seq_lengths, batch_first=True)
        # print(pack)
        # raise()  #输入，输出，隐藏尺寸
        h0 = Variable(torch.zeros(self.n_layers, len(seq_lengths), self.hidden_size)).type(ftype)
        out, _ = self.rnn(pack, h0)
        #使用多余计算，然后利用花式索引找到对应的输出
        # print(_)
        # raise()
        # print(out)
        # raise()
        unpacked=nn_utils.rnn.pad_packed_sequence(out,batch_first=True)
        # print(unpacked)
        out_index = unpacked[0][[_ for _ in range(current_batch)], seq_lengths_input-1].reshape(1, current_batch, self.hidden_size)

        # time.sleep(10)
        return out_index

class Attention(nn.Module):
    def __init__(self,user_sum,item_sum,batch_size,hidden_size):
        super(Attention,self).__init__()
        self.user_sum=user_sum
        self.location_sum=item_sum
        self.hidden_size=hidden_size
        self.batch_size=batch_size
        self.liner = torch.nn.Linear(self.hidden_size, self.hidden_size)
        # self.liner_seqatta = torch.nn.Linear(self.hidden_size*2, self.hidden_size)
        # self.context = torch.nn.Parameter(torch.FloatTensor(self.hidden_size))

    # def forward(self,short_perference,current_perference,current_batch):
    #     current_perference = current_perference.transpose(0,1)
    #     # print(current_perference.shape)
    #     # raise()
    #     # print(current_perference.shape)
    #     # raise()
    #     current_perference = current_perference.repeat(1, short_perference[0].shape[1], 1)  #15表示最长长度
    #     # print(current_perference.shape)
    #     # print(current_perference)
    #     # print(short_perference[0])
    #     short_current_perference = torch.mul(short_perference[0], current_perference)
    #     # print(short_current_perference)
    #     short_current_line_input = short_current_perference.reshape(-1, self.hidden_size)  #-1表示送入神经网络的batch
    #
    #     # lineroutput = self.liner(short_current_line_input)  #加了个激活函数，效果好很多
    #     lineroutput = torch.tanh(self.liner(short_current_line_input))
    #     # print(lineroutput)
    #     lineroutput1 = lineroutput.reshape(current_batch, -1, self.hidden_size)  #-1表示序列长度
    #     # print(lineroutput)
    #     lineroutput2 = torch.sum(lineroutput1, dim=-1)
    #     softmax_lineroutput = torch.softmax(lineroutput2, dim=-1)
    #     # print(softmax_lineroutput)
    #     softmax_lineroutput = softmax_lineroutput.unsqueeze(-1)
    #     # print(softmax_lineroutput)
    #     softmax_lineroutput = softmax_lineroutput.repeat(1, 1, self.hidden_size)
    #     # print(softmax_lineroutput)
    #     # print(short_perference[0])
    #     # print(softmax_lineroutput)
    #     short_current_softmax_perference = torch.mul(short_perference[0], softmax_lineroutput)
    #     # print(short_current_softmax_perference)
    #     short_current_perference = torch.sum(short_current_softmax_perference, dim=1)
    #     # print(short_current_perference)  # batch,hidden_size
    #     return short_current_perference
    #

    def forward_batch_onebyone(self,short_perference,current_perference,current_batch,length_input):
        current_perference=current_perference[0] #batch,hidden—size
        lista=[]


        #分层注意力(真）：
        for i in range(current_batch):   #会比上面快那么一丁点，10秒左右吧
            h_i=short_perference[0][i][:length_input[i]]
            u_s=current_perference[i]
            u_i=torch.tanh(self.liner(h_i))
            us=torch.sum(torch.mul(u_i,u_s),dim=-1)
            alpha_i=torch.softmax(us,dim=-1)  #长度*1
            v=torch.sum(torch.mul(h_i,alpha_i.view(-1,1)),dim=0)
            lista.append(v)
        short_current_perference=torch.stack(lista)

        return short_current_perference


class Attention_perference(nn.Module):
    def __init__(self,hidden_size):
        super(Attention_perference,self).__init__()

        self.hidden_size=hidden_size
        self.liner = torch.nn.Linear(self.hidden_size, 1)
        self.context=nn.Parameter(torch.FloatTensor(1,self.hidden_size))  #一行嵌入列的上下文向量
        self.context.data.uniform_(-0.5,0.5)

    def forward(self,a1,a2,a3,a4):
        lista=[]
        for one_a1,one_a2,one_a3,one_a4 in zip(a1,a2,a3,a4):
            a_yuanshi=torch.stack([one_a1,one_a2,one_a3,one_a4])
            a1=torch.mul(a_yuanshi,self.context)
            a=torch.tanh(self.liner(a1))
            a_softmax=torch.softmax(a,dim=0)
            a=torch.sum(a_softmax*a_yuanshi,dim=0)
            lista.append(a)
        batch_a=torch.stack(lista)
        return batch_a  #batch,self.hidden   batch-a表示用户加过注意力后的总和偏好






sign = 0
class Gru_Recommendation(nn.Module):
    def __init__(self,user_sum,item_sum,batch_size,hidden_size,poi_emb_data):
        super(Gru_Recommendation, self).__init__()
        self.user_sum=user_sum
        self.item_sum=item_sum
        self.batch_size=batch_size
        self.hidden_size=hidden_size

        self.user_emb = nn.Embedding(self.user_sum, self.hidden_size)  # 短期和当前使用同一个嵌入向量,
        self.item_emb = nn.Embedding(self.item_sum, self.hidden_size)
        self.week_emb = nn.Embedding(8, 8)  # 星期
        self.check_time_emb = nn.Embedding(25, 25)  # 每天小时



        # raise()
        #用不到用户的嵌入向量

        #初始化嵌入向量
        # nn.init.normal_(self.item_emb.weight, std=0.01)
        # nn.init.normal_(self.user_emb.weight, std=0.01)
        # nn.init.normal_(self.week_emb.weight, std=0.01)
        # nn.init.normal_(self.check_time_emb.weight, std=0.01)


        self.model=Short_Term_GRU(user_sum,item_sum,batch_size=self.batch_size,hidden_size=self.hidden_size,item_emb=self.item_emb,user_emb=self.user_emb,week_emb=self.week_emb,check_time_emb=self.check_time_emb)
        self.current_model=Current_GRU(user_sum,item_sum,batch_size=self.batch_size,hidden_size=self.hidden_size,item_emb=self.item_emb,user_emb=self.user_emb,week_emb=self.week_emb,check_time_emb=self.check_time_emb)
        self.attention=Attention(user_sum,item_sum,batch_size=self.batch_size,hidden_size=self.hidden_size)
        #self.attention_preference = Attention_perference(hidden_size=hidden_size) #这是源代码
        self.attention_preference=Attention_perference(hidden_size=self.hidden_size)#这是我改动了一点，加了个self

        ###################考虑的因素个数
        # self.fenleiliner = torch.nn.Linear(self.hidden_size*2, self.hidden_size) #【用户向量，关系向量，短期目前交互向量，当前向量】
        #单一偏好分类
        self.fenleiliner = torch.nn.Linear(self.hidden_size, self.item_sum)  #

        self.catliner1 = torch.nn.Linear(self.hidden_size , self.hidden_size)
        self.catliner2 = torch.nn.Linear(self.hidden_size*2, self.hidden_size)
        self.catliner3 = torch.nn.Linear(self.hidden_size * 3, self.hidden_size)
        self.catliner4 = torch.nn.Linear(self.hidden_size*2 +100, self.hidden_size)  #100表示用户关系的嵌入尺寸
        ####################################################################

        #损失函数
        self.crossentropy=nn.CrossEntropyLoss()


        self.reset_parameters()  # 初始化变量参数-0.5-0.5之间，理论上正太分布初始化最好


        # 对签到时间信息做特殊的处理，使其在训练中不可以被训练
        self.week_emb.weight.requires_grad = False
        self.check_time_emb.weight.requires_grad=False
        self.week_emb.weight.data=torch.zeros(8, 8).scatter_(1, torch.LongTensor([[_] for _ in range(8)]),1)
        self.check_time_emb.weight.data=torch.zeros(25, 25).scatter_(1, torch.LongTensor([[_] for _ in range(25)]),1)
        # raise()
        # self.item_emb.weight.resquires_grad=False
        # self.item_emb.weight.data=torch.FloatTensor(poi_emb_data)

    def reset_parameters(self):  # 初始化所有定义的参数
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)


    def forward(self,target_input,month_poi_sequence_input,month_week_sequence_input,month_time_sequence_input,month_sequence_length_input,
                                                session_poi_sequence_input,session_week_sequence_input,session_time_sequence_input,session_sequence_length_input,check_history_input,current_batch,user_relation_input,neg_target_input):

        #########3                         计算用户的短期偏好
        short_perference,lastone_short_perference = self.model.forward(month_poi_sequence_input,month_week_sequence_input,month_time_sequence_input,month_sequence_length_input,current_batch)
        lastone_short_perference = lastone_short_perference[0]

        #########                          计算用户的当前偏好
        current_perference = self.current_model.forward(session_poi_sequence_input,session_week_sequence_input,session_time_sequence_input,session_sequence_length_input,current_batch)
        current_perference_real = current_perference  #为什么输出是三维的数据，不修改的原因是不知道current-pre在之后会不会像列表一样会变化呀
        ##########################################
        #print("current_preference:",len(current_perference))
        #print(current_perference)
        #############                       计算用户的交互偏好
        short_current_interact_perference = self.attention.forward_batch_onebyone(short_perference,current_perference,current_batch,month_sequence_length_input)  #这里面修改了目前的偏好是否会影响外面的偏好
        ###################################一定要验证current_perference的值变化了没有。应该没有变化，我都是使用的赋给新值的方法
        #print("short",len(short_current_interact_perference),len(short_current_interact_perference[1]))
        #print(short_current_interact_perference)

        ########                       计算用户的关系偏好
        user_relation_perference = Variable(torch.FloatTensor(user_relation_input)).type(ftype)  # 用户关系偏好，偏好预训练得到
        ####################################################################################
        #print("use_relation",len(user_relation_perference),len(user_relation_perference[1]))
        #print(user_relation_perference)
        ###                   计算用户的偏好偏好的整合方式@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

        # sum-perference表示偏好的混合
        ############################使用单一的偏好模型
        # sum_perference=lastone_short_perference  #使用短期偏好   #准确率可以达到100%
        # sum_perference=current_perference_real[0]  #使用目前偏好   #准确率先升40再降36最后可以达到100%
        # sum_perference=short_current_interact_perference #使用交互偏好  #准确率可以达到100%，但是拟合速度有点慢
        # sum_perference=user_history_perference
        # sum_perference =user_relation_perference
        # sum_perference = self.catliner1(sum_perference)  # 后面有加tanh
        # 说明每种偏好都能够成功的对用户的签到序列进行建模
        ###################################################


        ############全加   偏好拼接  训练集能达到100%说明当前和短期偏好都有影响
        ############当前和交互过的偏好。大约10次能够预测训练集中的90%左右了。
        sum_perference=torch.cat(
            [
                user_relation_perference,   #用户关系偏好
                short_current_interact_perference,  #周期偏好
                current_perference_real[0],         #当前偏好
            ], dim=-1)   #历史访问兴趣点偏好
        sum_perference = self.catliner4(sum_perference)  # 后面有加tanh
        # sum_perference = self.catliner4(sum_perference)  # 后面有加tanh
        # sum_perference = self.catliner4(sum_perference)  # 后面有加tanh
        # sum_perference = self.catliner4(sum_perference)  # 后面有加tanh
        ##################################3


        # #方法2 使用偏好相加，平均或者相乘的形式表示用户的混合偏好
        # #方法2 使用偏好相加，平均或者相乘的形式表示用户的混合偏好
        # ##################################################
        # #相加                     训练数据集上拟合可以达到100%   current-preference—real和current-preference相等
        # # sum_perference=torch.add(current_perference_real[0],last_short_perference[0])
        #                         # 3个相加也能达到100%准确率
        # # sum_perference=torch.add(torch.add(current_perference[0],last_short_perference),short_current_interact_perference)
        #
        # sum_perference=torch.add(current_perference[0],short_current_interact_perference)
        # sum_perference=torch.add(sum_perference,user_relation_perference)  #用户关系
        # sum_perference=torch.add(sum_perference,user_history_perference)  #用户历史偏好
        # #
        # #
        # # #平均                     平均也能达到100%准确率
        # sum_perference = sum_perference
        #相乘      还没实验   方法不好，很难拟合，而且迭代慢
        #sum_perference = torch.mul(current_perference_real[0], lastone_short_perference)  #迭代很慢



        ###方法3：使用注意力整合用户的综合偏好
        # sum_perference=self.attention_preference(
        #     user_relation_perference,
        #     short_current_interact_perference,
        #     current_perference_real[0],
        #     user_history_perference
        # )
        ####

        sum_perference=torch.tanh(sum_perference)    #好像训练速度（参数不变的情况下，迭代次数）会有点影响，准确率没有任何影响
        # sum_perference=torch.relu(sum_perference)  #还没有测试两者的区别？？？？
        ###########################################################################################3
        #print("sum_preference",len(sum_perference),len(sum_perference[1]))
        #print(sum_perference)

        ################                 交叉熵损失计算（将其看成分类任务）分为使用内积和全连接两种   #############3
        #####最终偏好的计算finall—perference的计算
        ##########方法1               使用单一偏好能够训练100%数据。 （最优模型）
        ############ 内积形式使用偏好和嵌入兴趣点的表示用户访问兴趣点的偏好,这种方法没毛病，可能不是最好的。但一定不差
        # finally_perference_score=torch.matmul(sum_perference,Variable(torch.t(self.item_emb.weight)))  #大约迭代15次达到100%
        finally_perference_score = torch.matmul(sum_perference, torch.t(self.item_emb.weight))  # 大约迭
        #print("zheshifenshu ---------------------------",finally_perference_score,"kankanxiaoguoba--------------------------")
        #########损失函数：交叉熵,bpr,负采样  bpr和负采样作为损失函数本文都还没有使用
        ##########计算最终的n个兴趣点分值偏好于目标兴趣点的误差，使用交叉熵函数
        loss=self.crossentropy(finally_perference_score, Variable(torch.LongTensor(target_input).type(ltype)))
        return finally_perference_score.cpu().detach().numpy(), loss
#事先从当前序列的K近邻序列中计算出每个兴趣点在该序列可能出现的概率，存到一个本地文件
df = pd.read_table('testSco100.txt',sep=' ', header = None)
df = df.values
ftarget=open("target.txt","r")  #读取用户ID和目标
targetlist=[]
id_list=[]
for i in ftarget:
    a=int(i.split("\t")[-1].replace("\n",""))
    id_list.append(int(i.split("\t")[0]))
    targetlist.append(a)

#1	4,6,4,5,4,3,2	0,1,5,6,6,3,3	1,19,3,15,19,19,22
fmonth=open("month.txt","r")  #读取序列的最大长度
month_max_length=0
for i in fmonth:
    a=len(i.split("\t")[1].split(","))
    if(a>month_max_length):
        month_max_length=a
fmonth.close()
fmonth=open("month.txt","r")
month_poi_sequence=[]
month_week_sequence=[]
month_time_sequence=[]
id_list=[]
month_sequence_length=[]
for i in fmonth:
    a=i.replace("\n","").split("\t")
    id_list.append(a[0])
    month_sequence_length.append(len(a[1].split(",")))
    if(len(a[1].split(","))<=month_max_length):
        one_poi_sequence=a[1].split(",")
        one_poi_sequence.extend([0 for i in range(month_max_length-len(one_poi_sequence))])
        one_week_sequence=a[2].split(",")
        one_week_sequence.extend([0 for i in range(month_max_length-len(one_week_sequence))])  #这里面0就有含义了，应该不能用0，先这样
        one_time_sequence=a[3].split(",")
        one_time_sequence.extend([0 for i in range(month_max_length-len(one_time_sequence))])
    month_poi_sequence.append(one_poi_sequence)
    month_week_sequence.append(one_week_sequence)
    month_time_sequence.append(one_time_sequence)
fmonth.close()

fsession=open("session.txt","r")  #读取序列的最大长度
session_max_length=0
for i in fsession:
    a=len(i.split("\t")[1].split(","))
    if(a>session_max_length):
        session_max_length=a
fsession.close()


fsession=open("session.txt","r")
session_poi_sequence=[]
session_week_sequence=[]
session_time_sequence=[]
session_sequence_length=[]
for i in fsession:
    a=i.replace("\n","").split("\t")
    session_sequence_length.append(len(a[1].split(",")))
    if(len(a[1].split(","))<session_max_length):
        one_poi_sequence=a[1].split(",")
        one_poi_sequence.extend([0 for i in range(session_max_length-len(one_poi_sequence))])
        one_week_sequence=a[2].split(",")
        one_week_sequence.extend([0 for i in range(session_max_length-len(one_week_sequence))])  #这里面0就有含义了，应该不能用0，先这样
        one_time_sequence=a[3].split(",")
        one_time_sequence.extend([0 for i in range(session_max_length-len(one_time_sequence))])
    session_poi_sequence.append(one_poi_sequence)
    session_week_sequence.append(one_week_sequence)
    session_time_sequence.append(one_time_sequence)
fsession.close()

users_checkins_dict={}
fuser_check=open("user_check.txt","r",encoding="utf-8")
for i in fuser_check:
    a=i.strip().split("\t")
    users_checkins_dict[int(a[0])]=[int(_) for _ in a[1].split(",")]
    if(len(users_checkins_dict[int(a[0])])>=100):
        users_checkins_dict[int(a[0])]=users_checkins_dict[int(a[0])][:100]  #只取用户最近签到的100个数据作为历史数据
        pass
    else:
        users_checkins_dict[int(a[0])].extend([0 for i in range(100-len(users_checkins_dict[int(a[0])]))])
        # pass
fuser_check.close()

# raise()


# ####################分割测试集和训练集
t_id_index_list=[]  #读取列表第一个作为测试集，因为签到是逆序的：：：注意
t_id_dict={}
for i in range(len(id_list)):
    try:
        t_id_dict[int(id_list[i])]
    except:
        t_id_dict[int(id_list[i])]=1
        t_id_index_list.append(int(i))




# if 1:
t_target=[]
t_month_poi_sequence=[]
t_month_week_sequence=[]
t_month_time_sequence=[]
t_month_sequence_length=[]
t_session_poi_sequence=[]
t_session_week_sequence=[]
t_session_time_sequence=[]
t_session_sequence_length=[]
t_id=[]


#删除索引从后向前，防止前面索引变化
t_id_index_list=t_id_index_list[::-1]   #逆序，先删除索引大的.里面的值表示索引
for i in t_id_index_list:
    try:
        t_month_poi_sequence.append(month_poi_sequence[i])
        del month_poi_sequence[i]
    except:
        print("a")
        pass

    t_month_week_sequence.append(month_week_sequence[i])
    del month_week_sequence[i]

    t_month_time_sequence.append(month_time_sequence[i])
    del month_time_sequence[i]

    t_month_sequence_length.append(month_sequence_length[i])
    del month_sequence_length[i]

    t_session_poi_sequence.append(session_poi_sequence[i])
    del session_poi_sequence[i]

    t_session_week_sequence.append(session_week_sequence[i])
    del session_week_sequence[i]

    t_session_time_sequence.append(session_time_sequence[i])
    del session_time_sequence[i]

    t_session_sequence_length.append(session_sequence_length[i])
    del session_sequence_length[i]

    t_target.append(targetlist[i])
    del targetlist[i]

    t_id.append(id_list[i])
    del id_list[i]
# ########################分割训练和测试


###############################初始化模型参数
# raise()
# batch_size=3

# raise()
batch_size=16
hidden_size=240
# user_sum=2195
# item_sum=3521
user_sum=2195
item_sum=3521

epoch=20
shujuliang=1000#以上数据都是基于数据集为500，batch-size为3测试得到的
# shujuliang=10000
learn_rate=0.01   #学习率被我固定了
momentum=0.9
weight_decay=0.0001
################################################################################

print("user_sum:"+str(user_sum))
print("item_sum:"+str(item_sum))
print("batch_size:"+str(batch_size))
print("hidden_size:"+str(hidden_size))
print("eopch:"+str(epoch))
print("learn_rate:"+str(learn_rate))
print("momentum:"+str(momentum))
print("weight_decay:"+str(weight_decay))



poi_relation=[]
#用户的嵌入数据写入
# print("写入POI关系文件,txt")
# from gensim.models import KeyedVectors
# wv = KeyedVectors.load("数据集处理/POI_sum_relation_500.kv", mmap='r')
# # vector = wv['1']  # numpy vector of a word
# poi_relation=np.zeros((item_sum,hidden_size))  #用户总数已经是多1了的
# for i in range(1,item_sum):
#     # print(i)
#     try:
#         poi_relation[i]=wv[str(i)]   #str必须是字典
#     except:
#         poi_relation[i]=[0 for i in range(100)]
#         print("warning:目标 "+str(i)+" 不再嵌入表中！")


#声明模型
if(use_cuda==1):
    gru_recommendation=Gru_Recommendation(user_sum,item_sum,batch_size=batch_size,hidden_size=hidden_size,poi_emb_data=poi_relation).cuda()
else:
    gru_recommendation = Gru_Recommendation(user_sum, item_sum, batch_size=batch_size, hidden_size=hidden_size,poi_emb_data=poi_relation)

#优化器  SGD优化器，Adam优化器，其他优化器
optimizer=optim.SGD(gru_recommendation.parameters(),lr=learn_rate,momentum=momentum
                    ,weight_decay=weight_decay
                    )

######################训练数据排序
month_sequence_length=np.asarray(month_sequence_length,dtype=int)
# print(month_sequence_length)
arg_sort_list = month_sequence_length.argsort()[::-1]  #找到从小到大的排序索引

#找到从下到大的排序坐标
targetlist=np.asarray(targetlist,dtype=int)[arg_sort_list]


#保持所有类别的序列对应关系
id_list=np.asarray(id_list,dtype=int)[arg_sort_list]


month_sequence_length=np.asarray(month_sequence_length,dtype=int)[arg_sort_list]
month_poi_sequence=np.asarray(month_poi_sequence,dtype=int)[arg_sort_list]
month_week_sequence=np.asarray(month_week_sequence,dtype=int)[arg_sort_list]
month_time_sequence=np.asarray(month_time_sequence,dtype=int)[arg_sort_list]

session_sequence_length=np.asarray(session_sequence_length,dtype=int)[arg_sort_list]
session_poi_sequence=np.asarray(session_poi_sequence,dtype=int)[arg_sort_list]
session_week_sequence=np.asarray(session_week_sequence,dtype=int)[arg_sort_list]
session_time_sequence=np.asarray(session_time_sequence,dtype=int)[arg_sort_list]
########################################################################################



#################             测试数据排序             ##################################
t_month_sequence_length=np.asarray(t_month_sequence_length,dtype=int)
# print(month_sequence_length)
t_arg_sort_list = t_month_sequence_length.argsort()[::-1]  #找到从小到大的排序索引

#找到从下到大的排序坐标
t_targetlist=np.asarray(t_target,dtype=int)[t_arg_sort_list]   #找到排序后的大小值，是t_targetlist,不是t_target

#保持所有类别的序列对应关系
t_id=np.asarray(t_id,dtype=int)[t_arg_sort_list]

t_month_sequence_length=np.asarray(t_month_sequence_length,dtype=int)[t_arg_sort_list]
t_month_poi_sequence=np.asarray(t_month_poi_sequence,dtype=int)[t_arg_sort_list]
t_month_week_sequence=np.asarray(t_month_week_sequence,dtype=int)[t_arg_sort_list]
t_month_time_sequence=np.asarray(t_month_time_sequence,dtype=int)[t_arg_sort_list]

t_session_sequence_length=np.asarray(t_session_sequence_length,dtype=int)[t_arg_sort_list]
t_session_poi_sequence=np.asarray(t_session_poi_sequence,dtype=int)[t_arg_sort_list]
t_session_week_sequence=np.asarray(t_session_week_sequence,dtype=int)[t_arg_sort_list]
t_session_time_sequence=np.asarray(t_session_time_sequence,dtype=int)[t_arg_sort_list]
###################################################################################################


iter_sum=0
#迭代次数重新计算 训练集
if(len(session_poi_sequence)%batch_size==0):
    iter_sum=int(len(session_poi_sequence)/batch_size)
else:
    print(str(len(session_poi_sequence)%batch_size)+",训练集不为0,最后有剩余")
    # raise()
    iter_sum=int(len(session_poi_sequence)/batch_size)+1
shujuliang_sum=iter_sum


#测试集合迭代次数重新计算
t_iter_sum=0
if(len(t_session_poi_sequence)%batch_size==0):
    t_iter_sum=int(len(t_session_poi_sequence)/batch_size)

else:
    t_iter_sum=int(len(t_session_poi_sequence)/batch_size)+1
    print(str(len(t_session_poi_sequence) % batch_size) + ",测试集不为0，最后迭代有剩余")
    # raise()
t_shujuliang=t_iter_sum



print("写入用户关系文件,txt")    #用户关系文件被嵌入到100维度了
from gensim.models import KeyedVectors
wv = KeyedVectors.load("sum_relation_500.kv", mmap='r')
# vector = wv['1']  # numpy vector of a word
user_relation_dict={}
for i in range(1,user_sum):
    # print(i)
    try:
        user_relation_dict[i]=wv[str(i)]   #str必须是字典
    except:
        user_relation_dict[i]=[0 for i in range(100)]
        #print("warning:目标 "+str(i)+" 不再嵌入表中！")
# raise()








# print("随机生成用户关系序列.txt")
# user_relation_dict=np.random.randn(3000,100)
# print(user_relation_dict.shape)
# print(user_relation_dict)
# raise()




best={"hit_sum_top1":0,"hit_sum_top5":0,"hit_sum_top10":0,
      "ceshi_index_top1":0,"ceshi_index_top5":0,"ceshi_index_top10":0}


def findnegtarget(neg_id):
    neg_item=random.randint(1,item_sum)
    while(neg_item in users_checkins_dict[neg_id]):
        neg_item = random.randint(1, item_sum)

    return neg_item

def start_predict(batch_size=1):
    ###########                          开始预测
    print("开始预测")


    hit_sum_top1 = 0
    hit_sum_top5 = 0
    hit_sum_top10 = 0
    ceshi_index_top1 = 0
    ceshi_index_top5 = 0
    ceshi_index_top10 = 0
    ceshisum = 0
    ceshilosssum = 0
    t_shujuliang=len(t_target)
    #print("数据量："+str(t_shujuliang))

    for j in range(t_shujuliang):
        if (j == t_shujuliang - 1):
            print("h")
        t_target_input = t_targetlist[j * batch_size:(j + 1) * batch_size]
        t_month_poi_sequence_input = np.asarray(t_month_poi_sequence[j * batch_size:(j + 1) * batch_size], dtype=int)
        t_month_week_sequence_input = np.asarray(t_month_week_sequence[j * batch_size:(j + 1) * batch_size], dtype=int)
        t_month_time_sequence_input = np.asarray(t_month_time_sequence[j * batch_size:(j + 1) * batch_size], dtype=int)
        t_month_sequence_length_input = np.asarray(t_month_sequence_length[j * batch_size:(j + 1) * batch_size],
                                                   dtype=int)
        t_session_poi_sequence_input = np.asarray(t_session_poi_sequence[j * batch_size:(j + 1) * batch_size],
                                                  dtype=int)
        t_session_week_sequence_input = np.asarray(t_session_week_sequence[j * batch_size:(j + 1) * batch_size],
                                                   dtype=int)
        t_session_time_sequence_input = np.asarray(t_session_time_sequence[j * batch_size:(j + 1) * batch_size],
                                                   dtype=int)
        t_session_sequence_length_input_real = np.asarray(
            t_session_sequence_length[j * batch_size:(j + 1) * batch_size], dtype=int)

        t_id_input = np.asarray(t_id[j * batch_size:(j + 1) * batch_size], dtype=int)
        check_history_input = np.asarray([users_checkins_dict[int(_)] for _ in t_id_input], dtype=int)

        user_relation_input = np.asarray([user_relation_dict[int(_)] for _ in t_id_input], dtype=float)

        # session_sequence_length_input = [15 for i in range(batch_size)]
        current_batch = len(t_target_input)  # 防止最后一个batch_size不是2，是1
        # 目标函数计算损失不需要负采样，也和预测无关，因此我随便选择了一个函数
        predict_logit, ceshiloss = gru_recommendation(t_target_input, t_month_poi_sequence_input,
                                                      t_month_week_sequence_input,
                                                      t_month_time_sequence_input, t_month_sequence_length_input,
                                                      t_session_poi_sequence_input, t_session_week_sequence_input,
                                                      t_session_time_sequence_input, t_session_sequence_length_input_real,
                                                      check_history_input, current_batch, user_relation_input,
                                                      t_target_input)
        # print("ce")
        # print(ceshiloss)
        # time.sleep(1)
        #print(predict_logit)
        for i in range(len(predict_logit[0])):
            predict_logit[0][i]=predict_logit[0][i]+df[i][int(t_id_input)]

        ceshilosssum += ceshiloss
        for one_predict, one_target, one_id in zip(predict_logit, t_target_input, t_id_input):
            #print(len(one_predict),"one_preict:",one_predict)
            ceshisum += 1
            topk = list(np.asarray(one_predict).argsort()[-10:])[::-1]  # 排序索引
            # print("this is topk :",topk)
            if (int(one_target) in topk):
                hit_sum_top10 += 1
                one_index = topk.index(int(one_target)) + 1  # 索引从1开始计数
                ceshi_index_top10 += 1.0 / one_index

                if (int(one_target) in topk[:5]):
                    hit_sum_top5 += 1
                    one_index = topk[:5].index(int(one_target)) + 1
                    ceshi_index_top5 += 1.0 / one_index

                    if (int(one_target) in topk[:1]):
                        hit_sum_top1 += 1
                        one_index = topk[:1].index(int(one_target)) + 1
                        ceshi_index_top1 += 1.0 / one_index
                # print("id:"+str(one_id))
                # print(topk)
                # print("__________________")
                # print(one_target)

                # topk.index(one_target)
                # raise()
        # summ+=1
        # if(summ>shujuliang):
        #     break

    hit_sum_top1 = hit_sum_top1 / ceshisum
    hit_sum_top5 = hit_sum_top5 / ceshisum
    hit_sum_top10 = hit_sum_top10 / ceshisum
    if (hit_sum_top1 > best["hit_sum_top1"]):
        best["hit_sum_top1"] = hit_sum_top1
    if (hit_sum_top5 > best["hit_sum_top5"]):
        best["hit_sum_top5"] = hit_sum_top5
    if (hit_sum_top10 > best["hit_sum_top10"]):
        best["hit_sum_top10"] = hit_sum_top10

    ceshi_index_top1 = ceshi_index_top1 / ceshisum
    ceshi_index_top5 = ceshi_index_top5 / ceshisum
    ceshi_index_top10 = ceshi_index_top10 / ceshisum
    if (ceshi_index_top1 > best["ceshi_index_top1"]):
        best["ceshi_index_top1"] = ceshi_index_top1
    if (ceshi_index_top5 > best["ceshi_index_top5"]):
        best["ceshi_index_top5"] = ceshi_index_top5
    if (ceshi_index_top10 > best["ceshi_index_top10"]):
        best["ceshi_index_top10"] = ceshi_index_top10

    print("Experiment result: ",best)






########################         主函数            #####################
if 1:
    for i in range(epoch):
        print("epoch   "+str(i)+"\n")
        ####################################################### 训练
        epoch_loss=0
        summ=0
        shujuliang=[_ for _ in range(shujuliang_sum)]
        random.shuffle(shujuliang)
        #print("this is shujuliang",shujuliang[:10])
        for j in tqdm.tqdm(shujuliang):
            target_input=targetlist[j*batch_size:(j+1)*batch_size]
            month_poi_sequence_input=np.asarray(month_poi_sequence[j*batch_size:(j+1)*batch_size],dtype=int)
            month_week_sequence_input=np.asarray(month_week_sequence[j*batch_size:(j+1)*batch_size],dtype=int)
            month_time_sequence_input=np.asarray(month_time_sequence[j*batch_size:(j+1)*batch_size],dtype=int)
            month_sequence_length_input=np.asarray(month_sequence_length[j*batch_size:(j+1)*batch_size],dtype=int)
            session_poi_sequence_input=np.asarray(session_poi_sequence[j*batch_size:(j+1)*batch_size],dtype=int)
            session_week_sequence_input=np.asarray(session_week_sequence[j*batch_size:(j+1)*batch_size],dtype=int)
            session_time_sequence_input=np.asarray(session_time_sequence[j*batch_size:(j+1)*batch_size],dtype=int)
            session_sequence_length_input_real=np.asarray(session_sequence_length[j*batch_size:(j+1)*batch_size],dtype=int)
            id_input=np.asarray(id_list[j*batch_size:(j+1)*batch_size],dtype=int)
            # neg_target_input=np.asarray([findnegtarget(_) for _ in id_input],dtype=int)
            neg_target_input =[0]

            check_history_input=np.asarray([users_checkins_dict[int(_)] for _ in id_input],dtype=int)

            user_relation_input = np.asarray([user_relation_dict[int(_)] for _ in id_input], dtype=float)
            # month_sequence_length_input=[165,165]
            # session_sequence_length_input=[session_max_length for _ in range(batch_size)]
            current_batch=len(target_input)  #防止最后一个batch_size不是2，是1


            optimizer.zero_grad()
            _,loss=gru_recommendation(target_input,month_poi_sequence_input,month_week_sequence_input,month_time_sequence_input,month_sequence_length_input,
                                                session_poi_sequence_input,session_week_sequence_input,session_time_sequence_input,session_sequence_length_input_real,check_history_input,current_batch,user_relation_input,neg_target_input)
            # print(loss)
            epoch_loss+=loss
            loss.backward()
            #防止梯度爆炸
            torch.nn.utils.clip_grad_value_(gru_recommendation.parameters(), 2)
            optimizer.step()
            # time.sleep(2)
            # print(epoch_loss)
            # summ+=1
            # if(summ>shujuliang):
            #     break
            # print(gru_recommendation.week_emb.weight[1])
            # time.sleep(3)
            if(j%4000==0):
                start_predict()

        print(epoch_loss)
        print("一次训练完成")
        ################################################################################################



