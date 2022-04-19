#coding:utf-8
import torch

# 1	2011-02-12 00:53:18	1	40.75051626137434,-73.9934992790222
# 1	2011-02-10 22:33:06	2	40.645089355976346,-73.7845230102539
# 1	2011-02-10 19:01:05	3	43.680659965522246,-79.61208343505858
# 1	2011-01-30 19:00:22	4	34.043058660157634,-118.26724290847778
# 1	2011-01-09 15:40:44	5	34.098,-118.328395
# 1	2011-01-08 03:05:56	4	34.043058660157634,-118.26724290847778
# 1	2011-01-04 19:04:31	6	34.072222,-118.36737
# 1	2010-12-27 01:56:47	4	34.043058660157634,-118.26724290847778
# 1	2010-11-27 17:54:24	7	34.101748533093314,-118.34288120269775
# 1	2010-11-21 03:12:29	8	34.04266749498825,-118.25533390045166
# 1	2010-11-03 20:20:15	9	34.18223423782782,-118.88562083244324
# 1	2010-10-29 19:25:56	10	33.99164944077473,-118.44616770744324
# 1	2010-09-10 03:40:06	11	40.648785086174755,-73.78984451293945

f=open("my-CA-dataset-less10.txt","r")  #数据集必须按时间逆序
f.readline()  #赛选掉第一行无用信息
user_checkins_dict={}
for i in f:
    a=i.strip().split("\t")
    try:
        user_checkins_dict[a[0]]
    except:
        user_checkins_dict[a[0]]=[]
    user_checkins_dict[a[0]].append(a)
f.close()

f=open("user_check.txt","w+")
for i in user_checkins_dict:
    f.write(str(i)+"\t"+",".join([i[2] for i in user_checkins_dict[i][1:]])+"\n")
f.close()
#raise()

# summ=0
# for i in user_checkins_dict:
#     print(user_checkins_dict[i])
#     summ+=1
#     if(summ>20):
#         break
# raise()

def shijiancha(i,j):  #i时间一定要大于j时间
    pass
import datetime
import time

#创建文件
fmonth=open("month.txt","w+")
fsession=open("session.txt","w+")
ftarget=open("target.txt","w+")

chulijishu=0
print("要处理数据："+str(len(user_checkins_dict)))
import tqdm
for i in tqdm.tqdm(user_checkins_dict):  #遍历用户ID
    chulijishu+=1
    # print("处理到第"+str(chulijishu))
    id = i
    for ii in range(len(user_checkins_dict[i])-1):  #目标兴趣点，除第一个个签到无法作为目标。遍历用户目标兴趣点
        target=user_checkins_dict[i][ii][2]

        month_poi_sequence = []
        month_week = []
        month_check_time = []

        session_poi_sequence = []
        session_week = []
        session_check_time = []
        for j in range(ii+1,len(user_checkins_dict[i])):  #j表示目标的当前访问兴趣点,j=ii+1,懒得改了
            a_j = user_checkins_dict[i][j][1]  # 当前兴趣点签到时间
            for k in range(j,len(user_checkins_dict[i])):  #k表示当前兴趣点之后的兴趣点.
                a_k=user_checkins_dict[i][k][1]  #在当前兴趣点之前的签到时间
                shijian_j=datetime.datetime.strptime(time.strftime(a_j), "%Y-%m-%d %H:%M:%S")  #获取日期
                shijian_k=datetime.datetime.strptime(time.strftime(a_k), "%Y-%m-%d %H:%M:%S")  #获取日期
                if(shijian_j<shijian_k):  #说明这个签到时间i小于输入时间k
                    pass
                else: #小于等于0包括自身
                    if(shijian_j<=shijian_k+datetime.timedelta(days=60)):#说明是之后的签到
                        month_poi_sequence.insert(0,user_checkins_dict[i][k][2])
                        month_week.insert(0,str(shijian_k.weekday()))
                        month_check_time.insert(0,str(shijian_k.hour))
                        if(shijian_j<=shijian_k+datetime.timedelta(hours=6)):  #说明时当前之前6小时签到
                            session_poi_sequence.insert(0,user_checkins_dict[i][k][2])
                            session_week.insert(0,str(int(shijian_k.weekday())+1))  #这一点我改动了下让week大于0，还没运行过，不知道对不对
                            session_check_time.insert(0,str(int(shijian_k.hour)+1))
                    else:
                        break
            # if(len(month_week)==1)  #只包括自己
            break  #j只需要循环一次就好了，可以不用循环，懒得改了


        fmonth.write(str(id)+"\t"+",".join(month_poi_sequence)+"\t"+",".join(month_week)+"\t"+",".join(month_check_time)+"\n")
        fsession.write(str(id)+"\t"+",".join(session_poi_sequence)+"\t"+",".join(session_week)+"\t"+",".join(session_check_time)+"\n")
        ftarget.write(str(id)+"\t"+str(target)+"\n")
        # 写入(id,poi,month_week,month_check_time)
        # 写入（id,session_poi_sequence,session_week,session_check_time)
        # 写入目标
        # raise()

fmonth.close()
fsession.close()
ftarget.close()
print("完成！")
    #     如果len(monthpoi)=1为null   表示month为null因为包括目标元素
    #     使用前一个签到，如果前一个签到不存在就表示这是用户的第一次签到，可以不用放入训练集中。
    #     同理 会话信息





