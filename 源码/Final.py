# -*- coding: utf-8 -*-
import numpy as np
import os
import random
import math
import time
import pickle
from os.path import abspath, dirname, join
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class NewFactor:
      
    def prePossess(self, filename):
        data_root = os.path.abspath(join(dirname(abspath(__file__)), filename))
        f = open(data_root, 'r')
        lines = f.readlines()
        f.close()
        self.numOfUsers = 0
        self.numOfItems = 0
        userId = 0
        userItem = 0
        count = 0
        self.userItemRat = []  # 记录用户-物品-分数三元组(训练集)
        self.testUserItemRat = []  # 验证集
        self.items = {}
        self.userBias = {}  # 记录用户-偏置
        self.itemBias = {}  # 记录物品-偏置
        userTotalRatings = {}  # 记录用户-总评分
        userTotalItems = {}  # 记录用户-总评分商品数
        itemTotalRatings = {}  # 记录商品-总评分
        itemTotalUser = {}  # 记录商品-总评分用户数
        userRatings = 0.0

        # 用于对应物品的序号和编号
        num = 0
        total = 5021342
        dataset = np.linspace(1, total, num=total)
        trainLine, testLine = train_test_split(np.array(dataset), test_size=0.2)
        trainLine = set(trainLine) # 加速
        
        for line in lines:
            if total == 0:
                break
            total -= 1
            if line == "\n":
                continue
            uirTuple = []  # 三元组
            if line.__contains__("|"):
                # 该行是用户ID|该用户评分物品数
                # 先对上一个商品进行处理
                data = line.split("|")
                userId = int(data[0])
                userItem = int(data[1].strip())
                self.numOfUsers += 1  # 用户总数+1
                if userId not in self.userBias:
                    self.userBias[userId] = 0.0
                if userId not in userTotalRatings:  # 记录偏置
                    userTotalRatings[userId] = 0.0
                    userTotalItems[userId] = 0
            else:
                uirTuple.append(userId)  # uirTuple为当前三元组
                num += 1
                data = line.split()
                # 前者为物品号，后者为分数
                uirTuple.append(int(data[0]))
                if int(data[1]) == 0:
                    uirTuple.append(0.0000001)
                else:
                    uirTuple.append(float(data[1]))
                    # uirTuple.append(float(data[1])/50)
                if num in trainLine:
                    self.userItemRat.append(uirTuple)
                    if int(data[0]) not in self.items.keys():
                        # 记录item对应的序号
                        self.items[int(data[0])] = self.numOfItems
                        self.numOfItems += 1
                    userTotalRatings[userId] += float(data[1])
                    userTotalItems[userId] += 1
                    if int(data[0]) not in itemTotalRatings:  # 对当前商品，累加处理
                        itemTotalRatings[int(data[0])] = float(data[1])
                        itemTotalUser[int(data[0])] = 1
                    else:
                        itemTotalRatings[int(data[0])] += float(data[1])
                        itemTotalUser[int(data[0])] += 1

                else:
                    self.testUserItemRat.append(uirTuple)
        
        self.userItemRat = np.array(self.userItemRat)

        for item in itemTotalRatings:
            if itemTotalUser[item] != 0:
                self.itemBias[item] = itemTotalRatings[item] / itemTotalUser[item]
            else:
                self.itemBias[item] = 0.0

        for user in userTotalRatings.keys():
            if userTotalItems[user] != 0:
                self.userBias[user] = userTotalRatings[user] / userTotalItems[user]
            else:
                self.userBias[user] = 0.0
        
        self.overallMean = np.mean(self.userItemRat[:, 2])
        for i in self.userBias.keys():
            self.userBias[i] -= self.overallMean
        for j in self.itemBias.keys():
            self.itemBias[j] -= self.overallMean

        print("验证集有",len(self.testUserItemRat))
        print("训练集有",len(self.userItemRat))
        self.userItemRat = np.array(self.userItemRat)
        print("全局打分平均:",self.overallMean)

    def get_attribute(self, filename):  # 获得所有商品的属性
        data_root=os.path.abspath(join(dirname(abspath(__file__)), filename))
        f = open(data_root, 'r')
        lines = f.readlines()
        f.close()
        self.userAttriRat = []
        self.AttributeOfItem = {}  # 商品的属性
        UserToAttri = {}  # 用户对属性的[总评分,次数]
        AttriToUser = {}  # 属性对用户的[总评分,次数]
        UTAtotal = {}  # 用户创建过的总评分
        ATUtotal = {}  # 商品的总评分
        self.NumOfAttri = 0
        self.attris = {}  # 属性的重排序
        self.UserBiasA = {}  # 用户对属性评分的偏向
        self.AttriBias = {}  # 商品对用户评分的偏向
        self.overallMean2 = 0
        count=0
        for line in lines:
            count += 1
            item, a1, a2 = line.strip('\n').split('|')
            item = int(item)
            self.AttributeOfItem[item] = []
            if a1 != 'None':
                self.AttributeOfItem[item].append(int(a1))
                if a2 != 'None':
                    self.AttributeOfItem[item].append(int(a2))
            if count == 507172:
                break
        # 建立用户属性评分的稀疏矩阵
        for user, item, rating in self.userItemRat:
            user = int(user)
            if user not in UserToAttri.keys():  # 创建这个用户
                UserToAttri[user] = {}
            if item not in self.AttributeOfItem.keys():
                self.AttributeOfItem[item] = []
            for attri in self.AttributeOfItem[item]:
                # 对用户评分矩阵操作
                if attri not in UserToAttri[user].keys():
                    UserToAttri[user][attri] = [0, 0]
                UserToAttri[user][attri][0] += rating
                UserToAttri[user][attri][1] += 1

                # 对属性评分矩阵操作
                if attri not in AttriToUser.keys():
                    AttriToUser[attri] = {}
                    self.attris[attri] = self.NumOfAttri
                    self.NumOfAttri += 1
                if user not in AttriToUser[attri].keys():
                    AttriToUser[attri][user] = [0, 0]
                AttriToUser[attri][user][0] += rating
                AttriToUser[attri][user][1] += 1
        
        # 创建用户-属性评分三元组 获得用户的平均评分
        for user in UserToAttri.keys():
            UTAtotal[user] = 0
            for attri in UserToAttri[user].keys():
                total_rating = UserToAttri[user][attri][0]
                total_time = UserToAttri[user][attri][1]
                self.userAttriRat.append([user, attri, total_rating/total_time])
                UTAtotal[user] += total_rating/total_time
            if len(UserToAttri[user]) != 0:
                self.UserBiasA[user] = UTAtotal[user]/len(UserToAttri[user])
            else:
                self.UserBiasA[user] = 0
        
        for attri in AttriToUser.keys():
            ATUtotal[attri] = 0
            for user in AttriToUser[attri].keys():
                total_rating = AttriToUser[attri][user][0]
                total_time = AttriToUser[attri][user][1]
                ATUtotal[attri] += total_rating/total_time
            if len(AttriToUser[attri]) != 0:
                self.AttriBias[attri] = ATUtotal[attri]/len(AttriToUser[attri])
            else:
                self.AttriBias = 0

        self.userAttriRat = np.array(self.userAttriRat)

        self.overallMean2 = np.mean(self.userAttriRat[:, 2])
        for user in self.userBias.keys():
            self.UserBiasA[user] -= self.overallMean2
        for attri in self.AttriBias.keys():
            self.AttriBias[attri] -= self.overallMean2

    def __init__(self):
        # 训练数据预处理
        self.prePossess("train.txt")
        
        # 属性数据预处理
        self.get_attribute("itemAttribute.txt")

        self.k1 = 210
        self.steps1 = 12
        self.stuRate1 = 0.001
        self.lambda1 = 0.001

        self.k2 = 190
        self.steps2 = 8
        self.stuRate2 = 0.001
        self.lambda2 = 0.01

        self.coe1=0.75
        self.coe2=0.25

    def train(self, k1=210, lambda1=0.01, stuRate=0.001, steps=12, choose_step=False):  # 根据商品-用户模型进行打分训练

        # 随机数初始化矩阵
        self.qMatrix = 0.1 * np.random.randn(self.numOfUsers, k1) / np.sqrt(k1)  # u*k  q是用户矩阵
        self.pMatrix = 0.1 * np.random.randn(k1, self.numOfItems) / np.sqrt(k1)  # k*i  p是商品矩阵

        Rate = stuRate
        for step in range(steps):
            count = 0
            for user, item, rating in self.userItemRat:
                count += 1
                loss = 0.0
                user = int(user)
                item = int(item)
                # 转换为int很重要，否则下面报错
                
                # 随机梯度下降
                predict_rating = np.dot(self.qMatrix[user, :], self.pMatrix[:, self.items[item]])+self.overallMean+self.userBias[user]+self.itemBias[item]
                thisLoss = rating-predict_rating

                q_change = Rate*(thisLoss*self.pMatrix[:, self.items[item]]-lambda1*self.qMatrix[user, :])
                p_change = Rate*(thisLoss*self.qMatrix[user, :]-lambda1*self.pMatrix[:, self.items[item]])
                self.qMatrix[user, :] += q_change
                self.pMatrix[:, self.items[item]] += p_change

                self.userBias[user] += Rate * (thisLoss - lambda1 * self.userBias[user])
                self.itemBias[item] += Rate * (thisLoss - lambda1 * self.itemBias[item])
            if step == 4:
                Rate = Rate/10
            if choose_step and step > 6:
                thisRMSE = self.validate_item(lambda1=lambda1)
                if self.minLoss1 > thisRMSE:
                    self.minLoss1 = thisRMSE
                    self.k1 = k1
                    self.stuRate1 = stuRate
                    self.steps1 = step+1
                    self.lambda1 = lambda1
        
        loss = 0
        num = len(self.userItemRat)
        for user, item, rating in self.userItemRat:
            user = int(user)
            item = int(item)
            predict_rating = np.dot(self.qMatrix[user, :], self.pMatrix[:, self.items[item]])+self.overallMean+self.userBias[user]+self.itemBias[item]
            thisLoss = rating-predict_rating
            loss += np.square(thisLoss)/500
        loss1 = lambda1 * (((self.qMatrix * self.qMatrix).sum())/num)
        loss2 = lambda1 * (((self.pMatrix * self.pMatrix).sum())/num)
        loss3 = lambda1*((sum(list(map(lambda num: num*num, self.userBias.values()))))/num)
        loss4 = lambda1*((sum(list(map(lambda num: num*num, self.itemBias.values()))))/num)
        rmse = np.sqrt(loss/num*500+loss1+loss2+loss3+loss4)
        print('RMSE on train:'+str(rmse))

    def train_attri(self, k2=190, lambda2=0.01, stuRate=0.001, steps=8, choose_step = False):  # 根据属性-用户模型进行训练
        # 随机数初始化矩阵
        self.qMatrix2 = 0.1* np.random.randn(self.numOfUsers, k2) / np.sqrt(k2)  # u*k  q2是用户矩阵
        self.pMatrix2 = 0.1 * np.random.randn(k2, self.NumOfAttri) / np.sqrt(k2)  # k*i  p2是属性矩阵

        # 随机梯度下降
        Rate=stuRate
        for step in range(steps):
            count=0
            
            for user, attri, rating in self.userAttriRat:
                count+=1
                loss = 0.0
                user = int(user)
                attri = int(attri)
            
                predict_rating = np.dot(self.qMatrix2[user, :], self.pMatrix2[:, self.attris[attri]])+self.overallMean2+self.UserBiasA[user]+self.AttriBias[attri]
                thisLoss = rating-predict_rating

                q_change = Rate*(thisLoss*self.pMatrix2[:, self.attris[attri]]-lambda2*self.qMatrix2[user, :])
                p_change = Rate*(thisLoss*self.qMatrix2[user, :]-lambda2*self.pMatrix2[:,self.attris[attri]])
                self.qMatrix2[user, :] += q_change
                self.pMatrix2[:, self.attris[attri]] += p_change

                self.UserBiasA[user] += Rate * (thisLoss - lambda2 * self.UserBiasA[user])
                self.AttriBias[attri] += Rate * (thisLoss - lambda2 * self.AttriBias[attri])
            if step == 13:
                Rate = Rate/10
            if choose_step:  # 调参时使用，输出每次迭代的RMSE
                thisRMSE = self.validate_attri(lambda2=lambda2)
                if self.minLoss2 > thisRMSE:
                    self.minLoss2 = thisRMSE
                    self.k2 = k2
                    self.stuRate2 = stuRate
                    self.steps2 = step+1
                    self.lambda2 = lambda2

        loss = 0
        num = len(self.userAttriRat)
        for user, attri, rating in self.userAttriRat:
            user = int(user)
            attri = int(attri)
            predict_rating=np.dot(self.qMatrix2[user, :], self.pMatrix2[:, self.attris[attri]])+self.overallMean2+self.UserBiasA[user]+self.AttriBias[attri]
            thisLoss = rating-predict_rating
            loss += np.square(thisLoss)/500
        loss1 = lambda2 * (((self.qMatrix2 * self.qMatrix2).sum())/num)
        loss2 = lambda2 * (((self.pMatrix2 * self.pMatrix2).sum())/num)
        loss3 = lambda2*((sum(list(map(lambda num: num*num, self.UserBiasA.values()))))/num)
        loss4 = lambda2*((sum(list(map(lambda num: num*num, self.AttriBias.values()))))/num)
        rmse = np.sqrt((loss/num)*500+loss1+loss2+loss3+loss4)
        print('RMSE on train:'+str(rmse))

    def validate_item(self, lambda1=0.01):  # 仅基于商品-用户进行预测
        loss_item = 0
        for user, item, rating in self.testUserItemRat:
            user = int(user)
            item = int(item)
        
            if item not in self.itemBias.keys():
                if user not in self.userBias.keys():
                    p_rate1 = self.overallMean
                else:
                    p_rate1 = self.overallMean+self.userBias[user]
            else:
                if user not in self.userBias.keys():
                    p_rate1 = self.overallMean+self.itemBias[item]
                else:
                    p_rate1 = np.dot(self.qMatrix[user, :], self.pMatrix[:, self.items[item]])+self.overallMean+self.userBias[user]+self.itemBias[item]
            
            loss_item += np.square(rating-p_rate1)/500

        num = len(self.testUserItemRat)
        loss1 = lambda1 * (((self.qMatrix * self.qMatrix).sum())/num)
        loss2 = lambda1 * (((self.pMatrix * self.pMatrix).sum())/num)
        loss3 = lambda1*((sum(list(map(lambda num: num*num, self.userBias.values()))))/num)
        loss4 = lambda1*((sum(list(map(lambda num: num*num, self.itemBias.values()))))/num)
        rmse = np.sqrt(loss_item/num*500+loss1+loss2+loss3+loss4)

        return rmse

    def validate_attri(self, lambda2=0.01):  # 仅基于属性-用户进行预测
        loss_attri = 0
        for user, item, rating in self.testUserItemRat:
            count = 0
            user = int(user)
            item = int(item)
            p_rate2 = 0
            if user not in self.UserBiasA.keys():
                userb = 0
            else:
                userb = self.UserBiasA[user]
            if item not in self.AttributeOfItem.keys():  # 商品没有属性
                p_rate2 = self.overallMean2+userb
            else:  # 商品可能有属性
                for attri in self.AttributeOfItem[item]:
                    count += 1
                    if user not in self.UserBiasA.keys():  # 这个用户没打过分
                        if attri not in self.AttriBias.keys():
                            p_rate2 += self.overallMean2
                        else:
                            p_rate2 += self.overallMean2+self.AttriBias[attri]
                    else:  # 这个用户打过分
                        if attri not in self.AttriBias.keys():
                            p_rate2 += self.overallMean2+userb
                        else:
                            p_rate2 += np.dot(self.qMatrix2[user, :], self.pMatrix2[:, self.attris[attri]])+self.overallMean2+self.UserBiasA[user]+self.AttriBias[attri]

                if count == 0:
                    p_rate2 = self.overallMean2+userb
                else:
                    p_rate2 /= count
                
            loss_attri += np.square(rating-p_rate2)/500

        num = len(self.testUserItemRat)
        loss1 = lambda2 * (((self.qMatrix2 * self.qMatrix2).sum())/num)
        loss2 = lambda2 * (((self.pMatrix2 * self.pMatrix2).sum())/num)
        loss3 = lambda2*((sum(list(map(lambda num: num*num, self.UserBiasA.values()))))/num)
        loss4 = lambda2*((sum(list(map(lambda num: num*num, self.AttriBias.values()))))/num)
        rmse = np.sqrt((loss_attri/num)*500+loss1+loss2+loss3+loss4)
        
        return rmse   

    def validate(self, coe1=0.04, coe2=0.96, choose_coe=False):  # 基于商品-属性-用户进行打分预测
        loss = 0
        loss_item = 0
        loss_attri = 0
        for user, item, rating in self.testUserItemRat:
            count = 0
            user = int(user)
            item = int(item)

            # 计算用户对属性的打分
            p_rate2 = 0
            # 获得该用户打分的偏置
            if user not in self.UserBiasA.keys():
                userb = 0
            else:
                userb = self.UserBiasA[user]
            
            if item not in self.AttributeOfItem.keys():  # 商品没有属性
                p_rate2 = self.overallMean2+userb
            else:  # 商品可能有属性
                for attri in self.AttributeOfItem[item]:
                    count += 1
                    if user not in self.UserBiasA.keys():  # 这个用户没打过分
                        if attri not in self.AttriBias.keys():
                            p_rate2 += self.overallMean2
                        else:
                            p_rate2 += self.overallMean2+self.AttriBias[attri]
                    else:  # 这个用户打过分
                        if attri not in self.AttriBias.keys():
                            p_rate2 += self.overallMean2+userb
                        else:
                            p_rate2 += np.dot(self.qMatrix2[user, :], self.pMatrix2[:, self.attris[attri]])+self.overallMean2+self.UserBiasA[user]+self.AttriBias[attri]

                if count == 0:
                    p_rate2 = self.overallMean2+userb
                else:
                    p_rate2 /= count  # 用户对该商品属性打分的平均
            
            # 计算用户对商品的打分
            if item not in self.itemBias.keys():
                if user not in self.userBias.keys():
                    p_rate1 = self.overallMean
                else:
                    p_rate1 = self.overallMean+self.userBias[user]
            else:
                if user not in self.userBias.keys():
                    p_rate1 = self.overallMean+self.itemBias[item]
                else:
                    p_rate1 = np.dot(self.qMatrix[user, :], self.pMatrix[:, self.items[item]])+self.overallMean+self.userBias[user]+self.itemBias[item]
            predict_attri = p_rate2
            predict_item = p_rate1
            # 线性组合
            predict_rating = coe1*p_rate1+coe2*p_rate2
            
            predict_attri = round(predict_attri/10)*10
            predict_item = round(predict_item/10)*10
            predict_rating = round(predict_rating/10)*10

            if predict_rating > 100:
                predict_rating = 100
            if predict_rating < 0:
                predict_rating = 0
            if predict_attri > 100:
                predict_attri = 100
            if predict_attri < 0:
                predict_attri =0 
            if predict_item > 100:
                predict_item = 100
            if predict_item < 0:
                predict_item = 0
            
            thisLoss = rating-predict_rating
            loss += np.square(thisLoss)/500
            loss_attri += np.square(rating-predict_attri)/500
            loss_item += np.square(rating-predict_item)/500

        num = len(self.testUserItemRat)
        loss1 = coe1*self.lambda2 * (((self.qMatrix2 * self.qMatrix2).sum())/num) + \
            coe2*self.lambda1 * (((self.qMatrix * self.qMatrix).sum())/num)
        loss2 = coe1*self.lambda2 * (((self.pMatrix2 * self.pMatrix2).sum())/num) + \
            coe2*self.lambda1 * (((self.pMatrix * self.pMatrix).sum())/num)
        loss3 = coe1*self.lambda2*((sum(list(map(lambda num: num*num, self.UserBiasA.values()))))/num)\
            + coe2*self.lambda1*((sum(list(map(lambda num: num*num, self.userBias.values()))))/num)
        loss4 = coe1*self.lambda2*((sum(list(map(lambda num: num*num, self.AttriBias.values()))))/num)\
            + coe2*self.lambda1*((sum(list(map(lambda num: num*num, self.itemBias.values()))))/num)
        rmse = np.sqrt(loss/num*500+loss1+loss2+loss3+loss4)
        if not choose_coe:
            print('validation RMSE (only item) :'+str(np.sqrt(loss_item/num*500+loss1+loss2+loss3+loss4)))
            print('validation RMSE (only attribution) :'+str(np.sqrt(loss_attri/num*500+loss1+loss2+loss3+loss4)))
            print('validation RMSE (both item and attribution) :'+str(rmse))
        else:
            return rmse

    def get_result(self, filename):  # 获得对test打分的结果
        data_root = os.path.abspath(join(dirname(abspath(__file__)), filename))
        user_item_score = {}
        f = open(data_root, 'r')
        lines = f.readlines()
        f.close()
        coe1=self.coe1
        coe2=self.coe2
        for line in lines:
            if line.__contains__("|"):
                user, num = line.strip('\n').split('|')
                user = int(user)
                user_item_score[user] = {}
                user_item_score[user]['num'] = num
            else:  # 一下计算过程类似于validate函数
                item = int(line.strip('\n'))
                count = 0
                item = int(item)
                p_rate2 = 0
                if user not in self.UserBiasA.keys():
                    userb = 0
                else:
                    userb = self.UserBiasA[user]
                if item not in self.AttributeOfItem.keys():  # 商品没有属性
                    p_rate2 = self.overallMean2+userb
                else:  # 商品可能有属性
                    for attri in self.AttributeOfItem[item]:
                        count += 1
                        if user not in self.UserBiasA.keys():  # 这个用户没打过分
                            if attri not in self.AttriBias.keys():
                                p_rate2 += self.overallMean2
                            else:
                                p_rate2 += self.overallMean2+self.AttriBias[attri]
                        else:  # 这个用户打过分
                            if attri not in self.AttriBias.keys():
                                p_rate2 += self.overallMean2+userb
                            else:
                                p_rate2 += np.dot(self.qMatrix2[user, :], self.pMatrix2[:, self.attris[attri]]) + \
                                           self.overallMean2+self.UserBiasA[user]+self.AttriBias[attri]

                    if count == 0:
                        p_rate2 = self.overallMean2+userb
                    else:
                        p_rate2 /= count

                if item not in self.itemBias.keys():
                    if user not in self.userBias.keys():
                        p_rate1 = self.overallMean
                    else:
                        p_rate1 = self.overallMean+self.userBias[user]
                else:
                    if user not in self.userBias.keys():
                        p_rate1 = self.overallMean+self.itemBias[item]
                    else:
                        p_rate1 = np.dot(self.qMatrix[user, :], self.pMatrix[:, self.items[item]]) + \
                                  self.overallMean+self.userBias[user]+self.itemBias[item]
                predict_attri = p_rate2
                predict_item = p_rate1
                predict_rating = round(coe1*p_rate1+coe2*p_rate2)
                predict_rating = round(predict_rating/10)*10
                if predict_rating > 100:
                    predict_rating = 100
                if predict_rating < 0:
                    predict_rating = 0
                user_item_score[user][item] = int(predict_rating)
        
        self.test_result = user_item_score

    def save_result(self, filename):  # 写回结果文件
        root = os.path.abspath(join(dirname(abspath(__file__)), filename))
        file = open(root, 'a')
        file.seek(0)
        file.truncate()
        for user in self.test_result.keys():
            file.write(str(user)+'|'+self.test_result[user]['num']+'\n')
            for item in self.test_result[user].keys():
                if item == 'num':
                    continue
                file.write(str(item)+'  '+str(self.test_result[user][item])+'\n')
        file.close()

    def adjust(self):  # 对SVD调参
        # 对商品矩阵调参
        print("start to adjust parameters of SVD based on items.")
        self.minLoss1 = 300
        for i in range(18, 22):  # 对k1调参
            for sturate in [0.001, 0.0003]:  # 对学习率调参
                for lambda1 in [0.01, 0.1]:  # 对正则化因子调参
                    self.train(k1=i*10, lambda1=lambda1, stuRate=sturate, steps=15, choose_step=True)
        print("best k1:"+str(self.k1))
        print("best study rate:"+str(self.stuRate1))
        print("best lambda1:"+str(self.lambda1))
        print("best step1:"+str(self.steps1))
        
        # 对属性矩阵调参
        print("start to adjust parameters of SVD based on attributes.")
        self.minLoss2 = 300
        for i in range(15, 20):  # 对k2调参
            for sturate in [0.001, 0.0003]:  # 对学习率调参
                for lambda2 in [0.01, 0.1]:  # 对正则化因子调参
                    self.train_attri(k2=i*10, lambda2=lambda2, stuRate=sturate, steps=15, choose_step=True)
        print("best k2:"+str(self.k2))
        print("best study rate:"+str(self.stuRate2))
        print("best lambda2:"+str(self.lambda2))
        print("best step2:"+str(self.steps2))

    def adjust_coe(self):  # 调整线性组合的系数
        print("start to choose choose the best coefficient for linear combination.")
        self.minLoss3 = 300
        for coe1 in [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9,0.95]:
            rmse = self.validate(coe1=coe1, coe2=1-coe1, choose_coe=True)
            if rmse < self.minLoss3:
                self.minLoss3 = rmse
                self.coe1 = coe1
                self.coe2 = 1-coe1
        print("best coe1:"+str(self.coe1))
        print("best coe2:"+str(self.coe2))


def save_model(filename,ob):  # 保存模型
    root = os.path.abspath(join(dirname(abspath(__file__)), filename))
    file = open(root, 'wb')
    pickle.dump(ob, file)
    file.close()


def load_model(filename):  # 加载模型
    root = os.path.abspath(join(dirname(abspath(__file__)), filename))
    file = open(root, 'rb')
    ob = pickle.load(file)
    file.close()
    return ob


if __name__ == '__main__':

    load = ''
    model_root = os.path.abspath(join(dirname(abspath(__file__)), 'model.pkl'))
    exist = os.path.exists(model_root)

    if exist:  # 存在模型文件
        load = input("Do you want to load model? (input <y> to load model) ")
        if load == 'y':  # 读取模型
            total_time = time.time()
            print("start to load model.")
            newFactor = load_model(model_root)  # 读取原先训练好的类对象

    if not exist or load != 'y':  # 不存在模型文件或者用户不load模型
        print("start to arrange original data.")
        newFactor = NewFactor()
        adjust = input("Do you want to adjust parameters? (input <y> to adjust parameters) ")
        if adjust == 'y':  # 调整参数
            total_time = time.time()
            adjust_start = time.time()
            newFactor.adjust()
            print("adjust time:", (time.time()-adjust_start)/60)  # 对SVD调参的时间

            newFactor.train(k1=newFactor.k1, lambda1=newFactor.lambda1, stuRate=newFactor.stuRate1, steps=newFactor.steps1)
            newFactor.train_attri(k2=newFactor.k2, lambda2=newFactor.lambda2, stuRate=newFactor.stuRate2, steps=newFactor.steps2)

            adjust_start = time.time()
            newFactor.adjust_coe()
            print("adjust coe time:", (time.time()-adjust_start)/60)  # 获得最佳线性组合的时间

        else:  # 按照原先设定的参数进行训练
            total_time = time.time()
            print("strat to train SVD based on item.")
            newFactor.train(k1=newFactor.k1, lambda1=newFactor.lambda1, stuRate=newFactor.stuRate1, steps=newFactor.steps1)
            print("start to train SVD based on attribution.")
            newFactor.train_attri(k2=newFactor.k2, lambda2=newFactor.lambda2, stuRate=newFactor.stuRate2, steps=newFactor.steps2)

        save_model(model_root,newFactor)  # 保存模型
    
    print("start to validate.")
    newFactor.validate(coe1=newFactor.coe1, coe2=newFactor.coe2)  # 输出在验证集上的RMSE

    print("start to get results for test.txt.")
    newFactor.get_result(filename='test.txt')  # 获得在测试集上的结果
    newFactor.save_result(filename='result.txt')  # 测试集上结果保存

    print('total time:',(time.time()-total_time)/60)  # 输出花费的总时间
    print("All finished.")

    input("Press <enter> to quit")