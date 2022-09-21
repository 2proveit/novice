#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
root_path = r'D:\QQ接收文件\FileRecv\MobileFile\\'
info_name = r'vil_info.xlsx'
house_area = r'house_area.xlsx'
field_area = r'field_area.xls'


# In[2]:


vil_info = pd.read_excel(root_path + info_name)
vil_house = pd.read_excel(root_path + house_area)
vil_field = pd.read_excel(root_path + field_area)
vil_info.head()


# In[3]:


huzhu_count = len(vil_info.loc[vil_info.relation =='户主'])
huzhu = vil_info.loc[vil_info.relation =='户主']
print("共有{}户".format(huzhu_count))
print(huzhu)


# In[4]:


hu = []
huzhu_name = huzhu.name # 户主姓名
hu_name = list(huzhu_name)


# In[5]:


hu_num_count = [] # 家庭人口数
temp = 0
huzhu_index_list = huzhu_name.index.tolist
for i in huzhu_index_list():    
    hu_num_count.append(i - temp)
    temp = i
hu_num_count.remove(0)
hu_num_count.append(2)

print('共有{}户\n'.format(len(hu_num_count)))
print(hu_num_count,'\n\n','计算完毕!'.center(100,'*'))


# In[ ]:





# In[6]:


label_num_count=[]
edu_count = []
edu_expend = []
med_expend = []

for i in range(huzhu_count):
    label=0
    edu = 0 # 3 为大专及以上， 2 为初高中， 1 为初中及以下
    num = 0 # 家庭参保人数
    edu_spd= 0 # 家庭读书花费
    
    # 每户进行判断
    if i<huzhu_count-1:
        age_cut = list(vil_info.age[huzhu_index_list()[i]:huzhu_index_list()[i+1]])
        sex_cut = list(vil_info.sex[huzhu_index_list()[i]:huzhu_index_list()[i+1]])
        
        for j in range(len(age_cut)):
            # 判断劳动人口
            if (sex_cut[j] == '男' and age_cut[j] < 60.0 and age_cut[j] > 14.0 ) or (sex_cut[j] == '女' and age_cut[j] < 55.0 and age_cut[j] > 14.0):
                label +=1
            else:
                label +=0
            # 判断最高学历
            if age_cut[j] >= 18.0 and age_cut[j] <= 35.0:
                edu = 3
            elif age_cut[j] > 35:
                if edu == 0:
                    edu = 1
            elif edu <=2:
                    edu = 2
            # 判断参保险支出
            if age_cut[j]<70:
                num+=1
            # 判断教育支出
            if age_cut[j]>=4.0 and age_cut[j]<=6:
                edu_spd += int(np.random.uniform(low=4000,high=5000))
            elif age_cut[j]>6 and age_cut[j]<=13:
                edu_spd += int(np.random.uniform(low=300,high=400))
            elif age_cut[j]>13 and age_cut[j]<=16:
                edu_spd += int(np.random.uniform(low=3500,high=4000))
            elif age_cut[j]>16 and age_cut[j]<=19:
                edu_spd += int(np.random.uniform(low=13000,high=15000))
            elif age_cut[j]>19 and age_cut[j]<=22:
                edu_spd += int(np.random.uniform(low=7000,high=8000))
                
        edu_count.append(edu)
        if label >=4:
            label_num_count.append("4个及以上")
        else:
            label_num_count.append(label)
        med_expend.append(num*280)# 参保每人280元
        edu_expend.append(edu_spd)
        
edu_count.append(3)  
label_num_count.append(2)
med_expend.append(2*280)
edu_expend.append(0)
print("共有{}户\n".format(len(label_num_count)),'各户有劳动人口:\n',label_num_count)
print("\n各户学历为",edu_count)
print("\n各户参保支出为",med_expend)
print("计算完毕！".center(120,"*"))


# In[7]:


edu_class_count = []
for i in edu_count:
    if i==1:
        edu_class_count.append('初中及以下')
    if i==2:
        edu_class_count.append('中专及高中')
    if i==3:
        edu_class_count.append('大专及以上')
print("各户最高学历",edu_class_count)


# In[8]:


# 家庭收入
arg_income = list(np.random.uniform(low=1.5,high=3,size=1091)*1000)
for i in range(len(arg_income)):
    arg_income[i] = int(arg_income[i])
print('各户种养产值为\n',arg_income)

busi_income = list(np.random.uniform(low=3,high=5,size=1091)*10000)
for i in  range(len(busi_income)):
    busi_income[i] = int(busi_income[i])
print('各户经营产值为\n',busi_income)

labor_income = list(np.random.uniform(low=9,high=12,size=1091)*10000)
for i in  range(len(labor_income)):
    labor_income[i] = int(labor_income[i])
print('各户务工产值为\n',labor_income)



other_income = list(np.random.normal(loc=12000,scale=6000,size=1091))
for i in  range(len(other_income)):
    if other_income[i]<0:
        other_income[i] = abs(other_income[i])
    other_income[i] = int(other_income[i])
print('各户其他收入为\n',other_income)


# In[9]:


# 支出
arg_expend = []
for i in arg_income:
    arg_expend.append(int(i/int(np.random.uniform(low=9,high=12))))
print('各户种养产值为\n',arg_expend)

busi_expend = []
for i in busi_income:
    busi_expend.append(int(i/int(np.random.uniform(low=7,high=8))))
print("各户经营成本为\n",busi_expend)

other_expend = []
for i in other_income:
    other_expend.append(int(i/int(np.random.uniform(low=13,high=15))))
print("各户其他支出成本为\n",other_expend)

heal_expend = []
for i in range(len(other_expend)):
    heal_expend.append(int(np.random.uniform(low=1000,high=1500)))
print('医疗支出为\n',heal_expend)

party = []
for i in range(len(other_expend)):
    party.append('否')
    


# In[ ]:


# 资产评估

house_property = []
field_property = []

print(vil_house)
for i in huzhu_name:
    try:
        house_property.append(vil_house.loc[vil_house['name'] == i ].area)
    except KeyError:
        print('找不到{}的房屋面积'.format(i))
        house_property.append(np.random.uniform(low=125,high=150))
    try:
        field_property.append(vil_field.loc[vil_field['name'] == i,].area)
    except KeyError:
        field_property.append(int(np.random.uniform(low=1.2,high=1.5)))

house_price =[]
field_price = []
for i in range(len(huzhu_name)):
    house1 = list(house_property[i])
    field1 = list(field_property[i])
    if len(house1)!=0:
        house_price.append(int(house1[0]*800))
    else:
        house_price.append(int(np.random.uniform(low=120,high=150)*800))

    if len(field1)!=0:
        field_price.append(field1[0])
    else:
        f1 = round(np.random.uniform(low=1.2,high=1.5),2)
        field_price.append(f1)





field_price1=[]
for i in field_price:
    field_price1.append(int(i*41120))
print('每户土地补贴金额\n',house_price)
          

care_income = []
for i in  field_price:
    care_income.append(i*132)
print('各户惠农补贴为\n',care_income)

yes_lst = []
for i in range(len(other_expend)):
    yes_lst.append(np.random.randint(low=28,high=30))


# In[ ]:


output = pd.DataFrame({'户主':huzhu_name,
                       '家庭人口数':hu_num_count,
                       '劳动力数':label_num_count,
                       '最高学历':edu_class_count,
                       '是否党员':party,
                       '种养产值':arg_income,
                       '经营收入':busi_income,
                       '务工收入':labor_income,
                       '惠农补贴':care_income,
                       '其他收入':other_income,
                       '种养支出':arg_expend,
                       '经营支出':busi_expend,
                       '医疗支出':heal_expend,
                       '教育支出':edu_expend,
                       '参保支出':med_expend,
                       '其他支出':other_expend,
                       '房产价格':house_price,
                       '土地资源面积':field_price,
                       '土地资源价格':field_price1,
                       # '新农合':yes_lst,
                       '乡风评议':yes_lst
                       })

output.to_csv(r'C:\Users\asus\Desktop\share\output1.csv',encoding='utf_8_sig')


# In[ ]:




