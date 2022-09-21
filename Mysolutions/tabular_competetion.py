import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xg
from sklearn.metrics import mean_squared_error as mse
from sklearn.svm import SVR as svr
import matplotlib.pyplot as plt

train = pd.read_csv(r"D:\Kaggle\tabular-playground-series-sep-2022\train.csv")
test = pd.read_csv(r"D:\Kaggle\tabular-playground-series-sep-2022\test.csv")

n_train = train.shape[0]
test_ID = test.row_id
all_data = pd.concat((train, test)).reset_index(drop=True)  # 融合两个数据
all_data = all_data.drop(columns = "row_id")
all_data["date"] = pd.to_datetime(all_data["date"], format="%Y-%m-%d")
all_data["year"] = all_data.date.dt.year
all_data['month'] = all_data.date.dt.month
all_data['day'] = all_data.date.dt.day
all_data['day_of_week'] = all_data['date'].dt.day_of_week
all_data['day_of_year'] = all_data['date'].dt.day_of_year
all_data['is_weekend'] = np.where(all_data['day_of_week'].isin([5,6]), 1,0)
all_data = all_data.drop(columns = "date")

country_list = pd.unique(all_data.country)
for i, elem in enumerate(country_list):
    all_data = all_data.replace(elem,i)
store_list = pd.unique(all_data.store)
for i ,elem in enumerate(store_list):
    all_data = all_data.replace(elem,i)
all_data = all_data.replace("Kaggle Advanced Techniques",3)
all_data = all_data.replace("Kaggle Recipe Book",2)
all_data = all_data.replace("Kaggle Getting Started",1)
all_data = all_data.replace("Kaggle for Kids: One Smart Goose",0)


train = all_data[:n_train]
test = all_data[n_train:].drop(columns = "num_sold")
X = train.drop(columns = 'num_sold')
y = train.num_sold


X_train, X_val, y_train, y_val = train_test_split(X,y,random_state=42)


all_d = all_data.drop(columns = 'num_sold')
svr_ = svr()
svr_.fit(X_train,y_train)
svr_pre1 = svr_.predict(train)
svr_pre2 = svr_.predict(test)
train['svr'] = svr_pre1
test['svr'] = svr_pre2

X_train, X_val, y_train, y_val = train_test_split(X,y,random_state=42)

xgr = xg.XGBRegressor()
xgr.fit(X_train,y_train)
pre = xgr.predict(X_val)
loss = mse(y_val,pre)
print(loss)

prediction = xgr.predict(test)
pd.DataFrame({"row_id": test_ID,
              "num_sold": prediction}).to_csv("tabular_sep2022_submission.csv", index=False)
# 划分数据集
