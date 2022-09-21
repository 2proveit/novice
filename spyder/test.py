import pandas as pd
import numpy as np

labor_income = list(np.random.uniform(low=19,high=23,size=1091)*10000)
for i in range(len(labor_income)):
    labor_income[i] = int(labor_income[i])

print('各户务工产值为\n',labor_income)

busi_income = list(np.random.uniform(low=3,high=5,size=1091)*10000)
for i in  range(len(busi_income)):
    busi_income[i] = int(busi_income[i])
print('各户经营产值为\n',busi_income)


output = pd.DataFrame({
    '经营收入':busi_income,
    '务工收入':labor_income

})
output.to_csv(r'C:\Users\asus\Desktop\share\new.csv',encoding='utf_8_sig')