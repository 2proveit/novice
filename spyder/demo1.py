import uiautomator2 as u2
import numpy as np
import pandas as pd
import time
d = u2.connect()
info = pd.read_excel(r'C:\Users\asus\Desktop\share\imput.xlsx')

i='甘双月'# 此处显示姓名

huzhu_name = list(info.name)
def reason():
    if np.random.randint(0, 2):
        d.xpath(r"//*[@text='农户无手机']").click_exists(timeout=1)
    else:
        d.xpath(r"//*[@text='外出务工']").click_exists(timeout=1)
d.xpath(
    r"//*[@resource-id='app']/android.view.View[1]/android.view.View[1]/android.view.View[2]/android.view.View[1]/android.view.View[1]/android.view.View[2]").get_text()
def password_input():
    for i in range(10):
        temp = np.random.randint(low=0, high=10)
        d.xpath(r"//*[@text='{}']".format(temp)).click_exists(timeout=0.5)
        time.sleep(0.5)




d.xpath("//*[@text='{}']".format(i)).click(timeout=0.5)
d.xpath(r"//*[@text='已读，跳过']").click_exists(timeout=1)
d.xpath(r"//*[@text='特殊验证']").click_exists(timeout=1)
# 点击特殊验证原因
d.xpath(
    r"//*[@resource-id='app']/android.view.View[1]/android.view.View[2]/android.view.View[1]/android.view.View[2]/android.view.View[1]/android.view.View[2]").click_exists(
    timeout=1)
reason()
# 点击验证码区域
d.xpath(
    "//*[@resource-id='app']/android.view.View[1]/android.view.View[2]/android.view.View[1]/android.view.View[4]").click_exists(timeout=1)
password_input()

# 点击提交
d.xpath("//*[@text='提交']").click_exists()

# 点击家庭收入信息
print("开始填写家庭收入信息...")
d.xpath(
    r"//*[@resource-id='incomeList']/android.view.View[1]/android.view.View[2]/android.view.View[1]").click_exists()

d.xpath(
    r"//*[@resource-id='app']/android.view.View[1]/android.view.View[2]/android.view.View[1]/android.view.View[1]/android.view.View[1]/android.view.View[1]/android.view.View[1]/android.view.View[1]/android.view.View[1]/android.view.View[3]/android.view.View[1]/android.view.View[2]").click_exists()
time.sleep(0.1)
d.send_keys("{}".format(list(info.loc[info.name == i].arg_income)[0]), clear=True)  # 更换成种养收入
print("种养收入填写完毕...")
d.xpath(
    r"//*[@resource-id='app']/android.view.View[1]/android.view.View[2]/android.view.View[1]/android.view.View[1]/android.view.View[1]/android.view.View[1]/android.view.View[1]/android.view.View[1]/android.view.View[1]/android.view.View[4]/android.view.View[1]/android.view.View[2]").click_exists()
time.sleep(0.1)
d.send_keys("{}".format(list(info.loc[info.name == i].busi_income)[0]), clear=True)  # 更换成经营收入
print("经营收入填写完毕...")
d.xpath(
    r"//*[@resource-id='app']/android.view.View[1]/android.view.View[2]/android.view.View[1]/android.view.View[1]/android.view.View[1]/android.view.View[1]/android.view.View[1]/android.view.View[1]/android.view.View[1]/android.view.View[5]/android.view.View[1]/android.view.View[2]").click_exists()
time.sleep(0.1)
d.send_keys("{}".format(list(info.loc[info.name == i].labor_income)[0]), clear=True)  # 更换成务工收入
print("务工收入填写完毕...")
time.sleep(0.1)
d.xpath(
    r"//*[@resource-id='app']/android.view.View[1]/android.view.View[2]/android.view.View[1]/android.view.View[1]/android.view.View[1]/android.view.View[1]/android.view.View[1]/android.view.View[1]/android.view.View[1]/android.view.View[6]/android.view.View[1]/android.view.View[2]").click_exists()
time.sleep(0.1)
d.send_keys("{}".format(list(info.loc[info.name == i].care_income)[0]), clear=True)  # 更换成惠农补贴收入
print("惠农补贴填写完毕...")
d.xpath(
    r"//*[@resource-id='app']/android.view.View[1]/android.view.View[2]/android.view.View[1]/android.view.View[1]/android.view.View[1]/android.view.View[1]/android.view.View[1]/android.view.View[1]/android.view.View[1]/android.view.View[7]/android.view.View[1]/android.view.View[1]").click_exists()
time.sleep(0.1)
d.send_keys("{}".format(list(info.loc[info.name == i].other_income)[0]), clear=True)  # 更换成其他收入收入
print("其他收入填写完毕...")
d.xpath(
    r"//*[@resource-id='app']/android.view.View[1]/android.view.View[2]/android.view.View[1]/android.view.View[1]/android.view.View[1]/android.view.View[1]/android.view.View[1]/android.view.View[1]/android.view.View[2]/android.view.View[2]/android.view.View[1]/android.view.View[1]").click_exists()
time.sleep(0.1)
d.send_keys("6", clear=True)  # 更换成持续性经营时间
print("持续时间填写完毕...")
d.xpath(
    r"//*[@resource-id='app']/android.view.View[1]/android.view.View[2]/android.view.View[1]/android.view.View[1]/android.view.View[1]/android.view.View[1]/android.view.View[1]/android.view.View[1]/android.view.View[1]/android.view.View[6]/android.view.View[1]/android.view.View[2]").click_exists()
time.sleep(0.1)
d.send_keys("{}".format(list(info.loc[info.name == i].care_income)[0]), clear=True)  # 更换成惠农补贴收入
d.xpath(r"//*[@text='完成']").click_exists()
print("家庭收入已填写完成...")

# 点击家庭支出信息
d.xpath(
    r"//*[@resource-id='incomeList']/android.view.View[1]/android.view.View[2]/android.view.View[1]").click_exists()
print("开始填写家庭支出信息...")
d.xpath(
    r"//*[@resource-id='app']/android.view.View[1]/android.view.View[2]/android.view.View[1]/android.view.View[1]/android.view.View[1]/android.view.View[1]/android.view.View[1]/android.view.View[1]/android.view.View[1]/android.view.View[3]/android.view.View[1]/android.view.View[2]").click_exists()
time.sleep(0.1)

d.send_keys("{}".format(list(info.loc[info.name == i].arg_expend)[0]), clear=True)  # 更换成种养支出

d.xpath(
    r"//*[@resource-id='app']/android.view.View[1]/android.view.View[2]/android.view.View[1]/android.view.View[1]/android.view.View[1]/android.view.View[1]/android.view.View[1]/android.view.View[1]/android.view.View[1]/android.view.View[4]/android.view.View[1]/android.view.View[2]").click_exists()
time.sleep(0.1)
d.send_keys("{}".format(list(info.loc[info.name == i].busi_expend)[0]), clear=True)  # 更换成经营支出

d.xpath(
    r"//*[@resource-id='app']/android.view.View[1]/android.view.View[2]/android.view.View[1]/android.view.View[1]/android.view.View[1]/android.view.View[1]/android.view.View[1]/android.view.View[1]/android.view.View[1]/android.view.View[5]/android.view.View[1]/android.view.View[2]").click_exists()
time.sleep(0.1)
d.send_keys("{}".format(list(info.loc[info.name == i].med_expend)[0]), clear=True)  # 更换成医疗支出

d.xpath(
    r"//*[@resource-id='app']/android.view.View[1]/android.view.View[2]/android.view.View[1]/android.view.View[1]/android.view.View[1]/android.view.View[1]/android.view.View[1]/android.view.View[1]/android.view.View[1]/android.view.View[6]/android.view.View[1]/android.view.View[2]").click_exists()
time.sleep(0.1)
d.send_keys("{}".format(list(info.loc[info.name == i].edu_expend)[0]), clear=True)  # 更换成教育支出

d.xpath(
    r"//*[@resource-id='app']/android.view.View[1]/android.view.View[2]/android.view.View[1]/android.view.View[1]/android.view.View[1]/android.view.View[1]/android.view.View[1]/android.view.View[1]/android.view.View[1]/android.view.View[7]/android.view.View[1]/android.view.View[2]").click_exists()
time.sleep(0.1)
d.send_keys("{}".format(list(info.loc[info.name == i].ins_expend)[0]), clear=True)  # 更换成参保支出

d.xpath(
    r"//*[@resource-id='app']/android.view.View[1]/android.view.View[2]/android.view.View[1]/android.view.View[1]/android.view.View[1]/android.view.View[1]/android.view.View[1]/android.view.View[1]/android.view.View[1]/android.view.View[8]/android.view.View[1]/android.view.View[1]").click_exists()
time.sleep(0.1)
d.send_keys("{}".format(list(info.loc[info.name == i].other_expend)[0]), clear=True)  # 更换成其他支出

d.xpath(r"//*[@text='完成']").click_exists()

# 住房资产
# 房屋资产不变
d.xpath(
    r"//*[@resource-id='incomeList']/android.view.View[1]/android.view.View[2]/android.view.View[1]").click_exists()
d.xpath(r"//*[@text='完成']").click_exists()
time.sleep(0.1)

# 土地资产
# 点击土地
d.xpath(
    r"//*[@resource-id='incomeList']/android.view.View[1]/android.view.View[2]/android.view.View[1]").click_exists()
# # 点击土地面积
# d.xpath(
#     r"//*[@resource-id='app']/android.view.View[1]/android.view.View[2]/android.view.View[1]/android.view.View[1]/android.view.View[1]/android.view.View[1]/android.view.View[1]/android.view.View[1]/android.view.View[4]/android.view.View[1]/android.view.View[2]").click_exists()
# d.send_keys("{}".format(list(info.loc[info.name == i].field_area)[0]), clear=True)  # 换成土地面积
# # 点击评估价格
# d.xpath(
#     r"//*[@resource-id='app']/android.view.View[1]/android.view.View[2]/android.view.View[1]/android.view.View[1]/android.view.View[1]/android.view.View[1]/android.view.View[1]/android.view.View[1]/android.view.View[8]/android.view.View[1]/android.view.View[2]").click_exists()
# d.send_keys("{}".format(list(info.loc[info.name == i].field_price)[0]), clear=True)  # 换成土地价格
# time.sleep(0.1)
# # 点击完成
d.xpath(r"//*[@text='完成']").click_exists()

# 点击资产设备

d.xpath(
    r"//*[@resource-id='incomeList']/android.view.View[1]/android.view.View[2]/android.view.View[1]").click_exists()
property = list(info.loc[info.name == i].property_price)[0]
if property != 0:
    d.xpath(
        r"//*[@resource-id='app']/android.view.View[1]/android.view.View[2]/android.view.View[1]/android.view.View[1]/android.view.View[1]/android.view.View[1]/android.view.View[1]/android.view.View[1]/android.view.View[2]/android.view.View[1]/android.view.View[3]").click_exists()
    time.sleep(0.1)
    d.swipe(0.571, 0.877, 0.624, 0.647)  # 滑动到私家车区域
    d.xpath(r"//*[@text='私家车']").click_exists()
    d.xpath(
        r"//*[@resource-id='app']/android.view.View[1]/android.view.View[2]/android.view.View[1]/android.view.View[1]/android.view.View[1]/android.view.View[1]/android.view.View[1]/android.view.View[1]/android.view.View[3]/android.view.View[1]/android.view.View[3]").click_exists()
    time.sleep(0.1)
    for i in range(np.random.randint(2, 6)):
        d.swipe(0.554, 0.841, 0.543, 0.957)
    # d.xpath(r"d(text='确认')").click_exists()
    time.sleep(0.5)
    d.click(0.877, 0.662)  # 点击确定

    d.xpath(
        r"//*[@resource-id='app']/android.view.View[1]/android.view.View[2]/android.view.View[1]/android.view.View[1]/android.view.View[1]/android.view.View[1]/android.view.View[1]/android.view.View[1]/android.view.View[4]/android.view.View[1]/android.view.View[2]").click_exists()
    time.sleep(0.1)
    d.send_keys('1',clear=True)  # 点击车辆数量为1
    d.xpath(
        r"//*[@resource-id='app']/android.view.View[1]/android.view.View[2]/android.view.View[1]/android.view.View[1]/android.view.View[1]/android.view.View[1]/android.view.View[1]/android.view.View[1]/android.view.View[5]/android.view.View[1]/android.view.View[2]").click_exists()
    time.sleep(0.1)
    temp = np.random.randint(16, 19) * 10000
    d.send_keys('{}'.format(temp),clear=True)
    d.xpath(
        r"//*[@resource-id='app']/android.view.View[1]/android.view.View[2]/android.view.View[1]/android.view.View[1]/android.view.View[1]/android.view.View[1]/android.view.View[1]/android.view.View[1]/android.view.View[6]/android.view.View[1]/android.view.View[2]").click_exists()
    time.sleep(0.1)
    d.send_keys('{}'.format(temp - 30000),clear=True)
    d.xpath(r"//*[@text='完成']").click_exists()
else:
    time.sleep(0.1)
    d.xpath(r"//*[@text='完成']").click_exists()

d.xpath(r"//*[@text='提交']").click_exists()
time.sleep(1)
d.xpath(r"//*[@text='下一步']").click_exists()
time.sleep(0.1)
d.xpath(r"//*[@text='提交']").click_exists()
time.sleep(0.1)
d.xpath(r"//*[@text='确定']").click_exists()
# 签名
d.swipe(0.161, 0.315, 0.336, 0.31)
d.swipe(0.336, 0.31, 0.315, 0.518)
d.swipe(0.315, 0.518, 0.638, 0.373)
d.swipe(0.638, 0.373, 0.252, 0.384)
d.xpath(r"//*[@text='b8QowAGglgJFw5rjhAAAAAElFTkSuQmCC']").click_exists()
d.xpath(r"//*[@text='确认提交']").click_exists()
# 确认返回村中
time.sleep(0.5)
d.xpath(r"//*[@text='确定']").click_exists()