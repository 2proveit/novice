# 爬虫是获取互联网资源的手段
# 需求：用程序模拟浏览器，获取需要的资源或者
# 服务器渲染和客户端渲染
from urllib.request import urlopen  # 敲击回回车按键的行为就是在request

url = 'http://www.baidu.com'
resp = urlopen(url) #响应

# resp.read() # 从resp中读取内容
# print(resp.read().decode("utf-8")) # 将字符串解码

with open("mybaidu.html", mode="w",encoding="utf-8") as f:
    f.write(resp.read().decode("utf-8"))
    print("over!")

# 熟练使用浏览器抓包工具
