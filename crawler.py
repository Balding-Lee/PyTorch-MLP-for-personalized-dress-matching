"""
Author: Qizhi Li
爬虫
"""
from selenium import webdriver
import re
import pandas as pd
import pyprind


# url = 'http://www.tianqihoubao.com/lishi/chengdu/month/202001.html'
User_Agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.169 Safari/537.36'

opt = webdriver.ChromeOptions()
# opt.set_headless()  #无界面模式
opt.add_argument('user-agent=%s' % User_Agent)
opt.add_argument('log-level=3')
nopicture = {'profile.managed_default_content_settings.images': 2}  # 无图片模式
opt.add_experimental_option('prefs', nopicture)
chromedriver = r'C:\Users\Liqz\AppData\Local\Google\Chrome\Application\chromedriver.exe'


def chrome():
    driver = webdriver.Chrome(chromedriver, options=opt)
    driver.implicitly_wait(3)  # 隐式等待是设置全局的查找页面元素的等待时间，在这个时间内没找到指定元素则抛出异常，只需设置一次
    driver.set_window_size(1366, 768)
    driver.set_page_load_timeout(60)  # 这个应该设置得足够大
    # driver.set_script_timeout(6) #据说上一条指令要和这一条配合才行
    # executor_url=driver.command_executor._url
    # session_id=driver.session_id
    return driver


driver = chrome()

url_head = 'http://www.tianqihoubao.com/lishi/chengdu/month/'
# 需要爬取的数据
months = ['202001', '202002', '202003', '202004', '202005', '202006', '202007', '202008',
          '202009', '202010', '202011', '202012', '202101', '202102', '202103', '202104',
          '202105', '202106', '202107', '202108', '202109', '202110']
# not_number = re.compile('^[0-9]')

all_data = []

pper = pyprind.ProgPercent(len(months))
for month in months:
    url = url_head + month + '.html'
    driver.get(url)  # 获取网页
    trs = driver.find_elements_by_css_selector('table[class="b"] > tbody > tr')  # 获得页面的tr数据

    for i, tr in enumerate(trs):
        # 跳过第一行
        if i == 0:
            continue

        data_one_day = []
        tds = tr.find_elements_by_tag_name('td')
        date = tds[0].text

        weather = tds[1].text.split(' /')  # [weather_morning, weather_night]
        weather_morning = weather[0]
        weather_night = weather[1]

        temperature = tds[2].text.split(' / ')  # [temperature_morning, temperature_night]
        # 我也不知道为什么'^[0-9]'没法清理这个字符, 简直离谱
        temperature_morning = re.sub('℃', '', temperature[0])
        temperature_night = re.sub('℃', '', temperature[1])

        data_one_day.append(date)
        data_one_day.append(weather_morning)
        data_one_day.append(weather_night)
        data_one_day.append(temperature_morning)
        data_one_day.append(temperature_night)

        all_data.append(data_one_day)
    pper.update()

df = pd.DataFrame(all_data, columns=['日期', '早晨天气', '晚间天气', '最高气温', '最低气温'])
df.to_csv('./data/data.csv', encoding='utf_8_sig', index=False)

















