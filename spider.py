import time

import requests
import json
import re
import csv
import os

from pandas.core.reshape.util import tile_compat

header = {
    "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
}


def append_to_csv(file_path, title, score, scoreCount, styles, actors):
    title = title.replace(","," ")
    # 检查文件是否存在，不存在则创建
    if not os.path.exists(file_path):
        with open(file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            # 写入表头（可选，如果文件已存在且包含表头则不需要）
            # writer.writerow(["title", "styles"])
            pass  # 文件不存在但不需要立即写入数据，因为我们要追加
    # 以追加模式打开文件
    with open(file_path, mode='a', newline='', encoding='utf-8') as file:
        print(title + "," + score + "," + scoreCount + "," + styles.replace('"', '')+"," + actors)
        file.write(title + "," + score + "," + scoreCount + "," + styles.replace('"', '')+"," + actors + "\n")


def parseStyleFromHtml(html):
    # 正则表达式模式
    # 这个模式匹配方括号内的所有内容，包括逗号和引号
    pattern1 = r'"styles":\[(.*?)\]'
    # 使用re.search找到匹配项
    match1 = re.search(pattern1, html)
    # 如果找到了匹配项，提取组内容
    if match1:
        # 提取整个方括号内的内容（包括逗号和引号）
        return "[" + match1.group(1).replace(",", "|") + "]"
    else:
        print("没有找到匹配项")
        raise ValueError("无标签")
def remove_empty_items(input_list):
    # 使用列表推导式来过滤掉空字符串项
    result_list = [item for item in input_list if item != '']
    return result_list
def process_list(input_list):
    # 遍历列表，创建一个新的列表来存储修改后的结果
    result = []
    for item in input_list:
        # 检查是否以 ':' 结尾
        if item.endswith('：'):
            continue  # 直接跳过该项，相当于删除
        # 检查是否包含 ':' 并且分割为两部分
        elif '：' in item:
            parts = item.split('：')
            if len(parts) == 2:
                result.append(parts[1])
            else:
                # 如果分割后不是两部分，可以选择直接保留原项或做其他处理
                # 这里我们选择保留原项
                result.append(item)
        else:
            # 如果没有 ':'，直接保留原项
            result.append(item)
    return remove_empty_items(result)
def parseActorFromHtml(html):
    # 正则表达式模式
    # 这个模式匹配方括号内的所有内容，包括逗号和引号
    pattern1 = r'<div class="mediainfo_mediaDesc__jjRiB" title="(.*?)">'
    # 使用re.search找到匹配项
    match1 = re.search(pattern1, html)
    # 如果找到了匹配项，提取组内容
    if match1:
        act_str = match1.group(1)
        act_list = process_list(act_str.split(' '))
        return "["+'|'.join(act_list) + "]"
    else:
        print("没有找到匹配项")
        raise ValueError("无配音演员")


def parseScoreCountFromHtml(html):
    # 正则表达式模式
    # 这个模式匹配方括号内的所有内容，包括逗号和引号
    pattern = r'<div class="mediainfo_ratingText__N8GtM">(.*?)人评分</div>'
    # 使用re.search找到匹配项
    match = re.search(pattern, html)
    # 如果找到了匹配项，提取组内容
    if match:
        # 提取整个方括号内的内容（包括逗号和引号）
        match_str = match.group(1)
        if "万" in match_str:
            num = float(match_str[:-1]) * 10000
            return str(num)[:-2]
        else:
            return match_str
    else:
        print("没有找到匹配项")
        raise ValueError("无评分")


def getDataListFormReqByPage(pagesize, page):
    params = {
        "pagesize": pagesize,
        "page": page
    }
    html = requests.get(
        "https://api.bilibili.com/pgc/season/index/result?st=1&year=-1&season_version=-1&spoken_language_type=-1&area=-1&is_finish=-1&copyright=-1&season_status=-1&season_month=-1&style_id=-1&order=3&sort=0&season_type=1&type=1",
        headers=header, params=params)
    html = html.text
    jsonData = json.loads(html)
    return jsonData['data']['list']


if __name__ == '__main__':
    for i in range(4):
        for one in getDataListFormReqByPage(1000, i + 1):
            detailHtml = requests.get(one['link'], headers=header).text
            time.sleep(0.1)
            try:
                append_to_csv("data.csv", one['title'], one['score'], parseScoreCountFromHtml(detailHtml),
                              parseStyleFromHtml(detailHtml), parseActorFromHtml(detailHtml))
            except Exception as e:
                print(str(e))
                continue
