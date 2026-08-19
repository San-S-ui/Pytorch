"""
TMDB-TOP300电影榜单数据统计
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from pandas import Series

# 展示中文
plt.rcParams['font.sans-serif'] = ['SimHei']

def load_data():
    """
    加载电影数据
    """
    return pd.read_csv(r'D:\pytorch\jupyter\data\movies.csv', usecols=['电影名','年份','上映时间','类型','时长','评分','语言'], dtype={'年份':'Int64'})

def create_subplots():
    """
    创建子图布局
    """
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 12), dpi=100)
    fig.suptitle('TMDB-TOP300电影榜单数据统计', fontsize=23, x=0.5, y=0.98)
    fig.subplots_adjust(hspace=0.6, wspace=0.2)
    
    axes1: Axes = axes[0][0]
    axes2: Axes = axes[0][1]
    axes3: Axes = axes[1][0]
    axes4: Axes = axes[1][1]
    
    return fig, axes, axes1, axes2, axes3, axes4

def process_year_data(data):
    """
    处理年份数据并生成折线图所需的数据
    """
    # 缺失值处理
    data['年份'] = data['年份'].fillna(data['上映时间'].str[:4])
    
    # 分组统计
    year_count = data.groupby('年份')['年份'].count()
    
    # 组装数据
    min_year = year_count.index.min()
    max_year = year_count.index.max()
    x = [i for i in range(min_year, max_year+1)]
    y = [int(year_count.get(i, 0)) for i in x]
    
    return x, y

def plot_year_trend(axes1, x, y):
    """
    绘制每年电影数量变化折线图
    """
    axes1.plot(x, y, color='green')
    axes1.set_title('每年电影数量变化折线图', fontsize=15)
    axes1.set_xlabel('年份', fontsize=12)
    axes1.set_ylabel('电影数量', fontsize=12)
    
    axes1.set_xticks(x[::8])
    y_ticks = [i for i in range(0,31,3)]
    axes1.set_yticks(y_ticks)
    axes1.grid(linestyle='--', alpha=0.5)

def process_language_data(data):
    """
    处理语言数据并生成柱状图所需的数据
    """
    language_count = data.groupby('语言')['语言'].count().sort_values(ascending=False)
    x_language = language_count.index.tolist()
    y_language_count = language_count.values.tolist()
    return x_language, y_language_count

def plot_language_bar(axes2, x_language, y_language_count):
    """
    绘制不同语言电影数量柱状图
    """
    axes2.bar(x_language, y_language_count, color='green', width=0.7)
    axes2.set_title('不同语言电影数量柱状图', fontsize=15)
    axes2.set_xlabel('语言', fontsize=12)
    axes2.set_ylabel('电影数量', fontsize=12)
    axes2.grid(linestyle='--', alpha=0.5)
    axes2.tick_params(axis='x', rotation=90)

def process_genre_data(data):
    """
    处理类型数据并生成柱状图所需的数据
    """
    type_count = {}
    for types in data['类型'].str.split(','):
        for t in types:
            if t in type_count:
                type_count[t] += 1
            else:
                type_count[t] = 1
    
    x_types = list(type_count.keys())
    y_values = list(type_count.values())
    return x_types, y_values

def plot_genre_bar(axes3, x_types, y_values):
    """
    绘制不同类型电影数量柱状图
    """
    axes3.bar(x_types, y_values, color='green', width=0.7)
    axes3.set_title('不同类型电影数量柱状图', fontsize=15)
    axes3.set_xlabel('类型', fontsize=12)
    axes3.set_ylabel('电影数量', fontsize=12)
    axes3.grid(linestyle='--', alpha=0.5)
    axes3.tick_params(axis='x', rotation=90)

def process_score_data(data):
    """
    处理评分数据并生成饼状图所需的数据
    """
    score_count = data.groupby('评分')['评分'].count()
    
    # 合并小数据(比例 < 2%) --> 其他
    total = score_count.sum()
    large_scores = score_count.loc[score_count >= total*0.02]
    small_scores = score_count.loc[score_count < total*0.02]
    
    if small_scores.shape[0] > 0:
        large_scores['其他'] = small_scores.sum()
    
    scores = large_scores.index.tolist()
    scores_values = large_scores.values.tolist()
    return scores, scores_values

def plot_score_pie(axes4, scores, scores_values):
    """
    绘制不同评分电影数量占比饼状图
    """
    axes4.pie(scores_values, labels=scores, autopct='%1.1f%%', startangle=0, radius=1)
    axes4.set_title('不同评分电影数量占比饼状图', fontsize=15)
    axes4.legend(loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.3))

def main():
    """
    主函数
    """
    # 加载数据
    data = load_data()
    
    # 创建子图
    fig, axes, axes1, axes2, axes3, axes4 = create_subplots()
    
    # 需求一：每年电影数量变化折线图
    x, y = process_year_data(data)
    plot_year_trend(axes1, x, y)
    
    # 需求二：不同语言电影数量柱状图
    x_language, y_language_count = process_language_data(data)
    plot_language_bar(axes2, x_language, y_language_count)
    
    # 需求三：不同类型电影数量柱状图
    x_types, y_values = process_genre_data(data)
    plot_genre_bar(axes3, x_types, y_values)
    
    # 需求四：不同评分电影数量占比饼状图
    scores, scores_values = process_score_data(data)
    plot_score_pie(axes4, scores, scores_values)
    
    # 保存图片
    plt.savefig(r'D:\pytorch\python_ai\resources\TMDB-TOP300.png')
    
    # 显示画布
    plt.show()

if __name__ == '__main__':
    main()