import requests
import csv
from lxml import html
import re
#路径
CSV_FILE = r'D:\pytorch\jupyter\data\movies.csv'
BASE_URL = "https://www.themoviedb.org/"
Top_URL_1 = "https://www.themoviedb.org/movie/top-rated"#默认只访问第一页
Top_URL_2 = "https://www.themoviedb.org/discover/movie/items"


#获取电影年份
def get_movie_year(movie_year):
    movie_year = movie_year[0].strip() if movie_year else ''
    return movie_year.replace('(','').replace(')','')

#获取电影上映时间
def get_movie_release_date(data):
    data = data[0].strip() if data else ''
    match = re.search(r'\d{4}-\d{2}-\d{2}', data)
    return match.group() 

#获取电影时长(转为分钟)
def get_movie_runtime(time):
    time = time[0].strip() if time else ''
    h_res = re.search(r'(\d+)h', time)
    m_res = re.search(r'(\d+)m', time)
    h = int(h_res.group(1)) if h_res else 0
    m = int(m_res.group(1)) if m_res else 0
    return h*60 + m


#获取电影详情
def get_movie_info(movie_info_url):
    # 发送请求,获取数据
    response = requests.get(movie_info_url)
    # 解析html文件,将解析结果转换为文档对象
    document = html.fromstring(response.text)
    #返回电影详情 - xpath语法
    #获取电影名字
    movie_name = document.xpath('//*[@id="original_header"]/div[2]/section/div[1]/h2/a/text()')
    movie_year = document.xpath('//*[@id="original_header"]/div[2]/section/div[1]/h2/span/text()')
    data = document.xpath('//*[@id="original_header"]/div[2]/section/div[1]/div/span[@class="release"]/text()')
    tag = document.xpath('//*[@id="original_header"]/div[2]/section/div[1]/div/span[@class="genres"]/a/text()')    
    time = document.xpath('//*[@id="original_header"]/div[2]/section/div[1]/div/span[@class="runtime"]/text()')    
    score = document.xpath('//*[@id="consensus_pill"]/div/div[1]/div/div/@data-percent')
    language = document.xpath('//*[@id="media_v4"]/div/div/div[2]/div/section/div[1]/div/section[1]/p[3]/text()')
    director = document.xpath('//*[@id="original_header"]/div[2]/section/div[3]/ol/li[2]/p[1]/a/text()')
    slogans = document.xpath('//*[@id="original_header"]/div[2]/section/div[3]/h3[1]/text()')
    description = document.xpath('//*[@id="original_header"]/div[2]/section/div[3]/div/p/text()')
    # print(movie_name)
    # print(movie_year)
    # print(data)
    # print(tag)
    # print(time)
    # print(score)
    # print(language)
    # print(director)
    # print(slogans)
    # print(description)

    #返回电影详情
    movie_info = {
        '电影名':movie_name[0].strip() if movie_name else '',
        '电影年份':get_movie_year(movie_year),
        '上映时间':get_movie_release_date(data),
        '电影标签': ','.join(tag) if tag else '',
        '电影时长':get_movie_runtime(time),
        '电影评分':score[0].strip() if score else '',
        '电影语言':language[0].strip() if language else '',
        '电影导演': ','.join(director) if director else '',
        '电影标语':slogans[0].strip() if slogans else '',
        '电影描述':description[0].strip() if description else '',
    }
    # print(movie_info)
    return movie_info

    
#保存电影详情
def save_all_movie(all_movies):
    with open(CSV_FILE, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['电影名','电影年份','上映时间','电影标签','电影时长','电影评分','电影语言','电影导演','电影标语','电影描述'])
        writer.writeheader()
        writer.writerows(all_movies)

def main():
    all_movies = []
    for page in range(1,16):
        # 1. 发送请求,获取数据
        if page==1:
            response = requests.get(Top_URL_1,timeout=60)
        else:
            response = requests.post(Top_URL_2,f'air_date.gte=&air_date.lte=&certification=&certification_country=CN&debug=&first_air_date.gte=&first_air_date.lte=&include_adult=false&include_softcore=false&latest_ceremony.gte=&latest_ceremony.lte=&page={page}&primary_release_date.gte=&primary_release_date.lte=&region=&release_date.gte=&release_date.lte=2027-02-17&show_me=everything&sort_by=vote_average.desc&vote_average.gte=0&vote_average.lte=10&vote_count.gte=300&watch_region=CN&with_genres=&with_keywords=&with_networks=&with_origin_country=&with_original_language=&with_watch_monetization_types=&with_watch_providers=&with_release_type=&with_runtime.gte=0&with_runtime.lte=400',timeout=60)
            
        print(f'第{page}页数据获取完成')
        # 解析html文件,将解析结果转换为文档对象
        document = html.fromstring(response.text)
        # 解析电影列表 - xpath语法
        movie_list = document.xpath('//div[@class="media-list-results contents"]/div')
        # print(movie_list)

        for movie in movie_list:
            movie_urls = movie.xpath('./div/div/a/@href')
            if movie_urls:
                movie_info_url = BASE_URL + movie_urls[0]
                # print(movie_info_url)
                # 发送请求,获取数据
                movie_info=get_movie_info(movie_info_url)
                all_movies.append(movie_info)
    #保存数据为csv文件
    print('开始保存数据为csv文件...')
    save_all_movie(all_movies)
    print(f'共获取到{len(all_movies)}条数据,完成保存为{CSV_FILE}')




if __name__ == '__main__':
    main()