from lxml import html

#读取html文件
with open('./resources/仙逆人物志.html', 'r', encoding='utf-8') as f:
    response = f.read()
    #解析html文件,将解析结果转换为文档对象
    doc = html.fromstring(response)
    #解析表头
    th_list = doc.xpath('//table/thead/tr/th/text()')
    print(th_list)
    #解析表格数据,只解析第一行数据
    td_list = doc.xpath('//table/tbody/tr[1]/td/text()')
    print(td_list)
    #获取所有行的数据，每行数据转换为列表存储
    tr_list = doc.xpath('//table/tbody/tr')
    #遍历所有行，获取每行数据
    for tr in tr_list:
        tr_list = tr.xpath('./td/text()')
        print(tr_list)

