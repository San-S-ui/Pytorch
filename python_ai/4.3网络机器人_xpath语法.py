"""
xpath语法
/  从根节点开始选择 /html/body/div/h1
//  从当前节点开始选择 //h1
.  从当前节点开始选择 ./a
[n]  选择第n个元素 //p[2]
[last()]  选择最后一个元素 //p[last()]
[@attr]  选择包含指定属性的元素 //p[@color]
[@attr="value"]  选择包含指定属性值的元素 //p[@color="red"]
*  选择所有元素 //body/div/*
@*  选择元素的所有属性 //body/div/a/@*
text()  选择元素的文本内容 //h1/text()

"""
from lxml import html

#读取html文件
with open('./resources/仙逆人物志.html', 'r', encoding='utf-8') as f:
    response = f.read()
    #解析html文件,将解析结果转换为文档对象
    document = html.fromstring(response)
     # 解析表头 - xpath语法
    # /table/thead/tr/th/text() : 表示从根节点开始匹配
    # //table/thead/tr/th/text(): 从任意位置开始匹配
    # th_list = document.xpath("/html/body/div/div/table/thead/tr/th/text()")
    # th_list = document.xpath("//table/thead/tr/th/text()")
    th_list = document.xpath("//thead/tr/th/text()")
    print(th_list)

    # tr[2] : 表示匹配第2个tr标签
    td_list = document.xpath("//tbody/tr[2]/td/text()")
    print(td_list)

    # last() : 表示匹配最后一个
    td_list = document.xpath("//tbody/tr[last()-1]/td/text()")
    print(td_list)

    # p[@class]: 表示匹配class属性为p的标签
    p_list = document.xpath("//p[@class]/text()")
    print(p_list)

    # p[@class='xn']: 表示匹配class属性为xn的p标签
    p_list = document.xpath("//p[@class='xn']/text()")
    print(p_list)

    # * : 表示匹配任意标签
    th_list = document.xpath("//thead/tr/*/text()")
    print(th_list)

    # @src: 表示匹配src属性
    # @* : 表示匹配任意属性
    # a_list = document.xpath("//td/img/@src")
    a_list = document.xpath("//td/img/@*")
    print(a_list)

