#-*- conding:utf-8 -*-
import argparse
from icrawler.builtin import BaiduImageCrawler,BingImageCrawler,GoogleImageCrawler,GreedyImageCrawler,UrlListCrawler
from tqdm import tqdm
import os

import base64
from six.moves.urllib.parse import urlparse
from icrawler import ImageDownloader
import traceback

def parse_cmd():
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", type=str, default="Baidu", help="Baidu, Bing, Google, UrlList, Greedy")
    parser.add_argument("--input", default=None, help="input")
    parser.add_argument("--save_path", default=None, help="save path")
    parser.add_argument("--max_num", type=int, default=1000, help="max_num")
    parser.add_argument("--save_url", action="store_true", default=False, help="save_url")
    parser.add_argument("--parser_threads", type=int, default=2, help="parser threads")
    parser.add_argument("--downloader_threads", type=int, default=4, help="downloader threads")
    return parser.parse_args()

class PrefixNameDownloader(ImageDownloader):

    def get_filename(self, task, default_ext):
        filename = super(PrefixNameDownloader, self).get_filename(
            task, default_ext)
        return 'prefix_' + filename

def write_url_file(url_file, winfo):
    with open(url_file, "a") as f:
        f.write(winfo)

class Base64NameDownloader(ImageDownloader):

    def get_filename(self, task, default_ext):
        url_path = urlparse(task['file_url'])[2]
        if '.' in url_path:
            extension = url_path.split('.')[-1]
            if extension.lower() not in ['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'ppm', 'pgm']:
                extension = default_ext
        else:
            extension = default_ext
        # works for python 3
        filename = base64.b64encode(url_path.encode()).decode()
        global save_url
        global url_file 
        if save_url: 
            winfo = task['file_url'] + "\n"
            write_url_file(url_file, winfo)
        return '{}.{}'.format(filename, extension)

class UrlNameDownloader(ImageDownloader):

    def get_filename(self, task, default_ext):
        url_path = urlparse(task['file_url'])[2]
        if '.' in url_path:
            extension = url_path.split('.')[-1]
            if extension.lower() not in ['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'ppm', 'pgm']:
                extension = default_ext
        else:
            extension = default_ext
        # works for python 3
        filename = url_path
        global save_url
        global url_file 
        if save_url: 
            winfo = task['file_url'] + "\n"
            write_url_file(url_file, winfo)
        return '{}.{}'.format(filename, extension)
# ref: https://www.jianshu.com/p/839fb07a7aac
def get_registered_func():
    registered_func = {}
    registered_func["Baidu"] = "BaiduImageCrawler"
    registered_func["Bing"] = "BingImageCrawler"
    registered_func["Google"] = "GoogleImageCrawler"
    registered_func["UrlList"] = "UrlListCrawler"
    registered_func["Greedy"] = "GreedyImageCrawler"
    return registered_func

#图片爬虫
def image_crawler(args, keyword, func):
    save_path = os.path.join(args.save_path, keyword)
    storage = {'root_dir': save_path}
    crawler = eval(func)(downloader_cls=Base64NameDownloader,
                               parser_threads=args.parser_threads,
                               downloader_threads=args.downloader_threads, 
                               storage=storage)
    crawler.crawl(keyword=keyword,
                       max_num=args.max_num)

def crawl_image_from_search_engine(args):
    '''
    crawl image from Baidu or Bing or Google
    '''
    infile = args.input
    with open(infile, "r") as f:
        print("\033[32m")
        for line in tqdm(f.readlines()):
            keyword = line.rstrip()
            global url_file 
            url_file = os.path.join(args.save_path, keyword + ".txt")
            print("\033[32mcrawling image with keyword [{}]\033[37m".format(keyword))
            if args.type in registered_func.keys():
                func = registered_func[args.type]
                print("\033[41mcrawl image from [{}]\033[40m".format(func))
                image_crawler(args, keyword, func)
            else:
                print("\033[32mnot support search engine [{}]\033[37m".format(args.type))

            print("\033[32m")

def get_file_list(source_input):
    file_list = []
    if os.path.isfile(source_input):
        file_list.append(source_input)
    elif os.path.isdir(source_input):
        for root, dirs, files in os.walk(source_input): 
            if len(files) > 0:
                for per_file in files:
                    if per_file.find(".txt") != -1:
                        file_list.append(os.path.join(root, per_file))
    return file_list

def crawl_image_from_urllist(args):
    infile = args.input
    print(infile)
    file_list = get_file_list(infile)
    print(len(file_list))
    for per_file in file_list:
        #keyword = os.path.splitext(os.path.basename(per_file))[0]
        #global url_file 
        #url_file = os.path.join(args.save_path, keyword + ".txt")
        #print("\033[32mcrawling image with keyword [{}]\033[37m".format(keyword))
        #save_path = os.path.join(args.save_path, keyword)
        save_path = os.path.abspath(args.save_path)
        storage = {'root_dir': save_path}
        urllist_crawler = UrlListCrawler(downloader_cls=Base64NameDownloader,
                                         downloader_threads=args.downloader_threads, 
                                         storage=storage)
        #输入url的txt文件
        urllist_crawler.crawl(url_list = per_file,
                                max_num = args.max_num)

def crawl_image_greedy(args):
    storage = {'root_dir': args.save_path}
    global url_file 
    url_file = os.path.join(args.save_path, args.input + ".txt")
    #greedy_crawler = GreedyImageCrawler(downloader_cls=Base64NameDownloader,
    greedy_crawler = GreedyImageCrawler(
                                        downloader_threads=args.downloader_threads, 
                                        storage=storage)

    greedy_crawler.crawl(domains=args.input, 
                         max_num=args.max_num)

if __name__ == "__main__":
    registered_func = get_registered_func()
    args = parse_cmd()
    global save_url
    save_url = args.save_url
    if not os.path.exists(args.save_path):
        print(args.save_path)
        os.makedirs(args.save_path)
    if args.type == "UrlList":
        crawl_image_from_urllist(args)
    elif args.type == "Greedy": 
        crawl_image_greedy(args) 
    else:
        crawl_image_from_search_engine(args)
