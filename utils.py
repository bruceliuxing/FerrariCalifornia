#!/usr/bin/env python
 #coding:UTF-8

 import os
 import sys
 import cv2
 import requests
 import numpy
 import logging
 import logging.handlers
 import pypinyin
 import traceback
 import base64
 
 if sys.version_info.major == 3:
     from concurrent import futures
 else:
     logging.info("make sure futures module installed! [pip install futures]")
     exit(0)
 assert sys.version_info.major == 3
 assert sys.version_info.minor >= 2, "concurrent.futures must be with higher than python3.2 version!"

 _curpath = os.path.dirname(os.path.abspath(__file__))

 public_datasets_path = os.path.join(_curpath, 'public_datasets')


 def getlogger(log_file):
     """
     initialize logger handle

     logging.basicConfig < handler.setLevel < logger.setLevel
     """
     logger = logging.getLogger()
     logger.setLevel(logging.DEBUG)
     fh = logging.handlers.TimeRotatingFieldHandler(filename=log_file, when='midnight',
                                                    interval=1, backupCount=10, encoding='UTF-8')
     #fh.setLevel(logging.DEBUG)
     formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
     fh.setFormatter(formatter)
     logger.setHandler(fh)
     return logger

 def chinese2pinqin(word):
     """
     transform Chinese charactor to pinyin format
     """
     s = ''
     word_pinyin = pypinyin.pinyin(word, style=pypinyin.NORMAL)
     for i in word_pinyin:
         s += ''.join(i)
     return s
  
  def is_ZH(string):
     """
     '包含汉字的返回TRUE'
     'ord(c_str) > 255 的字符均为中文'
     """
     for c_str in string:
         if '\u4e00' <= c_str <='\u9fa5':
             return True
     return False

 def cv_imread_check(image_path):
     """
     obtain valid image RGB format data
     image_path: [image_path/image_url]
     """
     if image_path.startswith("http"):
         image_bin = requests.get(image_path).content
     else:
         if not os.path.exists(image_path):
             logging.warning("image file not exists! [{}]".format(image_path))
             image_bin = None
         if not os.path.isfile(image_path):
             logging.warning("image path must be a file! [{}]".format(image_path))
             image_bin = None
         if is_ZH(image_path):
             logging.info("image path includes Chinese charactor! \
                          You'd better not next time![{}]".format(image_path))
             image_bin = open(image_path, 'rb').read()
     if image_bin is None:
         logging.info("Fail to read image file [{}]!".format(image_path))
         return None
     image = cv2.imdecode(numpy.frombuffer(image_bin, dtype=numpy.uint8), cv2.IMREAD_COLOR)
     if image.shape[-1] == 4:
         image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
     return image[:,:,::-1]

 def list_groups(init_list, children_list_len=None, children_list_num=None):
     """
     spplit list to sublists

     init_list:          original list
     children_list_len:  length of sublist
     children_list_num:  number of sublist
     """
     if children_list_num is not None:
         if len(init_list) % children_list_num == 0:
             children_list_len = int(len(init_list) // children_list_num)
         else:
             children_list_len = int(len(init_list) // (children_list_num - 1))
     if children_list_len <= 0:
         return init_list
     list_of_groups = zip(*(iter(init_list),) *children_list_len)
     end_list = [list(i) for i in list_of_groups]
     count = len(init_list) % children_list_len
     end_list.append(init_list[-count:]) if count !=0 else end_list
     return end_list
  
  def file2b64(filepath):
     if filepath.startswith("http"):
         b64_data = base64.b64encode(requests.get(filepath).content)
     else:
         b64_data = base64.b64encode(open(filepath, 'rb').read())
     return b64_data
 def base_worker(line_info, server_addr, modelname='button', timeout=10):
     line_list = line_info.strip().split('\t')
     csid, url, imgb64 = line_list[:3]

     tagpaths = ["rcpt_tag.{}".format(modelname)]

     model_output = process_model(url, imgb64, tagpaths, server_addr, timeout)
     output = '\t'.join([csid, url, model_output])
     return output

 def call_back(future):
     result = '\t'.join([future.arg['csid'], future.arg['url']])
     result = result + '\t' + json.dumps({"".format(future.arg['modelname']): []})
     if future.cancelled():
         logging.info('process failed! cancelled! [{}]'.format(future.arg))
     elif future.done():
         error = future.exception()
         if error:
             logging.info('process failed! [{}] return error:{}'.format(future.arg, error))
         else:
             result = future.result()
     #fout_file = os.path.join(_curpath, 'feed_output/{}'.format(future.arg['modelname']), 'data')
     fout_file = sys.argv[2]
     with open(fout_file, 'a+', encoding='UTF-8') as fout:
         fout.write('{}\n'.format(result))

 def thread_worker(ttasks, modelname='button', tworker_num=5, timeout=10):
     server_addrs = get_bns_addrs(TAG_SERVER_ONLINE_BNS)
     with futures.ThreadPoolExecutor(max_workers=tworker_num) as t_executor:
         for ttask in ttasks:
             task_future = t_executor.submit(base_worker, ttask, random.choice(server_addrs), modelname, timeout)
             csid, url = ttask.strip().split('\t')[:2]
             task_future.arg = {'csid':csid, 'url':url, 'modelname':modelname}
             task_future.add_done_callback(call_back)

 def process_worker(ptasks, modelname='button', pworker_num=1, tworker_num=10):
     if pworker_num is None:
         pworker_num = os.cpu_count() or 1
     if len(ptasks) < pworker_num:
         pworker_num = 1
     tasks_lists = list_groups(ptasks, len(ptasks) // pworker_num)

     p_executor = futures.ProcessPoolExecutor(max_workers=pworker_num)
     #p_executor.map(thread_worker, ptasks, chunksize=10)
     task_futures = [p_executor.submit(thread_worker, ptask, modelname, tworker_num) for ptask in tasks_lists]
     p_executor.shutdown(wait=True)
  
class opts(object):
     def __init__(self):
         self.parser = argparse.ArgumentParser()
         self.parser.add_argument('--annotation_path', default='data/annotations', help='标注精灵助手标注结果(json形式导出)目录dir')
         self.parser.add_argument('--classes', default='classes.txt', help='标注数据类别名称文件地址')
         self.parser.add_argument('--out_anno_path', default="annotations", help='输出coco格式json文件')
     def parse(self, args=''):
         """
         opt = opts().parse("--annotation_path crawler_annotation --classes classes.txt --out_anno_path annotations/".split(" "))
         """
         if args == '':
             opt = self.parser.parse_args()
         else:
             opt = self.parser.parse_args(args)
         opt.annotation_path = os.path.join(CUR_PATH, opt.annotation_path)
         opt.classes = os.path.join(CUR_PATH, opt.classes)
         if not os.path.exists(opt.annotation_path):
             print("{} not exists!".format(opt.annotation_path))
         if not os.path.exists(opt.classes):
             print("{} not exists!".format(opt.classes))
         return opt
      
  
  
  
  
  def main():
     pass

 if __name__ == "__main__":
     main()
 else:
     print("import module [{}] succ!".format(os.path.join(_curpath, __file__)))
