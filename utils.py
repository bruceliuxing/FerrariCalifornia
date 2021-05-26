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
     fh = logging.handlers.TimedRotatingFileHandler(filename=log_file, when='midnight',
                                                    interval=1, backupCount=10, encoding='UTF-8')
     #fh.setLevel(logging.DEBUG)
     formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
     fh.setFormatter(formatter)
     logger.addHandler(fh)
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

 def md5(string):
     return hashlib.md5(string).hexdigest()
 
 def process_bar(percent, start_str='', end_str='100%', total_length=50):
     bar = ''.join(["\033[31m%s\033[0m"%'   '] * int(percent * total_length)) + ''
     bar = '\r' + start_str + bar.ljust(total_length) + ' {:0>4.4f}%|'.format(percent*100) + end_str
     sys.stdout.write(bar)
     sys.stdout.flush()
 #    print(bar, end='', flush=True)
 
 
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
      
sobel_x_kernel = np.array([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]])
 sobel_y_kernel = np.array([[-1, -2, -1],
                            [0, 0, 0],
                            [1, 2, 1]])
 sobel_45_kernel = np.array([[-1, -2, 0],
                             [-1, 0, 1],
                             [0, 1, 2]])
 sobel_135_kernel = np.array([[0, -1, -2],
                              [1, 0, -1],
                              [2, 1, 0]])
 def calculate_line_formula(endpoint):
     """
     endpoint: np.array([[x1,y1,x2,y2]], dtype=np.float32)
     return: A, B, C
     """
     if isinstance(endpoint, np.ndarray):
         x1, y1, x2, y2 = np.split(endpoint, endpoint.shape[1], axis=1)
     else:
         x1, y1, x2, y2 = endpoint
     #     k = (y2 - y1) / (x2 - x1 + 1e-6)
     #     b = np.mean(np.array([y1, y2]) - np.array([x1, x2]) * k)
     #     return k, b
     A = y2 - y1
     B = x1 - x2
     C = -0.5 * (A * x1 + B * y1 + A * x2 + B * y2)

     return np.hstack((A, B, C))


 def calculate_cross_point(line1, line2):
     """
     line: (k, b)
     return: crossing point corordidate:(x, y)
     """
     #     A1, B1, C1 = line1
     #     A2, B2, C2 = line2
     A1, B1, C1 = line1
     A2, B2, C2 = line2

     x = -1 * (C1 * B2 - C2 * B1) / (A1 * B2 - A2 * B1 + 1e-6)
     y = -1 * (C1 * A2 - C2 * A1) / (B1 * A2 - B2 * A1 + 1e-6)
     return x, y
    
def is_point_in_poly(point, point_list):
     """
     point: input point which needs to be calculate
     point_list: points with clock order of poly
     """
     isum = 0
     point_x, point_y = point
     icount = len(point_list)

     if icount < 3:
         return False
     for i in range(icount):
         p_start_x, p_start_y = point_list[i]
         if i == icount - 1:
             p_end_x, p_end_y = point_list[0]
         else:
             p_end_x, p_end_y = point_list[i+1]
     if ((p_end_y > point_y) and (point_y >= p_start_y)) or ((p_end_y < point_y) and (point_y <= p_start_y)):
         A, B, C = calculate_line_formula((p_start_x, p_start_y, p_end_x, p_end_y))
         if not A == 0:
             point_x_infer = -1 * (B*point_y + C)/A
             if point_x_infer < point_x:
                 isum += 1
     if isum % 2 != 0:
         return True
     else:
         return False

 def point_line_distance(point, line):
     """
     point: (x, y)
     line:  (A, B, C)直线方程参数
     """
     x, y = point
     if isinstance(line, np.ndarray):
         A, B, C = np.split(line, line.shape[1], axis=1)
     else:
         A, B, C = line

     #     return np.abs(A*x + B*y + C) / (np.sqrt(A*A + B*B))

     return np.abs(A*x + B*y + C) / (np.linalg.norm(np.hstack((A, B)), axis=1)[:, np.newaxis] + 1e-6)
  
  
  
  
  def main():
     pass

 if __name__ == "__main__":
     main()
 else:
     print("import module [{}] succ!".format(os.path.join(_curpath, __file__)))
