/usr/bin/bash

PYTHON=$(which python)
#download from baidu
#${PYTHON} crawl_image_new.py --type "Baidu" --input brand_vehicle_watch.txt --save_path 'data/vehicle/images/' --max_num 500 --save_url

#download form urlfile
${PYTHON} crawl_image_new.py --type "UrlList" --input data/vehicle/sample*.txt --save_path 'data/vehicle/images/images/' --max_num 200000
