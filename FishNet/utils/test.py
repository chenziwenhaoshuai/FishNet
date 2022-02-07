import time
print('covert csv to json...')
import csv2json_detect
import csv2json_point
import csv2json_seg
print('covert over,please wait for the file writing...')
time.sleep(5)
print('start coco eval...')
import cocoeval
