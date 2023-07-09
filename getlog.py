# -*- coding: utf-8 -*-
import re
import urllib
import gzip
import os
import requests

url_path = "haihu_url_only2.txt" 
natural_url = 'http://tenhou.net/0/log/?'
archive_url = 'http://e.mjv.jp/0/log/archived.cgi?'
plain_url = 'http://e.mjv.jp/0/log/plainfiles.cgi?'
log_url = 'http://e5.mjv.jp/0/log/?'
mjlog_pass = './log_only2/'
xml_pass = './log_only2/xml/' 

def rewrite_haihu_url(haihu_url):
    if int(haihu_url[18]) == 0:
        rewrite_haihu = haihu_url[:18] + "3" + haihu_url[19:]
    else:
        rewrite_haihu = haihu_url
    
    print(rewrite_haihu)
    return rewrite_haihu


def gz(filename):
    f_in = open(xml_pass + filename + '.xml', 'rb')
    f_out = gzip.open(mjlog_pass + filename + '.mjlog', 'wb')
    f_out.writelines(f_in)
    f_out.close()
    f_in.close()
    # os.remove('./log/' + filename + '.xml')
    return

def download():
    f_url = open(url_path, 'r')
    # f_url = open('all_haihu_url.txt', 'r')
    for i, row in enumerate(f_url):
        print (str(i) + ", " + row)
        # rewrite_row = rewrite_haihu_url(row)
        rewrite_row = row
        urlidand = re.sub(r'.*log\=(.*)\n', r'\1', rewrite_row)
        urlid = re.sub(r'(.*)&.*', r'\1', urlidand)
        # f_in = open(xml_pass + urlidand + '.xml', 'w')
        f_in = open(xml_pass + urlid + '.xml', 'w')
        text = urllib.urlopen(natural_url + urlid).read()
        # print(text)
        if len(text) < 10 :
            text = urllib.urlopen(archive_url + urlid).read()

            if len(text) < 10:
                text = urllib.urlopen(plain_url + urlid).read()

                if len(text) < 10:
                    text = requests.get(log_url + urlid).text

   
        f_in.write(text)
        f_in.close()
        gz(urlid)
        # gz(urlidand)
    f_url.close()
    return

if __name__ == '__main__':
    download()
