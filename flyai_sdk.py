import hashlib
import json
import logging
import os
import platform
import sys
import time
import urllib
import urllib.parse
import uuid
import zipfile
from urllib import parse

import requests

'''
此文件为FlyAI的SDK框架，请参赛选手不要修改本文件，修改之后可能回造成本地和线上训练运行错误，影响比赛成绩。
常见问题：https://doc.flyai.com/question.html
'''
DOMAIN = 'https://www.flyai.com'
DATA_PATH = os.path.join(sys.path[0], 'data', 'input')
MODEL_PATH = os.path.join(sys.path[0], 'data', 'output', 'model')
token = "FlyAI:" + uuid.UUID(int=uuid.getnode()).hex[-12:]
check_list = ['numpy', 'scipy', 'transformers', 'pillow', 'jieba', 'huggingface-hub', 'imgaug',
              'keras-bert', 'nltk', 'pandas', 'scikit-learn', 'text2vec', 'torchvision',
              'tqdm', 'xgboost', 'torchtext', 'lightgbm', 'keras-transformer', 'imageio',
              'opencv-python', 'opencv-python-headless', 'scikit-image']


def get_json(url):
    requests.get(url=url)


def check():
    import pkg_resources
    logging.basicConfig(level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S',
                        format='%(asctime)s flyai.com温馨提示-%(message)s')
    logging.info("设备信息：{}".format(token))
    device_info = {}
    device_info['time'] = time.strftime("%Y-%m-%d %H:%M:%S")
    pythonv = platform.python_version()
    device_info["python_version"] = pythonv
    logging.info("Python版本：{}".format(pythonv))

    python_path = sys.executable
    device_info["python_path"] = python_path
    logging.info("Python路径：{}".format(python_path))

    is_torch = False
    pkg = {}
    for d in pkg_resources.working_set:
        if d.key == 'torch':
            is_torch = True
            logging.info("PyTorch版本：{}=={}".format(d.key, d.version))
            pkg[d.key] = d.version
        if d.key == 'tensorflow' or d.key == 'tensorflow-gpu':
            logging.info("TensorFlow版本：{}=={}".format(d.key, d.version))
            pkg[d.key] = d.version
        if d.key == 'keras':
            logging.info("Keras版本：{}=={}".format(d.key, d.version))
            pkg[d.key] = d.version
        if d.key in check_list:
            pkg[d.key] = d.version
        device_info['pkg_resources'] = pkg
    if is_torch:
        import torch
        if torch.cuda.is_available():
            cudav = torch.version.cuda
            device_info['cuda_version'] = cudav

            cudnnv = torch.backends.cudnn.version()
            device_info['cudnn_version'] = cudnnv

            logging.info("当前「支持」GPU运行环境，CUDA版本为：{}，CUDNN版本为：{}".format(cudav, cudnnv))
        device_info['uname'] = str(platform.uname())
    device_info['token'] = token
    device_info["data_path"] = DATA_PATH
    logging.info("当前比赛数据集路径：{}".format(DATA_PATH))

    device_info["model_path"] = MODEL_PATH
    logging.info("当前比赛模型路径：{}".format(MODEL_PATH))
    with open('./info.json', 'w') as f:
        f.write(json.dumps(device_info) + "\n")

    get_json(DOMAIN + "/train/async_config?device_info=" + json.dumps(device_info))


check()


class FlyAI(object):
    def download_data(self): pass

    def deal_with_data(self): pass

    def train(self): pass

    def load_model(self): pass

    def predict(self, **input_data): pass


class ProgressBar(object):

    def __init__(self, title,
                 count=0.0,
                 run_status=None,
                 fin_status=None,
                 total=100.0,
                 unit='', sep='/',
                 chunk_size=1.0):
        super(ProgressBar, self).__init__()
        self.info = "%s %s %.2f %s %s %.2f %s"
        self.title = title
        self.total = total
        self.count = count
        self.chunk_size = chunk_size
        self.status = run_status or ""
        self.fin_status = fin_status or " " * len(self.status)
        self.unit = unit
        self.seq = sep

    def __get_info(self):
        _info = self.info % (self.title, self.status,
                             self.count / self.chunk_size, self.unit, self.seq,
                             self.total / self.chunk_size, self.unit)
        return _info

    def refresh(self, count=1, status=None, is_print=False):
        self.count += count
        self.status = status or self.status
        if self.count >= self.total:
            self.status = status or self.fin_status
        if is_print:
            show_str = ('[%%-%ds]' % 50) % (
                    int(50 * (self.count / self.total * 100) / 100) * "#")  # 字'符串拼接的嵌套使用
            sys.stdout.write('\r%s %d%%' % (show_str, (self.count / self.total * 100)))
            sys.stdout.flush()


def genearteMD5(str):
    hl = hashlib.md5()
    hl.update(str.encode(encoding='utf-8'))
    return hl.hexdigest()


def data_download(url, data_dir, cache=True, is_print=False):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    path = os.path.join(data_dir, os.path.basename(url))
    if 'http' in url:
        result = parse.urlparse(url)
        result = os.path.basename(result.path[1:])

        cache_path = os.path.join(DATA_PATH, result)
        if os.path.exists(cache_path):
            file_size = os.path.getsize(cache_path)
        else:
            file_size = 0
        try:
            r = requests.get(url, stream=True)
        except:
            return path
        chunk_size = 1024  # 单次请求最大值
        content_size = int(r.headers['content-length'])  # 内容体总大小
        if content_size == file_size and cache:
            pass
        else:
            if is_print:
                if content_size < 1024 * 1024:
                    print(time.strftime(
                        "%Y-%m-%d %H:%M:%S") + " flyai.com温馨提示-正在下载 " + result + "总大小: " + str(
                        content_size // 1024) + 'KB')

                else:
                    print(time.strftime(
                        "%Y-%m-%d %H:%M:%S") + " flyai.com温馨提示-正在下载 " + result + " 总大小: " + str(
                        content_size // 1024 // 1024) + 'MB')
            progress = ProgressBar(result, total=content_size,
                                   unit="MB", chunk_size=chunk_size * 1024,
                                   run_status="downloading",
                                   fin_status="Download completed")
            with open(cache_path, "wb") as data:
                for chunk in r.iter_content(chunk_size=1024):
                    if chunk:
                        data.write(chunk)
                        if platform.system().lower() != 'linux':
                            progress.refresh(count=len(chunk), is_print=is_print)
        if 'zip' in os.path.basename(url) and not os.path.exists(
                os.path.join(data_dir, genearteMD5(result))) or not cache:
            un_zip(cache_path, data_dir)
            file = open(os.path.join(data_dir, "." + genearteMD5(result)), 'w')
            file.write("success")
            file.close()
    if 'zip' in os.path.basename(url):
        return data_dir
    elif 'http' not in url:
        return os.path.join(data_dir, url)
    else:
        return path


def un_zip(file_name, data_dir):
    zip_file = zipfile.ZipFile(file_name)
    if not os.path.exists(file_name):
        os.makedirs(file_name)
    for names in zip_file.namelist():
        zip_file.extract(names, data_dir)
    zip_file.close()


# 下载预训练模型
def download_model(model_url):
    logging.info("正在下载模型...")
    model_path = os.path.join(DATA_PATH, "model")
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    file_path = os.path.join(model_path, os.path.basename(model_url))
    r = requests.get(model_url)
    with open(file_path, "wb") as model:
        model.write(r.content)
    logging.info("模型下载成功，请到对应目录查看。")
    if 'zip' in os.path.basename(model_url):
        un_zip(file_path, model_path)
    return file_path


class DataHelper:

    def download_from_ids(self, *ids, log=True):
        '''
        下载想要使用的数据，会下载到data/input/id/下面，数据下载成功之后会更新项目json
        '''
        try:
            for id in ids:
                self.download_id(id, log)
        except requests.exceptions.ConnectionError:
            logging.error("网络无法访问，使用本地现有数据。如有疑问可扫码添加FlyAI小助手微信解决问题。")

    def download_id(self, id, log):
        data = self.get_json(DOMAIN + "/data/" + id + "?token=" + token)
        data = data['data']
        data_download(urllib.parse.unquote(data['data_url']), os.path.join(DATA_PATH, id),
                      is_print=log)
        if log:
            if 'name' in data:
                logging.info(
                    "{}数据已经下载到{}文件夹中。".format(data['name'], os.path.join(DATA_PATH, id, '')))
            else:
                logging.info("数据已经下载到{}文件夹中。".format(os.path.join(DATA_PATH, id, '')))

            logging.info("本地训练仅提供部分数据，提交到线上训练会提供全量数据。")
            if 'source' in data and data['source'] != "":
                logging.info("数据集详细信息请访问:{}".format(data['source']))
        logging.info("读取数据和模型请使用相对路径:from flyai_sdk import DATA_PATH MODEL_PATH")
        logging.info("项目中【main.py】为模型训练文件，参赛者可以自行读取数据、编写网络、训练模型等。")
        logging.info("项目中【prediction.py】为线上评估预测使用，参赛者需要根据规范，实现评估代码，线上才能跑出成绩。")
        logging.info("项目中【requirements.txt】为项目缩用到的python依赖，本地安装后请及时添加到给文件中，否则线上训练可能报错。")
        logging.info(
            "遇到问题不要着急，添加FlyAI技术客服微信【flyaixzs】，为您在线解答问题，或扫描项目里面的：FlyAI小助手二维码-FlyAI技术客服线解答您的问题.png")
        return data

    def get_json(self, url):
        response = requests.get(url=url)
        return response.json()
