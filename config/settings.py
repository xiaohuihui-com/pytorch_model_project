import os

# 🍚自定义日志的输出格式
formatter1_format = '[%(asctime)s] %(filename)s:%(lineno)d %(levelname)s: %(message)s'
formatter2_format = '[%(asctime)s] : %(message)s'

# 🍚通过变量的方式存放路径,也可以使用"os.path"来规范路径
# logfile_path1 = r'F:\Pycharm File\PycharmProjects\python正课\day18\a1.log'  # log文件名

logfile_path1 = os.path.join(os.path.dirname(__file__), 'result.log')  # log文件名


# 🍚log配置字典, 里面就是上面提到的四种对象
LOGGING_DIC = {
    'version': 1,  # 指定版本信息
    'disable_existing_loggers': False,  # 关闭已存在日志。默认False
    'datefmt': '%Y-%m-%d %H:%M:%S %p',  # 时间格式
    'formatters': {  # 固定格式不能修改
        "formatter1": {  # 开头自定义的日志输出格式名
            'format': formatter1_format  # "format" 固定格式不能修改
        },
        'formatter2': {
            'format': formatter2_format
        },
    },
    'filters': {},
    'handlers': {
        'file1_hanlder': {  # 自定义"handlers"名字,可以改
            'level': 'DEBUG',  # 日志过滤等级
            'class': 'logging.FileHandler',  # 保存到文件里面去(日志保存的形式)
            'formatter': 'formatter1',  # 绑定的日志输出格式
            'filename': logfile_path1,  # 制定日志文件路径
            'encoding': 'utf-8',  # 日志文件的编码，不再担心乱码问题
        },
        'terminal': {  # 自定义的"handlers"名字(终端)
            'level': 'DEBUG',  # 日志过滤等级
            'class': 'logging.StreamHandler',  # 打印到屏幕
            'formatter': 'formatter2'  # 日志输出格式
        },
    },
    #    🔰负责生产日志
    'loggers': {
        # '' 代表默认的,在执行'logging.getLogger("key")'时,在"loggers"里面没有找到这个"key"时就使用这个
        '': {
            # 这里把上面定义的两个handler都加上，即log数据既写入文件又打印到屏幕
            'handlers': ['file1_hanlder', 'terminal'],
            'level': 'INFO',
            'propagate': False,  # 向上(更高level的logger)传递,默认True, 通常设置为False
        },
        # 在执行'logging.getLogger("key")'时,在"loggers"里面找到这个"key"时就使用这个
        'terminal': {
            'handlers': ['terminal'],
            'level': 'INFO',
            'propagate': False,
        },
        'file': {
            'handlers': ['file1_hanlder'],
            'level': 'INFO',
            'propagate': False,
        },
    }
}
if __name__ == '__main__':
    import logging.config

    logging.config.dictConfig(LOGGING_DIC)
    logger = logging.getLogger('')  # 执行后会去日志字典的"logger"里面找这个'自定义日志名1'(key)
    logger.info('logging.config.dictConfig')
