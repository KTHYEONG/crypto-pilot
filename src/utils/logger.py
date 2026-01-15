import logging
import os
from datetime import datetime

class Logger:
    def __init__(self, name, log_dir):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # 파일 핸들러 (오늘 날짜)
        cur_date = datetime.now().strftime('%Y-%m-%d')
        file_handler = logging.FileHandler(os.path.join(log_dir, f"log_{cur_date}.log"), encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        
        # 콘솔 핸들러
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def get_logger(self):
        return self.logger
