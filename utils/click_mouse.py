# -*- coding: utf-8 -*-
# time: 2023/12/30 14:50
# file: move_mouse.py
# author: Shuai
# usage: click mouse to keep pycharm active


import pyautogui
import time
import sys


def move_and_click_square():
    click_count = 0  # 移动次数计数器

    try:
        while True:
            # 模拟点击左键
            pyautogui.click()

            # 更新移动次数信息并打印
            click_count += 1
            sys.stdout.write(f"\r点击次数: {click_count}")
            sys.stdout.flush()

            # 在目标位置停留一分钟
            time.sleep(10)

    except KeyboardInterrupt:
        print("\n程序结束.")


if __name__ == "__main__":
    move_and_click_square()




