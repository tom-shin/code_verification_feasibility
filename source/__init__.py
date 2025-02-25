import time
import traceback
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import sys
import os
import subprocess
import json
import chardet
import shutil
import re
import itertools
import platform
import uuid
import logging
import threading
import openai
from openai._exceptions import OpenAIError, AuthenticationError
import httpx
import requests
from collections import OrderedDict
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, Future
# from reportlab.pdfgen import canvas

from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtCore import pyqtSignal, QTimer, Qt, QThread, QObject, QModelIndex
from PyQt5.QtGui import QSyntaxHighlighter, QTextCharFormat, QColor
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QProgressBar, QPushButton, QHBoxLayout, QSpacerItem, \
    QSizePolicy, QRadioButton, QWidget, QMessageBox, QFileDialog, QApplication, QFileSystemModel

from langchain_community.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import (
    PythonLoader, NotebookLoader, TextLoader, JSONLoader, TextLoader, UnstructuredFileLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter

logging.basicConfig(level=logging.INFO)

EnablePrint = False


def PRINT_(*args):
    if EnablePrint:
        logging.info(args)


# ANSI 코드 정규식 패턴 (터미널 컬러 코드)
ANSI_ESCAPE = re.compile(r'\x1B[@-_][0-?]*[ -/]*[@-~]')


def load_module_func(module_name):
    mod = __import__(f"{module_name}", fromlist=[module_name])
    return mod


def json_dump_f(file_path, data, use_encoding=False):
    if file_path is None:
        return False

    if not file_path.endswith(".json"):
        file_path += ".json"

    if use_encoding:
        with open(file_path, 'rb') as f:
            result = chardet.detect(f.read())
            encoding = result['encoding']
    else:
        encoding = "utf-8"

    with open(file_path, "w", encoding=encoding) as f:
        json.dump(data, f, indent=4, ensure_ascii=False, sort_keys=False)

    return True


def json_load_f(file_path, use_encoding=False):
    if file_path is None:
        return False, False

    if use_encoding:
        with open(file_path, 'rb') as f:
            result = chardet.detect(f.read())
            encoding = result['encoding']
    else:
        encoding = "utf-8"

    with open(file_path, "r", encoding=encoding) as f:
        json_data = json.load(f, object_pairs_hook=OrderedDict)

    return True, json_data


# def load_markdown(data_path):
#     with open(data_path, 'rb') as f:
#         result = chardet.detect(f.read())
#         encoding = result['encoding']
#
#     headers_to_split_on = [
#         ("#", "Header 1"),
#         ("##", "Header 2"),
#         ("###", "Header 3"),
#         ("####", "Header 4"),
#     ]
#     markdown_splitter = MarkdownHeaderTextSplitter(
#         headers_to_split_on=headers_to_split_on
#     )
#
#     with open(data_path, 'r', encoding=encoding) as file:
#         PRINT_(data_path)
#         data_string = file.read()
#         documents = markdown_splitter.split_text(data_string)
#
#         # 파일명을 metadata에 추가
#         domain = data_path  # os.path.basename(data_path)
#         for doc in documents:
#             if not doc.metadata:
#                 doc.metadata = {}
#             doc.metadata["domain"] = domain  # Document 객체의 metadata 속성에 파일명 추가
#
#         return documents


# def load_txt(data_path):
#     with open(data_path, 'rb') as f:
#         result = chardet.detect(f.read())
#         encoding = result['encoding']
#
#     text_splitter = CharacterTextSplitter(
#         separator="\n",
#         length_function=len,
#         is_separator_regex=False,
#     )
#
#     with open(data_path, 'r', encoding=encoding) as file:
#         data_string = file.read().split("\n")
#         domain = data_path  # os.path.basename(data_path)
#         documents = text_splitter.create_documents(data_string)
#
#         for doc in documents:
#             if not doc.metadata:
#                 doc.metadata = {}
#             doc.metadata["domain"] = domain  # Document 객체의 metadata 속성에 파일명 추가
#
#         return documents


# def load_general(base_dir):
#     data = []
#     cnt = 0
#     for root, dirs, files in os.walk(base_dir):
#         for file in files:
#             if file.endswith(".txt"):
#                 file_path = os.path.join(root, file)
#                 if os.path.getsize(file_path) > 0:
#                     cnt += 1
#                     data += load_txt(file_path)
#
#     PRINT_(f"the number of txt files is : {cnt}")
#     return data


# def load_document(base_dir):
#     data = []
#     cnt = 0
#     for root, dirs, files in os.walk(base_dir):
#         for file in files:
#             if file.endswith(".md"):
#                 file_path = os.path.join(root, file)
#                 if os.path.getsize(file_path) > 0:
#                     cnt += 1
#                     data += load_markdown(file_path)
#
#     PRINT_(f"the number of md files is : {cnt}")
#     return data


# def get_markdown_files(source_dir):
#     dir_ = source_dir
#     loader = DirectoryLoader(dir_, glob="**/*.md", loader_cls=UnstructuredMarkdownLoader)
#     documents = loader.load()
#     PRINT_("number of doc: ", len(documents))
#     return documents


def CheckDir(dir_):
    if os.path.exists(dir_):
        shutil.rmtree(dir_)  # 기존 디렉터리 삭제

    os.makedirs(dir_)  # 새로 생성


def GetCurrentDate():
    current_date = datetime.now()

    # 날짜 형식 지정 (예: YYYYMMDD)
    formatted_date = current_date.strftime("%Y%m%d")

    return formatted_date


def save2html(file_path, data, use_encoding=False):
    if file_path is None:
        return False

    if not file_path.endswith(".html"):
        file_path += ".html"

    if use_encoding:
        with open(file_path, 'rb') as f:
            result = chardet.detect(f.read())
            encoding = result['encoding']
    else:
        encoding = "utf-8"

    with open(file_path, "w", encoding=encoding) as f:
        f.write(data)

    # PRINT_("Saved as HTML text.")


def save2txt(file_path, data, use_encoding=False):
    if file_path is None:
        return False

    if not file_path.endswith(".txt"):
        file_path += ".txt"

    if use_encoding:
        with open(file_path, 'rb') as f:
            result = chardet.detect(f.read())
            encoding = result['encoding']
    else:
        encoding = "utf-8"

    with open(file_path, "w", encoding=encoding) as f:
        f.write(data)

    # PRINT_("Saved as TEXT.")


def check_environment():
    env = ''
    system = platform.system()
    if system == "Windows":
        # Windows인지 확인, WSL 포함
        if "microsoft" in platform.version().lower() or "microsoft" in platform.release().lower():
            env = "WSL"  # Windows Subsystem for Linux
        env = "Windows"  # 순수 Windows
    elif system == "Linux":
        # Linux에서 WSL인지 확인
        try:
            with open("/proc/version", "r") as f:
                version_info = f.read().lower()
            if "microsoft" in version_info:
                env = "WSL"  # WSL 환경
        except FileNotFoundError:
            pass
        env = "Linux"  # 순수 Linux
    else:
        env = "Other"  # macOS 또는 기타 운영체제

    # PRINT_(env)
    return env


def user_subprocess(cmd=None, run_time=False, timeout=None, log=True, shell=True):
    line_output = []
    error_output = []
    timeout_expired = False

    if sys.platform == "win32":
        # WSL 명령으로 변환
        if not shell:
            cmd.insert(0, "wsl")
        else:
            cmd = rf"wsl {cmd}"

    if run_time:
        try:
            with subprocess.Popen(cmd, shell=shell, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                  text=True) as process:
                while True:
                    # stdout, stderr에서 비동기적으로 읽기
                    line = process.stdout.readline()
                    if line:
                        line_output.append(line.strip())
                        cleaned_sentence = ANSI_ESCAPE.sub('', line)

                        if log:
                            PRINT_(cleaned_sentence.strip())  # 실시간 출력
                        if "OPTYPE : DROPOUT" in line:
                            if log:
                                PRINT_("operror")

                    err_line = process.stderr.readline()
                    if err_line:
                        error_output.append(err_line.strip())
                        cleaned_sentence = ANSI_ESCAPE.sub('', err_line)
                        if log:
                            PRINT_("ERROR:", cleaned_sentence.strip())

                    # 프로세스가 종료되었는지 확인
                    if process.poll() is not None and not line and not err_line:
                        break

                # 프로세스 종료 코드 체크
                process.wait(timeout=timeout)

        except subprocess.TimeoutExpired:
            process.kill()
            if log:
                PRINT_("Timeout occurred, process killed.")
            error_output.append("Process terminated due to timeout.")
            timeout_expired = True

    else:
        try:
            result = subprocess.run(cmd, shell=shell, capture_output=True, text=False, timeout=timeout)

            # Decode stdout and stderr, handling encoding issues
            encoding = "utf-8"
            errors = "replace"
            if result.stdout:
                line_output.extend(result.stdout.decode(encoding, errors).splitlines())
            if result.stderr:
                error_output.extend(result.stderr.decode(encoding, errors).splitlines())

            if log:
                for line in line_output:
                    cleaned_sentence = ANSI_ESCAPE.sub('', line)
                    PRINT_(cleaned_sentence)  # 디버깅을 위해 주석 해제

                for err_line in error_output:
                    cleaned_sentence = ANSI_ESCAPE.sub('', err_line)
                    PRINT_("ERROR:", cleaned_sentence)  # 에러 메시지 구분을 위해 prefix 추가

        except subprocess.TimeoutExpired:
            if log:
                PRINT_("Timeout occurred, command terminated.")
            error_output.append("Command terminated due to timeout.")
            timeout_expired = True

        except Exception as e:
            # 기타 예외 처리 추가
            if log:
                PRINT_(f"subprocess Command exception failed:")
            error_output.append(f"subprocess Command exception failed:")

    return line_output, error_output, timeout_expired


def Open_QMessageBox(message="", yes_b=True, no_b=True):
    msg_box = QMessageBox()
    msg_box.setWindowTitle("Information")
    msg_box.setText(message)

    if yes_b and no_b:
        msg_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
    elif yes_b and not no_b:
        msg_box.setStandardButtons(QMessageBox.Yes)
    elif not yes_b and no_b:
        msg_box.setStandardButtons(QMessageBox.No)
    else:
        msg_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)

    # Always show the message box on top
    msg_box.setWindowFlags(msg_box.windowFlags() | Qt.WindowStaysOnTopHint)

    # 메시지 박스를 최상단에 표시
    answer = msg_box.exec_()

    if answer == QMessageBox.Yes:
        return True
    else:
        return False


def check_for_specific_string_in_files(directory, check_keywords):
    check_files = []  # 에러가 발견된 파일을 저장할 리스트
    context_data = {}

    # 디렉터리 내의 모든 파일 검사
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)

        # 파일인지 확인
        if os.path.isfile(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    # 파일을 줄 단위로 읽으면서 키워드 확인
                    for line in file:
                        if any(re.search(keyword, line) for keyword in check_keywords):
                            check_files.append(filename)  # 에러가 발견된 파일 추가
                            break  # 한 번 발견되면 해당 파일에 대한 검사는 종료
            except Exception as e:
                PRINT_(f"Error reading file {file_path}: {e}")

    return check_files, context_data


# 에러가 발생하면 해당 에러 내용까지 추출하는 구조
def upgrade_check_for_specific_string_in_files(directory, check_keywords):
    check_files = []  # 에러가 발견된 파일 목록을 저장할 리스트
    context_data = {}  # 파일별로 키워드 발견 시 해당 줄 주변 내용을 저장할 딕셔너리

    # 디렉터리 내의 모든 파일 검사
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)

        # 파일인지 확인
        if os.path.isfile(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    lines = file.readlines()  # 파일 내용을 줄 단위로 모두 읽음

                for i, line in enumerate(lines):
                    # 키워드가 현재 줄에 포함되어 있는지 확인
                    if any(re.search(keyword, line) for keyword in check_keywords):
                        check_files.append(filename)  # 에러가 발견된 파일 추가

                        # "Traceback (most recent call last)" 라인을 위쪽으로 탐색
                        start_index = 0
                        for j in range(i, -1, -1):  # 발견된 줄부터 역방향 탐색
                            if "Traceback (most recent call last)" in lines[j]:
                                start_index = j
                                break

                        # 아래쪽으로는 발견된 줄의 다음 2줄까지 포함
                        # end_index = min(len(lines), i + 2)
                        # 아래 방향으로 "Command:"가 나오기 전까지 포함
                        end_index = i
                        for j in range(i + 1, len(lines)):
                            if "Command:" in lines[j]:
                                end_index = j
                                break
                        else:
                            end_index = len(lines)  # "Command:"가 없으면 파일 끝까지

                        # 각 라인의 끝에 줄바꿈 추가
                        context = [line + "\n" if not line.endswith("\n") else line for line in
                                   lines[start_index:end_index]]

                        # 파일 이름을 키로 사용하여 해당 내용 저장
                        if filename not in context_data:
                            context_data[filename] = []
                        context_data[filename].append(''.join(context))
                        break  # 한 번 발견되면 해당 파일에 대한 검사는 종료

            except Exception as e:
                PRINT_(f"Error reading file {file_path}: {e}")

    return check_files, context_data


def remove_dir(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)


def remove_alldata_files_except_specific_extension(directory, extension):
    # 주어진 디렉토리 내 모든 파일과 서브디렉토리 확인
    for root, dirs, files in os.walk(directory, topdown=False):
        for name in files:
            file_path = os.path.join(root, name)

            if name.endswith('.caffemodel') or name.endswith('.prototxt'):  # or name.endswith('.protobin'):
                continue

            elif name.endswith(f'.{extension}'):
                continue

            try:
                os.remove(file_path)
                PRINT_(f"삭제됨: {file_path}")
            except Exception as e:
                PRINT_(f"파일 삭제 실패: {file_path}, 이유: {e}")

        for name in dirs:
            # 서브디렉토리는 파일이 모두 삭제된 후에 삭제
            dir_path = os.path.join(root, name)
            shutil.rmtree(dir_path)  # 디렉토리 삭제


def separate_folders_and_files(directory_path):
    directory, file_name = os.path.split(directory_path)

    return directory, file_name


def separate_filename_and_extension(filename):
    name, extension = os.path.splitext(filename)

    return name, extension.replace(".", "")


def get_mac_address():
    mac = uuid.getnode()
    # if (mac >> 40) % 2:  # 유효한 MAC 주소인지 확인
    #     return "000000000000"  # 기본값 반환
    mac_address = ':'.join(f'{(mac >> i) & 0xff:02x}' for i in range(40, -1, -8))
    return "".join(mac_address.split(":"))


# /////////////////////////////////////////////////////////////////////////////////////////
def find_and_stop_qthreads():
    app = QApplication.instance()
    if app:
        for widget in app.allWidgets():
            if isinstance(widget, QThread) and widget is not QThread.currentThread():
                PRINT_(f"Stopping QThread: {widget}")
                widget.quit()
                widget.wait()

    # QObject 트리에서 QThread 찾기
    for obj in QObject.children(QApplication.instance()):
        if isinstance(obj, QThread) and obj is not QThread.currentThread():
            PRINT_(f"Stopping QThread: {obj}")
            obj.quit()
            obj.wait()


def stop_all_threads():
    current_thread = threading.current_thread()

    for thread in threading.enumerate():
        if thread is current_thread:  # 현재 실행 중인 main 스레드는 제외
            continue

        if isinstance(thread, threading._DummyThread):  # 더미 스레드는 제외
            PRINT_(f"Skipping DummyThread: {thread.name}")
            continue

        PRINT_(f"Stopping Thread: {thread.name}")

        if hasattr(thread, "stop"):  # stop() 메서드가 있으면 호출
            thread.stop()
        elif hasattr(thread, "terminate"):  # terminate() 메서드가 있으면 호출
            thread.terminate()

        if thread.is_alive():
            thread.join(timeout=1)  # 1초 기다린 후 종료


def cleanup_root_temp_folders(BASE_DIR):
    """temp_dir 내에서 root_temp_로 시작하는 모든 폴더를 삭제합니다."""
    for root, dirs, files in os.walk(BASE_DIR, topdown=False):
        for dir_name in dirs:
            # 'root_temp_'로 시작하는 폴더를 찾기
            if dir_name.startswith("root_temp_"):
                dir_path = os.path.join(root, dir_name)
                PRINT_(f"Deleting folder: {dir_path}")  # 삭제하려는 폴더 경로 출력
                shutil.rmtree(dir_path)  # 해당 폴더 및 그 하위 항목 삭제


class EmittingStream(QObject):
    textWritten = pyqtSignal(str)

    def write(self, text):
        self.textWritten.emit(str(text))

    def flush(self):
        pass


class ProgressDialog(QDialog):  # This class will handle both modal and non-modal dialogs
    progress_stop_sig = pyqtSignal()

    def __init__(self, message, modal=True, show=False, parent=None, unknown_max_limit=False,
                 on_count_changed_params_itself=False):

        super().__init__(parent)

        self.setWindowTitle(message)

        self.unknown_max_limit = unknown_max_limit

        # Set the dialog as modal or non-modal based on the 'modal' argument
        if modal:
            self.setModal(True)
        else:
            self.setWindowModality(QtCore.Qt.NonModal)

        self.resize(700, 100)  # Resize to desired dimensions

        self.max_cnt = 100
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setMaximum(int(self.max_cnt))

        self.label = QLabel("", self)
        self.close_button = QPushButton("Close", self)

        # Create a horizontal layout for the close button and spacer
        h_layout = QHBoxLayout()
        spacer = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        h_layout.addSpacerItem(spacer)
        h_layout.addWidget(self.close_button)

        # Create the main layout
        layout = QVBoxLayout(self)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.label)

        self.radio_button = QRadioButton("", self)
        layout.addWidget(self.radio_button)

        layout.addLayout(h_layout)
        self.setLayout(layout)

        # Close button click event
        self.close_button.clicked.connect(self.close)

        # Show or hide the close button based on 'show'
        if show:
            self.close_button.show()
        else:
            self.close_button.hide()

        # Timer to toggle radio button every 500ms
        self.timer = QTimer(self)

        self.radio_state = False  # Initial blink state

        if self.unknown_max_limit:
            # Remove the progress bar format (e.g., "%" sign)
            self.progress_bar.setFormat("")  # No percentage displayed

        self.timer.timeout.connect(self.toggle_radio_button)
        self.timer.start(100)  # 500ms interval

        self.cnt = 0
        self.on_count_changed_params_itself = on_count_changed_params_itself

    def getProgressBarMaximumValue(self):
        return self.max_cnt

    def setProgressBarMaximum(self, max_value):
        self.max_cnt = max_value
        self.progress_bar.setMaximum(int(max_value))

    def onCountChanged(self, value):
        self.progress_bar.setValue(int(value))

    def onProgressTextChanged(self, text):
        self.label.setText(text)

    def show_progress(self):
        if self.isModal():
            super().exec_()  # Execute as modal
        else:
            self.show()  # Show as non-modal

    def toggle_radio_button(self):
        if self.on_count_changed_params_itself:
            self.cnt += 1
            self.onCountChanged(self.cnt % self.max_cnt)

        if self.radio_state:
            self.radio_button.setStyleSheet("""
                        QRadioButton::indicator {
                            width: 12px;
                            height: 12px;
                            background-color: red;
                            border-radius: 5px;
                        }
                    """)
        else:
            self.radio_button.setStyleSheet("""
                        QRadioButton::indicator {
                            width: 12px;
                            height: 12px;
                            background-color: blue;
                            border-radius: 5px;
                        }
                    """)
        self.radio_state = not self.radio_state

    def closeEvent(self, event):
        self.timer.stop()
        event.accept()
        self.progress_stop_sig.emit()


class ColonLineHighlighter(QSyntaxHighlighter):
    def __init__(self, parent=None):
        super().__init__(parent)

        # 파란색 형식 (':'로 끝나는 줄)
        self.colon_format = QTextCharFormat()
        self.colon_format.setForeground(QColor("blue"))

        # 빨간색 형식 (특정 키워드)
        self.keyword_format = QTextCharFormat()
        self.keyword_format.setForeground(QColor("red"))

        # 녹색 형식 (특정 키워드와 ':'로 끝나는 줄)
        self.green_format = QTextCharFormat()
        self.green_format.setForeground(QColor("#00CED1"))

        # 강조할 키워드 설정
        self.keywords = re.compile(r'\b(global_config|model_config)\b')

        # tflite, onnx, caffemodel 키워드 설정
        self.special_keywords = re.compile(r'\b(.tflite|.onnx|.caffemodel)\b')

    def highlightBlock(self, text):
        # 'tflite', 'onnx', 'caffemodel'이 포함되고 ':'로 끝나는 경우 녹색으로 강조
        if self.special_keywords.search(text) and text.strip().endswith(':'):
            self.setFormat(0, len(text), self.green_format)

        # 키워드가 포함된 경우 빨간색 강조
        for match in self.keywords.finditer(text):
            start, end = match.span()
            self.setFormat(start, end - start, self.keyword_format)

        # ':'로 끝나는 줄에 파란색 적용 (키워드가 포함되지 않은 경우)
        if text.strip().endswith(':') and not self.keywords.search(text) and not self.special_keywords.search(text):
            self.setFormat(0, len(text), self.colon_format)


class LoadDirectoryThread(QThread):
    finished_load_project_sig = pyqtSignal(str)  # ret, failed_pairs, memory_profile 전달
    copy_status_sig = pyqtSignal(str, int)  # ret, failed_pairs, memory_profile 전달

    def __init__(self, m_source_dir, BASE_DIR, keyword_filter):
        super().__init__()
        self.running = True
        self.filter = keyword_filter

        self.src_dir = m_source_dir.replace("\\", "/")

        unique_id = str(uuid.uuid4())  # 고유한 UUID 생성
        self.target_dir = os.path.join(BASE_DIR, f"root_temp_{unique_id}", os.path.basename(self.src_dir)).replace("\\",
                                                                                                                   "/")

        cleanup_root_temp_folders(BASE_DIR=BASE_DIR)

    def run(self) -> None:
        # 코드 작성
        self.copy_directory_structure_2()
        self.finished_load_project_sig.emit(os.path.dirname(self.target_dir))

    def copy_directory_structure_1(self):
        if os.path.exists(self.target_dir):
            shutil.rmtree(self.target_dir)

        shutil.copytree(self.src_dir, self.target_dir)

    def copy_directory_structure_2(self, exclude=False):
        """
        exclude=True  → 제외 리스트에 있는 항목들을 제외한 나머지를 복사 (기본)
        exclude=False → 특정 확장자를 가진 파일이 있는 경우에만 해당 폴더를 포함하여 복사
        """
        filter_name = "exclude" if exclude else "include"

        CheckDir(dir_=self.target_dir)

        cnt = 0
        filter_list = self.filter  # 필터링할 단어 리스트

        # `include` 모드에서 확장자 리스트 설정
        include_mode_active = not exclude and bool(filter_list["include"])
        include_extensions = set(filter_list["include"]) if include_mode_active else set()

        for root, dirs, files in os.walk(self.src_dir):
            if not self.running:
                return "stop_copy_directory_structure_2"

            # `exclude` 모드인 경우 폴더 필터링
            if exclude:
                if any(excluded_word in os.path.basename(root) for excluded_word in filter_list["exclude"]):
                    continue
                dirs[:] = [d for d in dirs if not any(excluded_word in d for excluded_word in filter_list["exclude"])]

            # `include` 모드인 경우, 현재 폴더에 복사할 파일이 있는지 체크
            valid_files = []
            if include_mode_active:
                for file in files:
                    _, ext = os.path.splitext(file)
                    if ext in include_extensions:
                        valid_files.append(file)

                if not valid_files:
                    continue  # 포함할 파일이 없으면 폴더 자체를 복사하지 않음

            # 현재 폴더의 상대 경로를 대상 폴더 기준으로 계산
            relative_path = os.path.relpath(root, self.src_dir).replace("\\", "/")
            destination_dir = os.path.join(self.target_dir, relative_path).replace("\\", "/")

            if not os.path.exists(destination_dir):
                os.makedirs(destination_dir)

            # 파일 복사
            for file in valid_files if include_mode_active else files:

                if not self.running:
                    return "stop_copy_directory_structure_2_2"

                if exclude and any(excluded_word in file for excluded_word in filter_list["exclude"]):
                    continue

                source_file = os.path.join(root, file).replace("\\", "/")
                destination_file = os.path.join(destination_dir, file).replace("\\", "/")
                shutil.copy2(source_file, destination_file)

                cnt += 1
                self.copy_status_sig.emit(f"[{cnt}]   ../{os.path.basename(source_file)} copied", cnt)

    def stop(self):
        self.running = False
        self.quit()
        self.wait()


class RequestLLMThread(QThread):
    finished_analyze_sig = pyqtSignal(str)
    chunk_analyzed_sig = pyqtSignal(str)
    analysis_progress_sig = pyqtSignal(str)

    def __init__(self, ctrl_params):
        super().__init__()

        self.running = True
        self.root_dir = ctrl_params["project_src_file"]
        self.prompt = {
            "system_prompt": ctrl_params["system_prompt"],
            "user_prompt": ctrl_params["user_prompt"]
        }
        self.llm = ctrl_params["llm_model"]
        self.timeout = ctrl_params["timeout"]
        self.user_contents = ctrl_params["user_contents"]
        self.language = ctrl_params["language"]
        self.api_key = ctrl_params["llm_key"]
        self.max_token_limit = ctrl_params["max_token_limit"]

        self.client = None

        self.combined_content = ""
        self.file_metadata = []  # 파일 경로와 파일명을 저장할 리스트

        self.session = requests.Session()  # Create a session for HTTP requests

    @staticmethod
    def get_file_list(folder_path):
        """ 주어진 폴더 또는 파일에서 모든 파일 경로 리스트를 반환 """
        if os.path.isfile(folder_path):
            return [folder_path]
        elif os.path.isdir(folder_path):
            return [os.path.join(root, filename)
                    for root, _, files in os.walk(folder_path)
                    for filename in files]
        else:
            return []

    @staticmethod
    def read_files(file_paths):
        """ 파일 리스트를 받아서 내용을 읽고 메타데이터를 저장 """
        error = ""
        file_metadata = []
        for file_path in file_paths:
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()  # 전체 내용을 하나의 string으로 읽어 오기
                    file_metadata.append({
                        "file_name": os.path.basename(file_path),
                        "file_path": file_path,
                        "content": content
                    })

            except Exception as e:
                error += f"{file_path}\n"

        return file_metadata, error

    @staticmethod
    def slice_chunk_algorithm(text, max_length, overlap=2):
        """ 어떻게 slice 할 건지 .... def/class 단위로 코드 청크 분할 """
        pattern = r"^(def |class )"
        lines = text.split("\n")

        chunks = []
        current_chunk = []
        current_length = 0

        for i, line in enumerate(lines):
            if re.match(pattern, line) and current_length >= max_length:
                overlap_part = lines[max(0, i - overlap): i]
                chunks.append("\n".join(current_chunk + overlap_part))
                current_chunk = overlap_part[:]
                current_length = sum(len(l) for l in current_chunk)

            current_chunk.append(line)
            current_length += len(line)

        if current_chunk:
            chunks.append("\n".join(current_chunk))

        return chunks

    def read_all_raw_data(self, folder_path=None, user_contents=""):
        if folder_path is None:
            metadata = [{
                "file_name": "unknown",
                "file_path": "unknown",
                "content": user_contents
            }]
            return metadata, True

        file_paths = self.get_file_list(folder_path=folder_path)
        if len(file_paths) == 0:
            return "[Error] File or Directory not existed.", False

        metadata, error_msg = self.read_files(file_paths=file_paths)
        if len(metadata) == 0:
            return f"[Error] File Reading Error\n{error_msg}", False

        return metadata, True

    def llm_request(self, using_model="gpt-4o-mini", system_final_prompt="", user_final_prompt="", timeout=300):

        response, success = self.openai_request(using_model=using_model, system_final_prompt=system_final_prompt,
                                                user_final_prompt=user_final_prompt, timeout=timeout)

        return response, success

    def openai_request(self, using_model, system_final_prompt, user_final_prompt, timeout, http_request=True):
        if http_request:
            PRINT_("<<<   New Open Protocol     >>>>", self.running)
            """Make HTTP request and handle the response"""
            try:
                # Example API URL (replace with your actual API URL)
                url = "https://api.openai.com/v1/chat/completions"
                headers = {
                    "Authorization": f"Bearer {self.api_key}",  # Set your OpenAI API key
                    "Content-Type": "application/json"
                }
                payload = {
                    "model": f"{using_model}",
                    "messages": [
                        {"role": "system", "content": system_final_prompt},
                        {"role": "user", "content": user_final_prompt}
                    ],
                }

                response = self.session.post(url, json=payload, headers=headers, timeout=timeout)

                if response.status_code == 200:
                    chat_response = response.json().get("choices", [{}])[0].get("message", {}).get("content", "")
                    return chat_response, True
                else:
                    return f"Error: {response.status_code} - {response.text}", False
            except requests.exceptions.RequestException as e:
                return f"\nRequest error: {e}", False
        else:
            try:
                response = self.client.chat.completions.create(
                    model=using_model,
                    messages=[
                        {"role": "system", "content": system_final_prompt},
                        {"role": "user", "content": user_final_prompt}
                    ],
                    timeout=timeout  # openai 라이브러리에서는 request_timeout 사용
                )
                return response.choices[0].message.content, True

            except openai.OpenAIError as e:  # OpenAI 관련 오류
                return f"[Timed Out] OpenAI API Error: {str(e)} --> {timeout} s", False

            except Exception as e:  # 기타 예외 (어떤 모듈에서 발생하는지 확인)
                import traceback
                return f"[Error] Unexpected error: {str(e)}\n{traceback.format_exc()}", False

    def summarize_text(self, chunk, using_model, using_prompt, language, timeout):
        if not self.running:
            return "강제 종료 되었음.", False

        """ Use OpenAI to summarize long text """
        user_final_prompt = chunk + f"\n\n{using_prompt['user_prompt']}.  Answer in {language}"

        system_final_prompt = (
            f"{using_prompt['system_prompt']}."
        )

        response, success = self.llm_request(using_model=using_model, system_final_prompt=system_final_prompt,
                                             user_final_prompt=user_final_prompt, timeout=timeout)

        return response, success

    def slice_chunk_request_llm(self, metadata_s, max_length, using_model, using_prompt, language, timeout):
        """ 파일별 코드 청크를 나누고 개별 청크를 요약 """
        chunk_summaries = []
        for metadata in metadata_s:  # metadata는 파일 한개 전체의 내용이 들어가 있음.
            chunks = self.slice_chunk_algorithm(text=metadata["content"], max_length=max_length)  # 청크 분할 로직
            for chunk in chunks:
                summary, ret = self.summarize_text(chunk=chunk, using_model=using_model, using_prompt=using_prompt,
                                                   language=language, timeout=timeout)  # 개별 청크 요약
                chunk_summaries.append(f"File: {metadata['file_name']} (Path: {metadata['file_path']})\n{summary}")
                if not ret:
                    return chunk_summaries, False

        return chunk_summaries, True

    def generate_final_analysis(self, chunk_summaries=None, using_model="gpt-4o-mini",
                                using_prompt=None, language="english", timeout=300.0):
        if not self.running:
            return "최종 요약 결과 도출 전 강제 종료 되었습니다."

        """ 전체 요약 결과를 바탕으로 최종 분석 요청 """
        if chunk_summaries is None:
            user_final_prompt = (
                f"{using_prompt['user_prompt']}.  Answer in {language}"
            )
        else:
            user_final_prompt = (
                    "\n\n".join(chunk_summaries) +
                    f"\n\nWrite the final result incorporating any necessary additional feedback from the content above. Write in {language}"
            )

        system_final_prompt = (
            f"{using_prompt['system_prompt']}."
        )

        response, success = self.llm_request(using_model=using_model, system_final_prompt=system_final_prompt,
                                             user_final_prompt=user_final_prompt, timeout=timeout)

        return response

    def analyze_project_file(self, folder_path, user_contents, max_length=3000, using_model="gpt-4o-mini",
                             prompt=None,
                             language="english",
                             timeout=300):

        # 1. 선택된 모든 파일의 내용을 읽어 오기
        rawData, ret = self.read_all_raw_data(folder_path=folder_path, user_contents=user_contents)
        if not ret:
            return rawData

        # 2. 읽어온 데이터에 대해서 일정한 크기로 자른 후 자른 내용에 대해서 llm request하기
        chunk_summaries, ret = self.slice_chunk_request_llm(metadata_s=rawData, max_length=max_length,
                                                            using_model=using_model,
                                                            using_prompt=prompt, language=language, timeout=timeout)

        summarize_chunk_data = "\n\n".join(chunk_summaries)
        self.chunk_analyzed_sig.emit(summarize_chunk_data)

        if ret:
            if len(chunk_summaries) == 0:
                return f"[Error] Fail to Chunk Summary"
        else:
            return "\n".join(chunk_summaries)

        # 2. chunking 데이터를 LLM에 넣어 분석 결과 도출 단계
        result_message = self.generate_final_analysis(chunk_summaries=chunk_summaries, using_model=using_model,
                                                      using_prompt=prompt, language=self.language, timeout=timeout)

        return result_message

    def run(self) -> None:
        # 코드 작성
        if self.api_key is None:
            result_message = "[Error] Set the OPENAI_API_KEY environment variable"
        else:
            try:
                self.client = openai.OpenAI(api_key=self.api_key)
                result_message = self.analyze_project_file(folder_path=self.root_dir, user_contents=self.user_contents,
                                                           max_length=self.max_token_limit, using_model=self.llm,
                                                           prompt=self.prompt, language=self.language,
                                                           timeout=self.timeout)

            except openai.AuthenticationError:
                result_message = "Authentication Error: Please check your OpenAI API key."
            except openai.OpenAIError as e:
                result_message = f"API Request Error: {str(e)}"
            except Exception as e:
                result_message = f"An unexpected error occurred: {str(e)}\n{traceback.format_exc()}"
                print(result_message)  # 로그 출력
                sys.exit(1)  # 비정상 종료 (exit code 1

        self.finished_analyze_sig.emit(result_message)

    def stop(self):
        """Stop the thread and close the session"""
        self.running = False  # 종료 플래그 설정
        # First, ensure the session is closed safely
        if self.session:
            self.session.close()

        # Now, quit the thread and wait for it to finish
        self.quit()  # Quit the event loop
        self.wait()  # Wait for the thread to finish


# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

# 파일 확장자에 맞는 로더 매핑
EXTENSION_TO_LOADER = {
    ".py": PythonLoader,
    ".ipynb": NotebookLoader,
    ".txt": TextLoader,
    ".json": JSONLoader,
    "default": UnstructuredFileLoader,
}


def load_file(file_path):
    """
    주어진 파일 경로에 맞는 로더를 사용하여 문서를 로드합니다.
    :param file_path: 파일 경로
    :return: 로드된 문서 리스트
    """
    file_extension = os.path.splitext(file_path)[1].lower()  # 파일 확장자 추출

    # 해당 확장자에 맞는 로더를 찾음
    if file_extension in EXTENSION_TO_LOADER:
        loader_cls = EXTENSION_TO_LOADER[file_extension]
    else:
        loader_cls = EXTENSION_TO_LOADER["default"]

    # 파일에 맞는 로더 생성
    loader = DirectoryLoader(os.path.dirname(file_path), glob=os.path.basename(file_path), loader_cls=loader_cls)

    # 파일 로드
    return loader.load()


def load_files(project_dir):
    """
    주어진 디렉토리에서 모든 파일을 찾아, 해당 파일 확장자에 맞는 로더를 사용하여 문서를 로드하고 결합합니다.
    :param project_dir: 프로젝트 디렉토리 경로 또는 파일 경로
    :return: 결합된 문서 리스트
    """
    all_docs = []

    # project_dir이 파일인지 폴더인지 확인
    if os.path.isfile(project_dir):
        # 파일이 주어진 경우
        docs = load_file(project_dir)
        all_docs.extend(docs)  # 로드된 문서들을 결합

    elif os.path.isdir(project_dir):
        # project_dir이 폴더인 경우
        for root, dirs, files in os.walk(project_dir):
            for file_name in files:
                file_path = os.path.join(root, file_name)
                docs = load_file(file_path)
                all_docs.extend(docs)  # 로드된 문서들을 결합

    else:
        raise ValueError(f"주어진 경로가 유효한 파일 또는 디렉토리가 아닙니다: {project_dir}")

    return all_docs


class CodeAnalysisThread(QThread):
    finished_analyze_sig = pyqtSignal(str)
    chunk_analyzed_sig = pyqtSignal(str)
    analysis_progress_sig = pyqtSignal(str)

    def __init__(self, ctrl_params):
        super().__init__()

        self.running = True

        self.llm = ctrl_params["llm_model"]
        self.output_language = ctrl_params["language"]
        self.max_token_limit = ctrl_params["max_token_limit"]
        self.project_dir = ctrl_params["project_src_file"]
        self.user_contents = ctrl_params["user_contents"]
        self.user_prompt = f'{ctrl_params["user_prompt"]}. Answer in {self.output_language}.'
        self.system_prompt = ctrl_params["system_prompt"]
        self.max_context_size = 3
        self.temperature = 0.3

        self.OPENAI_API_KEY = ctrl_params["llm_key"]
        self.OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
        self.HEADERS = {
            "Authorization": f"Bearer {self.OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }
        # 세션 생성
        self.session = requests.Session()
        self.session.headers.update(self.HEADERS)

    def run(self):

        self.analysis_progress_sig.emit("Read all File Data...")

        # 프로젝트 디렉토리 내 모든 파일을 확인하고, 파일 확장자에 맞는 로더를 선택하여 파일을 로드
        # self.project_dir이 None인 경우 user_contents를 사용하도록 처리
        if self.project_dir is None:
            all_docs = [{"metadata": {"source": "user_input"}, "page_content": self.user_contents}]
            # self.project_dir가 None인 경우, 파일 구조를 따로 지정
            file_structure = "\n".join([doc['metadata']['source'] for doc in all_docs])  # doc['metadata']['source']로 접근
        else:
            # 프로젝트 디렉토리 내 모든 파일을 확인하고, 파일 확장자에 맞는 로더를 선택하여 파일을 로드
            all_docs = load_files(project_dir=self.project_dir)
            # 파일 구조를 LLM에 전달
            file_structure = "\n".join([doc.metadata['source'] for doc in all_docs])  # 기존대로 doc.metadata['source']로 접근

        init_message = {
            "model": f"{self.llm}",
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user",
                 "content": f"The overall code files and folder structure of the project are as follows:\n\n{file_structure}\n\nRemember this structure."}
            ],
        }

        # 초기 파일 구조 전달
        self._send_message(init_message)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.max_token_limit, chunk_overlap=100)

        previous_responses = []  # 모든 문서의 결과를 저장하는 리스트

        for doc in all_docs:
            # 파일 정보 가져오기
            if self.project_dir is None:
                file_name = doc['metadata']['source']
                content = doc['page_content']
            else:
                file_name = doc.metadata['source']
                content = doc.page_content

            self.analysis_progress_sig.emit(f"{file_name} Chunking...")
            chunks = text_splitter.split_text(content)

            file_responses = []  # 파일별 응답 저장 리스트 (각 파일별 context 유지)

            for idx, chunk in enumerate(chunks):
                context_messages = [{"role": "system", "content": self.system_prompt}]

                # 최대 context size만큼 이전 응답 포함
                for prev in previous_responses[-self.max_context_size:]:
                    context_messages.append({"role": "assistant", "content": prev})

                message_added = "Additionally, when analyzing, if file information is available, provide details about each file separately. Specify what kind of file it is and its role in the overall structure. When presenting this information in the response, start with [file information] on a new line, followed by the details from the next line onward."
                context_messages.append({
                    "role": "user",
                    "content": f"{chunk}\n\nfile name: {file_name} (Chunk {idx + 1}/{len(chunks)})\n{self.user_prompt}\n{message_added}"
                })

                payload = {"model": self.llm, "messages": context_messages, "temperature": self.temperature}

                response = self._send_message(payload)

                try:
                    result = response["choices"][0]["message"]["content"]
                    file_responses.append(result)  # 파일별 응답 저장
                    msg_progress = f"Finished Chunk Analysis: {file_name} (Chunk {idx + 1}/{len(chunks)})"
                    self.analysis_progress_sig.emit(msg_progress)

                except Exception as e:
                    msg_progress = f"Process Stopped.\n {e}. "
                    self.analysis_progress_sig.emit(msg_progress)
                    break

            # 파일별 응답을 전체 응답 리스트에 추가
            previous_responses.extend(file_responses)

        # 모든 문서의 분석 결과를 합쳐 요약 요청
        summarize_chunk_data = "\n\n".join(previous_responses)
        self.chunk_analyzed_sig.emit(summarize_chunk_data)

        # 최종 프로젝트 분석 요청        
        self.analysis_progress_sig.emit("Wait for Summarizing...")

        # >>>>>>>>>>> summarize_chunk_data가 매우 큰 경우에 대하여 다시 split 할 필요가 있음.
        final_payload = {
            "model": self.llm,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user",
                 "content": f"\n\n Summarize the following in {self.output_language}.\n\n{summarize_chunk_data}"}
            ],
            "temperature": self.temperature
        }
        response = self._send_message(final_payload)

        if response:
            final_result = response["choices"][0]["message"]["content"]
            self.finished_analyze_sig.emit(final_result)
        else:
            self.finished_analyze_sig.emit("Fail to Analysis")

    def stop(self):
        """Stop the thread and close the session"""
        self.running = False  # 종료 플래그 설정

        # 세션 종료
        self.session.close()

        self.quit()  # Quit the event loop
        self.wait()  # Wait for the thread to finish

    def _send_message(self, payload):
        """API 호출을 처리하는 내부 메서드"""
        if not self.running:
            self.analysis_progress_sig.emit("Terminated forcibly. Wait for normal closing")
            return None

        try:
            response = self.session.post(self.OPENAI_API_URL, json=payload)  # 여기 수정됨!
            if response.status_code == 200:

                print("test\n\n", response.json())
                return response.json()
            else:
                print(f"Fail to API Call: {response.status_code}, {response.text}")
                return None
        except requests.RequestException as e:
            print(f"Fail to Requesst: {e}")
            return None
