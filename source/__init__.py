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
from collections import OrderedDict
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, Future
from reportlab.pdfgen import canvas

from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtCore import pyqtSignal, QTimer, Qt, QThread, QObject, QModelIndex
from PyQt5.QtGui import QSyntaxHighlighter, QTextCharFormat, QColor
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QProgressBar, QPushButton, QHBoxLayout, QSpacerItem, \
    QSizePolicy, QRadioButton, QWidget, QMessageBox, QFileDialog, QApplication, QFileSystemModel

# from langchain_community.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader
# from langchain_text_splitters import MarkdownHeaderTextSplitter
# from langchain_text_splitters import CharacterTextSplitter

logging.basicConfig(level=logging.INFO)

EnablePrint = False


def PRINT_(*args):
    if EnablePrint:
        logging.info(args)


# ANSI 코드 정규식 패턴 (터미널 컬러 코드)
ANSI_ESCAPE = re.compile(r'\x1B[@-_][0-?]*[ -/]*[@-~]')


class ProgressDialog(QDialog):  # This class will handle both modal and non-modal dialogs    

    def __init__(self, message, modal=True, show=False, parent=None, unknown_max_limit=False,
                 self_onCountChanged_params=False):
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
        self.self_onCountChanged_params = self_onCountChanged_params

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

    def closeEvent(self, event):
        self.timer.stop()
        event.accept()

    def toggle_radio_button(self):
        if self.self_onCountChanged_params:
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


class LoadDir_Thread(QThread):
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

        self.cleanup_root_temp_folders(BASE_DIR=BASE_DIR)

    @staticmethod
    def cleanup_root_temp_folders(BASE_DIR):
        """temp_dir 내에서 root_temp_로 시작하는 모든 폴더를 삭제합니다."""
        for root, dirs, files in os.walk(BASE_DIR, topdown=False):
            for dir_name in dirs:
                # 'root_temp_'로 시작하는 폴더를 찾기
                if dir_name.startswith("root_temp_"):
                    dir_path = os.path.join(root, dir_name)
                    PRINT_(f"Deleting folder: {dir_path}")  # 삭제하려는 폴더 경로 출력
                    shutil.rmtree(dir_path)  # 해당 폴더 및 그 하위 항목 삭제

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
                PRINT_("force termination_main loop")
                break

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
                    PRINT_("force termination_sub_loop")
                    break

                if exclude and any(excluded_word in file for excluded_word in filter_list["exclude"]):
                    continue

                source_file = os.path.join(root, file).replace("\\", "/")
                destination_file = os.path.join(destination_dir, file).replace("\\", "/")
                shutil.copy2(source_file, destination_file)

                cnt += 1
                self.copy_status_sig.emit(f"[{cnt}]   ../{os.path.basename(source_file)} copied", cnt)

    def X_copy_directory_structure_2(self, exclude=False):
        """
        src_dir의 파일과 폴더를 대상 폴더로 복사하는 함수.
        
        exclude=True이면 제외 리스트에 있는 항목들을 제외한 나머지를 복사
        exclude=False이면 포함 리스트에 있는 항목들만 복사        """

        CheckDir(dir_=self.target_dir)

        cnt = 0
        filter_list = self.filter  # 필터링할 단어 리스트

        # src_dir의 모든 파일과 폴더를 재귀적으로 탐색
        for root, dirs, files in os.walk(self.src_dir):
            if not self.running:
                PRINT_("force termination_main loop")
                break

            # 디렉터리 필터링
            if exclude:
                # 제외 리스트에 있는 단어가 포함된 폴더는 건너뜀
                if any(excluded_word in os.path.basename(root) for excluded_word in filter_list["exclude"]):
                    continue
                    # dirs에서 제외할 폴더를 제거
                dirs[:] = [d for d in dirs if not any(excluded_word in d for excluded_word in filter_list["exclude"])]
            else:
                # 포함 리스트에 있는 단어가 포함되지 않은 폴더는 건너뜀
                if not any(included_word in os.path.basename(root) for included_word in filter_list["include"]):
                    continue
                    # dirs에서 포함할 폴더만 유지
                dirs[:] = [d for d in dirs if any(included_word in d for included_word in filter_list["include"])]

            # 현재 폴더의 상대 경로를 대상 폴더 기준으로 계산
            relative_path = os.path.relpath(root, self.src_dir).replace("\\", "/")
            destination_dir = os.path.join(self.target_dir, relative_path).replace("\\", "/")

            if not os.path.exists(destination_dir):
                os.makedirs(destination_dir)

            # 파일 필터링 및 복사
            for file in files:
                if not self.running:
                    PRINT_("force termination_sub_loop")
                    break

                if exclude:
                    # 제외 리스트에 있는 단어가 포함된 파일은 건너뜀
                    if any(excluded_word in file for excluded_word in filter_list["exclude"]):
                        continue
                else:
                    # 포함 리스트에 있는 단어가 포함되지 않은 파일은 건너뜀
                    if not any(included_word in file for included_word in filter_list["include"]):
                        continue

                source_file = os.path.join(root, file).replace("\\", "/")
                destination_file = os.path.join(destination_dir, file).replace("\\", "/")
                shutil.copy2(source_file, destination_file)  # 메타데이터 포함하여 복사

                cnt += 1
                self.copy_status_sig.emit(f"[{cnt}]   ../{os.path.basename(source_file)} copied", cnt)

                PRINT_(cnt)

    def stop(self):
        self.running = False
        self.quit()
        self.wait(3000)


"""
이미 요청이 진행 중인 상황에서 서버의 동작을 멈추고 싶은데....
"""


class LLM_Analyze_Prompt_Thread(QThread):
    finished_analyze_sig = pyqtSignal(str)
    chunk_analyzed_sig = pyqtSignal(str)

    def __init__(self, ctrl_params):
        super().__init__()

        self.running = True
        self.root_dir = ctrl_params["project_src_file"]
        self.prompt = ctrl_params["prompt"]
        self.llm = ctrl_params["llm_model"]
        self.timeout = ctrl_params["timeout"]
        self.user_contents = ctrl_params["user_contents"]
        self.language = ctrl_params["language"]
        self.api_key = ctrl_params["llm_key"]
        self.max_token_limit = ctrl_params["max_token_limit"]

        self.client = None

        self.combined_content = ""
        self.file_metadata = []  # 파일 경로와 파일명을 저장할 리스트

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
                    content = file.read()
                    file_metadata.append({
                        "file_name": os.path.basename(file_path),
                        "file_path": file_path,
                        "content": content
                    })

            except Exception as e:
                error += f"{file_path}\n"

        return file_metadata, error

    @staticmethod
    def chunk_logic_algorithm(text, max_length, overlap=2):
        """ def/class 단위로 코드 청크 분할 """
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

    def summarize_text(self, chunk, using_model, using_prompt, language, timeout):
        """ Use OpenAI to summarize long text """
        final_prompt = (
                "\n\n".join(chunk) +
                f"\n\n{using_prompt}.  Answer in {language}"
        )

        try:
            response = self.client.chat.completions.create(
                model=using_model,
                messages=[{"role": "user", "content": final_prompt}],
                timeout=timeout
            )
            return response.choices[0].message.content, True
        except openai.OpenAIError as e:  # OpenAI 관련 오류
            return f"[Timed Out] summarize_text - OpenAI API Error: {str(e)} --> {timeout} s", False

        except Exception as e:  # 기타 예외 (어떤 모듈에서 발생하는지 확인)
            import traceback
            return f"[Error] summarize_text - Unexpected error: {str(e)}\n{traceback.format_exc()}", False

    def summarize_chunks(self, metadata_s, max_length, using_model, using_prompt, language, timeout):
        """ 파일별 코드 청크를 나누고 개별 청크를 요약 """
        chunk_summaries = []
        for metadata in metadata_s:
            chunks = self.chunk_logic_algorithm(text=metadata["content"], max_length=max_length)  # 청크 분할 로직
            for chunk in chunks:
                summary, ret = self.summarize_text(chunk=chunk, using_model=using_model, using_prompt=using_prompt,
                                                   language=language, timeout=timeout)  # 개별 청크 요약
                chunk_summaries.append(f"File: {metadata['file_name']} (Path: {metadata['file_path']})\n{summary}")
                if not ret:
                    return chunk_summaries, False

        return chunk_summaries, True

    def generate_final_analysis(self, chunk_summaries=None, using_model="gpt-4o-mini",
                                using_prompt="please analyze prompt", language="english", timeout=300.0):

        """ 전체 요약 결과를 바탕으로 최종 분석 요청 """
        if chunk_summaries is None:
            final_prompt = (
                f"{using_prompt}.  Answer in {language}"
            )
        else:
            final_prompt = (
                    "\n\n".join(chunk_summaries) +
                    f"\n\n{using_prompt}.  Answer in {language}"
            )

        try:
            response = self.client.chat.completions.create(
                model=using_model,
                messages=[{"role": "user", "content": final_prompt}],
                timeout=timeout  # openai 라이브러리에서는 request_timeout 사용
            )
            return response.choices[0].message.content

        except openai.OpenAIError as e:  # OpenAI 관련 오류
            return f"[Timed Out] OpenAI API Error: {str(e)} --> {timeout} s"

        except Exception as e:  # 기타 예외 (어떤 모듈에서 발생하는지 확인)
            import traceback
            return f"[Error] Unexpected error: {str(e)}\n{traceback.format_exc()}"

    def analyze_project(self, folder_path, user_contents, max_length=3000, using_model="gpt-4o-mini",
                        prompt="please analyze",
                        language="english",
                        timeout=300):

        # 1. 선택된 모든 파일의 내용을 읽어 오고 또는 사용자가 정의한 내용을 읽어와 일정한 크기로 chunking 하는 부분임
        metadata = []

        if folder_path is not None:
            """ 프로젝트 코드 파일을 분석하고 OpenAI로 평가 """
            file_paths = self.get_file_list(folder_path=folder_path)
            if len(file_paths) == 0:
                return "[Error] File or Directory not existed."

            metadata, error_msg = self.read_files(file_paths=file_paths)
            if len(metadata) == 0:
                return f"[Error] File Reading Error\n{error_msg}"

        else:
            metadata.append(
                {
                    "file_name": "unknown",
                    "file_path": "unknown",
                    "content": user_contents
                }
            )

        chunk_summaries, ret = self.summarize_chunks(metadata_s=metadata, max_length=max_length,
                                                     using_model=using_model,
                                                     using_prompt=prompt, language=language, timeout=timeout)
        if ret:
            if len(chunk_summaries) == 0:
                return f"[Error] Fail to Chunk Summary"
        else:
            return "\n".join(chunk_summaries)

        summarize_chunk_data = "\n\n".join(chunk_summaries)
        self.chunk_analyzed_sig.emit(summarize_chunk_data)

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
                result_message = self.analyze_project(folder_path=self.root_dir, user_contents=self.user_contents,
                                                      max_length=self.max_token_limit, using_model=self.llm,
                                                      prompt=self.prompt, language=self.language, timeout=self.timeout)

            except openai.AuthenticationError:
                result_message = "Authentication Error: Please check your OpenAI API key."
            except openai.OpenAIError as e:
                result_message = f"API Request Error: {str(e)}"
            except Exception as e:
                result_message = f"An unexpected error occurred: {str(e)}\n{traceback.format_exc()}"

        self.finished_analyze_sig.emit(result_message)

    def stop(self):
        self.running = False
        self.quit()
        self.wait(3000)
