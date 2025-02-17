import time
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
from collections import OrderedDict
from datetime import datetime

from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtCore import pyqtSignal, QTimer, Qt, QThread, QObject, QModelIndex
from PyQt5.QtGui import QSyntaxHighlighter, QTextCharFormat, QColor
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QProgressBar, QPushButton, QHBoxLayout, QSpacerItem, \
    QSizePolicy, QRadioButton, QWidget, QMessageBox, QFileDialog, QApplication, QFileSystemModel

from langchain_community.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_text_splitters import CharacterTextSplitter

logging.basicConfig(level=logging.INFO)

EnablePrint = True


def PRINT_(*args):
    if EnablePrint:
        logging.info(args)


Version = "Feasibility for Code Verification 0.0.2 (made by tom.shin)"

keyword = {
    "element_1": [""],
    "test_model": ["onnx"],
    "error_keyword": ["Error Code:", "Error code:", "Error msg:"],
    "exclusive_dir": ["DATA", "recipe", "yolox_darknet", "etc"],
    "gpt_models": ["gpt-4o-mini", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"],
    "pre_prompt": [
        "The above is a summary of each part of the project, including file names and paths. "
        "Based on this, analyze the overall structure, relationships between files, code quality, "
        "and identify any issues or bugs in the code. "
        "Also, suggest improvements and provide example code improvements.\n\n"
        "Please provide the analysis and improvement suggestions in Korean"
    ]
}

# ANSI 코드 정규식 패턴 (터미널 컬러 코드)
ANSI_ESCAPE = re.compile(r'\x1B[@-_][0-?]*[ -/]*[@-~]')


class ProgressDialog(QDialog):  # This class will handle both modal and non-modal dialogs
    send_user_close_event = pyqtSignal(bool)

    def __init__(self, message, modal=True, show=False, parent=None):
        super().__init__(parent)

        self.setWindowTitle(message)

        # Set the dialog as modal or non-modal based on the 'modal' argument
        if modal:
            self.setModal(True)
        else:
            self.setWindowModality(QtCore.Qt.NonModal)

        self.resize(700, 100)  # Resize to desired dimensions

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setMaximum(int(100))

        self.label = QLabel("", self)
        self.close_button = QPushButton("Close", self)
        self.radio_button = QRadioButton("", self)

        # Create a horizontal layout for the close button and spacer
        h_layout = QHBoxLayout()
        spacer = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        h_layout.addSpacerItem(spacer)
        h_layout.addWidget(self.close_button)

        # Create the main layout
        layout = QVBoxLayout(self)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.label)
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
        self.timer.timeout.connect(self.toggle_radio_button)
        self.timer.start(500)  # 500ms interval

        self.radio_state = False  # Initial blink state

    def setProgressBarMaximum(self, max_value):
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
        self.send_user_close_event.emit(True)
        event.accept()

    def toggle_radio_button(self):
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


def load_markdown(data_path):
    with open(data_path, 'rb') as f:
        result = chardet.detect(f.read())
        encoding = result['encoding']

    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
        ("####", "Header 4"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on
    )

    with open(data_path, 'r', encoding=encoding) as file:
        PRINT_(data_path)
        data_string = file.read()
        documents = markdown_splitter.split_text(data_string)

        # 파일명을 metadata에 추가
        domain = data_path  # os.path.basename(data_path)
        for doc in documents:
            if not doc.metadata:
                doc.metadata = {}
            doc.metadata["domain"] = domain  # Document 객체의 metadata 속성에 파일명 추가

        return documents


def load_txt(data_path):
    with open(data_path, 'rb') as f:
        result = chardet.detect(f.read())
        encoding = result['encoding']

    text_splitter = CharacterTextSplitter(
        separator="\n",
        length_function=len,
        is_separator_regex=False,
    )

    with open(data_path, 'r', encoding=encoding) as file:
        data_string = file.read().split("\n")
        domain = data_path  # os.path.basename(data_path)
        documents = text_splitter.create_documents(data_string)

        for doc in documents:
            if not doc.metadata:
                doc.metadata = {}
            doc.metadata["domain"] = domain  # Document 객체의 metadata 속성에 파일명 추가

        return documents


def load_general(base_dir):
    data = []
    cnt = 0
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".txt"):
                file_path = os.path.join(root, file)
                if os.path.getsize(file_path) > 0:
                    cnt += 1
                    data += load_txt(file_path)

    PRINT_(f"the number of txt files is : {cnt}")
    return data


def load_document(base_dir):
    data = []
    cnt = 0
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".md"):
                file_path = os.path.join(root, file)
                if os.path.getsize(file_path) > 0:
                    cnt += 1
                    data += load_markdown(file_path)

    PRINT_(f"the number of md files is : {cnt}")
    return data


def get_markdown_files(source_dir):
    dir_ = source_dir
    loader = DirectoryLoader(dir_, glob="**/*.md", loader_cls=UnstructuredMarkdownLoader)
    documents = loader.load()
    PRINT_("number of doc: ", len(documents))
    return documents


def get_directory(base_dir, user_defined_fmt=None, file_full_path=False):
    data = set()

    for root, dirs, files in os.walk(base_dir):
        for exclusive in keyword["exclusive_dir"]:
            if exclusive in dirs:
                dirs.remove(exclusive)

        for file in files:
            if user_defined_fmt is None:
                if any(file.endswith(ext) for ext in keyword["test_model"]):
                    if file_full_path:
                        data.add(os.path.join(root, file))
                    else:
                        data.add(root)
            else:
                if any(file.endswith(ext) for ext in user_defined_fmt):
                    if file_full_path:
                        data.add(os.path.join(root, file))
                    else:
                        data.add(root)

    unique_paths = sorted(data)
    return unique_paths


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
                print("Timeout occurred, command terminated.")
            error_output.append("Command terminated due to timeout.")
            timeout_expired = True

        except Exception as e:
            # 기타 예외 처리 추가
            if log:
                print(f"subprocess Command exception failed:")
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
class OpenDir_WorkerThread(QThread):
    finished_open_dir_sig = pyqtSignal()  # ret, failed_pairs, memory_profile 전달

    def __init__(self, dirPath):
        super().__init__()
        self.running = True
        self.dirPath = dirPath.replace("\\", "/")

    def run(self) -> None:
        # 코드 작성

        self.finished_open_dir_sig.emit()

    def stop(self):
        self.running = False
        self.quit()
        self.wait(3000)


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
