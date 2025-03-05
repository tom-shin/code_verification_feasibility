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
import tiktoken
import ast
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

ANSI_ESCAPE = re.compile(r'\x1B(?:[@-_]|[[0-?]*[ -/]*[@-~])')


def handle_exception(e):
    traceback.print_exc()  # 상세한 traceback 정보 출력
    sys.exit(1)  # 강제 종료
    
    
class FileManager:
    EXTENSION_TO_LOADER = {
        ".py": PythonLoader,
        ".ipynb": NotebookLoader,
        ".txt": TextLoader,
        ".json": JSONLoader,
        "default": UnstructuredFileLoader,
    }

    FILTER = {
        "include": [
            ".py", ".c", ".cpp"
        ],
        "exclude": [
            ".git", ".idea", "pycache", ".zip", ".pdf", ".xlsx", ".bin", ".bat", ".onnx", "tflite", "caffe", "tool", "designer", "Gen-6"
            ]
    }

    @staticmethod
    def handle_exception(e):        
        traceback.print_exc()  # 상세한 traceback 정보 출력
        sys.exit(1)  # 강제 종료

    def load_file(self, file_path=None):
        try:
            folder = os.path.dirname(file_path)  # 폴더 경로
            filename = os.path.basename(file_path)  # 파일 이름 (확장자 포함)
            name, ext = os.path.splitext(filename)  # 이름과 확장자 분리

            # 해당 확장자에 맞는 로더를 찾음
            file_extension = ext.lower()

            if file_extension not in self.FILTER["include"]:
                return None

            if ext.lower() in self.EXTENSION_TO_LOADER:
                loader_cls = self.EXTENSION_TO_LOADER[file_extension]
            else:
                loader_cls = self.EXTENSION_TO_LOADER["default"]

            # 폴더는 DirectoryLoader / 단일 파일은 loader_cls 직접 사
            # loader = DirectoryLoader(folder, glob=os.path.basename(file_path), loader_cls=loader_cls)
            loader = loader_cls(file_path)

            return loader.load()

        except Exception as e:
            handle_exception(e)

    def load_files(self, project_dir):
        try:
            all_docs = []
            for root, _, files in os.walk(project_dir):
                for filename in files:
                    file_path = os.path.join(root, filename)
                    m_doc = self.load_file(file_path)

                    if m_doc is None:
                        continue

                    all_docs.extend(m_doc)  # 로드된 문서들을 결합

            return all_docs

        except Exception as e:
            handle_exception(e)

    def json_dump_f(self, file_path, data, use_encoding=False):
        try:
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

        except Exception as e:
            handle_exception(e)

    def json_load_f(self, file_path, use_encoding=False):
        try:
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

        except Exception as e:
            handle_exception(e)

    def save2html(self, file_path, data, use_encoding=False):
        try:
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

        except Exception as e:
            handle_exception(e)

    def save2txt(self, file_path, data, use_encoding=False):
        try:
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

        except Exception as e:
            handle_exception(e)


class OpenAIAssistant:
    base_url = "https://api.openai.com/v1"

    file_upload_url = f"{base_url}/files"
    create_assistant_url = f"{base_url}/assistants"
    create_thread_url = f"{base_url}/threads"

    MAX_INPUT_TOKEN = {
        "gpt-4-turbo": 128000,
        "gpt-4o-mini": 128000,  # 35000개가 효율이 가장 좋다
        "gpt-4o": 128000,  # 35000개가 효율이 가장 좋다
        "gpt-4": 8000,
        "gpt-3.5-turbo": 4096
    }

    @staticmethod
    def handle_exception(e):        
        traceback.print_exc()  # 상세한 traceback 정보 출력
        sys.exit(1)  # 강제 종료

    def __init__(self, c_ctrl_params=None, default_token=35000):
        try:
            self.api_key = c_ctrl_params["llm_key"]
            self.llm = c_ctrl_params["llm_model"]

            self.headers = {
                "Authorization": f"Bearer {self.api_key}",
                "OpenAI-Beta": "assistants=v2"
            }

            self.session = requests.Session()
            self.session.headers.update(self.headers)

            self.assistant_message_history = []
            self.path_for_file_analysis = ""

        except Exception as e:
            handle_exception(e)

    def upload_files(self, file_paths, include=None):
        print(f"파일 업로드 시작: {file_paths}")

        file_ids = []
        try:
            if isinstance(file_paths, str):
                # 단일 파일 경로인 경우
                if os.path.isfile(file_paths):
                    if include and file_paths.endswith(tuple(include)):
                        if os.path.getsize(file_paths) > 0:  # 파일이 비어있지 않은지 확인
                            with open(file_paths, "rb") as f:
                                upload_response = self.session.post(self.file_upload_url, files={"file": f},
                                                                    data={"purpose": "assistants"})
                                if upload_response.status_code == 200:
                                    file_ids.append(upload_response.json()["id"])
                                    print(f"파일 업로드 성공: {upload_response.text}")
                                else:
                                    print(f"파일 업로드 실패: {file_paths}, 오류: {upload_response.text}")
                                    raise Exception(f"파일 업로드 실패: {file_paths}")
                        else:
                            print(f"파일이 비어있어 건너뜁니다: {file_paths}")
                    else:
                        print(f"업로드 가능한 파일이 아닙니다: {file_paths}")
                        raise Exception(f"업로드 가능한 파일이 아닙니다: {file_paths}")

                elif os.path.isdir(file_paths):
                    # 폴더인 경우
                    for root, dirs, files in os.walk(file_paths):
                        for file in files:
                            if include and any(file.endswith(ext) for ext in include):
                                file_path = os.path.join(root, file)
                                if os.path.getsize(file_path) > 0:  # 파일이 비어있지 않은지 확인
                                    with open(file_path, "rb") as f:
                                        upload_response = self.session.post(self.file_upload_url, files={"file": f},
                                                                            data={"purpose": "assistants"})
                                        if upload_response.status_code == 200:
                                            file_ids.append(upload_response.json()["id"])
                                            print(f"파일 업로드 성공: {upload_response.text}")
                                        else:
                                            print(f"파일 업로드 실패: {file_path}, 오류: {upload_response.text}")
                                            raise Exception(f"파일 업로드 실패: {file_path}")
                                else:
                                    print(f"파일이 비어있어 건너뜁니다: {file_path}")

                else:
                    print(f"파일 또는 폴더가 존재하지 않습니다: {file_paths}")
                    raise Exception(f"파일 또는 폴더가 존재하지 않습니다: {file_paths}")

            self.path_for_file_analysis = file_paths

        except Exception as e:
            handle_exception(e)

        return file_ids

    def create_assistant(self, system_prompt='당신은 program 코드 개발 경역이 30년 이상된 전문가 입니다.', temperature=0.7):
        try:
            create_assistant_response = self.session.post(
                self.create_assistant_url,
                json={
                    "name": f"Code Analyzer: {os.path.basename(self.path_for_file_analysis)}",
                    "instructions": system_prompt,
                    "tools": [{"type": "code_interpreter"}],
                    "model": self.llm,
                    "temperature": temperature
                }
            )
            if create_assistant_response.status_code == 200:
                return create_assistant_response.json()["id"]
            else:
                raise Exception(f"어시스턴트 생성 실패: {create_assistant_response.text}")

        except Exception as e:
            handle_exception(e)

    def create_thread(self):
        try:
            """새로운 스레드 생성"""
            create_thread_response = self.session.post(self.create_thread_url)
            if create_thread_response.status_code == 200:
                return create_thread_response.json()["id"]
            else:
                raise Exception(f"스레드 생성 실패: {create_thread_response.text}")

        except Exception as e:
            handle_exception(e)

    def start_analysis(self, assistant_id, file_ids,
                       analysis_message=f"소스 파일 코드에 대해서 정적 분석 수행 하세요\n. 이슈 및 버그 코드에 대해서 개선 코드 제안 해 주세요\n",
                       temperature=0.7):
        """정적 분석 요청을 시작"""
        try:
            thread_id = self.create_thread()
            create_message_url = f"{self.base_url}/threads/{thread_id}/messages"

            attachments = [{"file_id": file_id, "tools": [{"type": "code_interpreter"}]} for file_id in file_ids]

            add_message_response = self.session.post(create_message_url,
                                                     json={
                                                         "role": "user", "content": analysis_message,
                                                         "attachments": attachments
                                                     }
                                                     )

            if add_message_response.status_code != 200:
                raise Exception(f"메시지 추가 실패: {add_message_response.text}")

        except Exception as e:
            handle_exception(e)

        try:
            run_url = f"{self.base_url}/threads/{thread_id}/runs"

            run_response = self.session.post(
                run_url,
                json={"assistant_id": assistant_id, "temperature": temperature}  # 코드 실행 도구 없음
            )

            if run_response.status_code == 200:
                run_id = run_response.json()["id"]
                print(f"분석 시작 (Run ID: {run_id}, Thread ID: {thread_id})")
                return run_id, thread_id

            else:
                raise Exception(f"정적 분석 실패: {run_response.text}")

        except Exception as e:
            handle_exception(e)

    def wait_for_run_completion(self, run_id, thread_id, interval=2):
        cnt = 0
        while True:
            try:
                status_response = self.session.get(f"{self.base_url}/threads/{thread_id}/runs/{run_id}")
                if status_response.status_code != 200:
                    raise Exception(f"실행 상태 조회 실패: {status_response.text}")

            except Exception as e:
                handle_exception(e)

            run_status = status_response.json()
            status = run_status.get("status", "unknown")
            cnt += 1
            print(f"현재 상태: {status}  {cnt}")

            # 분석이 완료된 상태일 경우 종료
            if status == "completed":
                return True
            elif status == "failed":
                raise Exception(f"분석: {status}")

            time.sleep(interval)

    def get_run_results(self, run_id, thread_id):
        try:
            result_response = self.session.get(f"{self.base_url}/threads/{thread_id}/messages")

            if result_response.status_code == 200:
                messages = result_response.json().get("data", [])

                # 분석된 메시지가 포함된 메시지를 찾아서 반환
                for message in messages:
                    if message.get("role") == "assistant":
                        contents = message.get("content", [])

                        # content가 리스트이므로 각 원소에서 "text" → "value" 추출
                        results = []
                        for item in contents:
                            if isinstance(item, dict) and item.get("type") == "text":
                                text_data = item.get("text", {})
                                value = text_data.get("value", "")
                                results.append(value)

                        return results  # 리스트로 반환

            else:
                raise Exception(f"결과 조회 실패: {result_response.text}")

        except Exception as e:
            handle_exception(e)

    def close(self):
        """세션 닫기"""
        self.session.close()


class OpenAISession:
    base_url = "https://api.openai.com/v1/chat/completions"

    MAX_INPUT_TOKEN = {
        "gpt-4-turbo": 128000,
        "gpt-4o-mini": 128000,  # 35000개가 효율이 가장 좋다
        "gpt-4o": 128000,  # 35000개가 효율이 가장 좋다
        "gpt-4": 8000,
        "gpt-3.5-turbo": 4096
    }

    @staticmethod
    def handle_exception(e):        
        traceback.print_exc()  # 상세한 traceback 정보 출력
        sys.exit(1)  # 강제 종료

    def __init__(self, c_ctrl_params=None, default_token=35000):
        try:
            self.api_key = c_ctrl_params["llm_key"]
            self.llm = c_ctrl_params["llm_model"]

            self.headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            self.session = requests.Session()
            self.session.headers.update(self.headers)

            self.assistant_message_history = []

        except Exception as e:
            handle_exception(e)

    def langchain_based_algorithm(self, string_content, f_limit_token):
        print("Called langchain_based_algorithm...")

        chunk_size = min(self.MAX_INPUT_TOKEN.get(self.llm, self.MAX_INPUT_TOKEN["gpt-4o-mini"]), f_limit_token)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=100)

        a_new_chunks = text_splitter.split_text("\n\n".join(string_content))

        return a_new_chunks

    @staticmethod
    def hca_tokenize_code(code):
        encoding = tiktoken.get_encoding("cl100k_base")  # GPT-4 기준
        return encoding.encode(code)

    @staticmethod
    def hca_extract_code_blocks(code):
        tree = ast.parse(code)
        blocks = []
        prev_lineno = 0
        lines = code.splitlines()

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
                start_lineno = node.lineno - 1  # Python AST는 1-based index
                if prev_lineno:
                    blocks.append('\n'.join(lines[prev_lineno:start_lineno]))
                prev_lineno = start_lineno

        if prev_lineno and prev_lineno < len(lines):
            blocks.append('\n'.join(lines[prev_lineno:]))

        return blocks

    def hca_split_large_blocks(self, blocks, max_tokens=1024):
        new_blocks = []

        for block in blocks:
            tokens = self.hca_tokenize_code(block)
            total_tokens = len(tokens)

            if total_tokens <= max_tokens:
                new_blocks.append(block)
            else:
                chunk_size = max_tokens
                token_chunks = [tokens[i: i + chunk_size] for i in range(0, total_tokens, chunk_size)]

                # 토큰을 문자열로 변환
                encoding = tiktoken.get_encoding("cl100k_base")
                for chunk_ in token_chunks:
                    new_blocks.append(encoding.decode(chunk_))

        return new_blocks

    def hca_merge_small_blocks(self, blocks, min_tokens=256):
        merged_blocks = []
        temp_block = ""

        for block in blocks:
            tokens = self.hca_tokenize_code(temp_block + block)

            if len(tokens) < min_tokens:
                temp_block += "\n" + block
            else:
                if temp_block.strip():
                    merged_blocks.append(temp_block)
                temp_block = block

        if temp_block.strip():
            merged_blocks.append(temp_block)

        return merged_blocks

    def hybrid_chunk_algorithm(self, string_content, limit_max_token, limit_min_token):
        print("Called hybrid_chunk_algorithm...")

        max_tokens = max(limit_max_token, limit_min_token)
        min_tokens = min(limit_max_token, limit_min_token)

        blocks = self.hca_extract_code_blocks(string_content)  # AST 기반 청킹
        split_blocks = self.hca_split_large_blocks(blocks, max_tokens=max_tokens)  # 긴 블록 나누기

        if max_tokens == min_tokens:
            return split_blocks

        final_blocks = self.hca_merge_small_blocks(split_blocks, min_tokens=min_tokens)  # 작은 블록 병합
        return final_blocks

    def split_text(self, string_content="", f_limit_max_token=4000, f_limit_min_token=256, use_defined_algorithm=False):
        if use_defined_algorithm:
            new_chunks = self.hybrid_chunk_algorithm(string_content=string_content, limit_max_token=f_limit_max_token,
                                                     limit_min_token=f_limit_min_token)
        else:
            new_chunks = self.langchain_based_algorithm(string_content=string_content, f_limit_token=f_limit_max_token)

        return new_chunks

    def chat_completions_stream(self, user_content='', system_content='', temperature=0.7, num_history=0):
        if num_history < 0:
            history_range = num_history
        else:
            history_range = -num_history  # 양수를 음수로 변환

        payload = {
            "model": self.llm,
            "messages": [
                            {"role": "system", "content": system_content},
                            {"role": "user", "content": user_content}
                        ] + self.assistant_message_history[history_range:],  # 히스토리 사용
            "temperature": temperature,
            "stream": True  # 스트리밍 활성화
        }

        try:
            with self.session.post(self.base_url, json=payload, stream=True) as a_response:
                if a_response.status_code == 200:
                    collected_messages = []
                    for line in a_response.iter_lines():
                        if line:
                            decoded_line = line.decode("utf-8").strip()
                            # ✅ JSON 파싱 전에 ANSI 코드 제거 적용
                            decoded_line = ANSI_ESCAPE.sub('', decoded_line)

                            if decoded_line.startswith("data: "):
                                decoded_line = decoded_line[6:]  # "data: " 부분 제거

                            if decoded_line == "[DONE]":  # 스트림 종료 신호
                                break

                            try:
                                json_data = json.loads(decoded_line)
                                if "choices" in json_data:
                                    content = json_data["choices"][0]["delta"].get("content", "")
                                    if content:
                                        print(content, end="", flush=True)  # 실시간 출력
                                        collected_messages.append(content)
                            except json.JSONDecodeError:
                                print(f"JSON 디코딩 오류 발생: {decoded_line}")

                    response_content = "".join(collected_messages)
                    self.assistant_message_history.append({"role": "assistant", "content": response_content})
                    return response_content
                else:
                    print(f"[Error] requests.exceptions.RequestException: {a_response.status_code}, {a_response.text}")
                    sys.exit(1)

        except Exception as e:
            handle_exception(e)

    def chat_completions_all_together(self, user_content='', system_content='', temperature=0.7, num_history=0):
        if num_history < 0:
            history_range = num_history
        else:
            history_range = -num_history  # 양수를 음수로 변환

        payload = {
            "model": self.llm,
            "messages": [
                            {"role": "system", "content": system_content},
                            {"role": "user", "content": user_content}
                        ] + self.assistant_message_history[history_range:],  # 히스토리 사용
            "temperature": temperature
        }

        try:
            a_response = self.session.post(self.base_url, json=payload)
            if a_response.status_code == 200:
                response_content = a_response.json()["choices"][0]["message"]["content"]
                self.assistant_message_history.append({"role": "assistant", "content": response_content})
                return response_content
            else:
                print(f"[Error] requests.exceptions.RequestException: {a_response.status_code}, {a_response.text}")
                sys.exit(1)

        except Exception as e:
            handle_exception(e)
        
    def close(self):
        """세션 닫기"""
        self.session.close()


def general_openai_api():
    api_method = False

    max_token = 512  # 최대 입력 토큰 수
    min_token = 256  # 최소 토큰 수 이하인 경우 인접 블록과 합침
    num_history_cnt = 3  # 과거 3개 기억하기
    temperature = 0.3
    use_hybrid = True
    system_prompt = "당신은 프로그램 코딩 경력이 최소 50년 이상 고급 기술자 입니다."
    user_prompt = "코드 정적 분석 해 주고, 에러, 버그 등 수정할 부분이 있으면 수정 코드 제시해 주세요"

    ctrl_params = {
        "llm_key": "sk-",
        "llm_model": "gpt-4o-mini"
    }

    project_dir = f"/home/tom/work/python_project/test_sample/exynos_ai_studio_verifier/source/source_files"

    # 전체 파일을 한번에 읽어와 전체 contents 기반 split 진행
    if api_method:
        file_instance = FileManager()
        docs = file_instance.load_files(project_dir)
        # for doc in docs:
        #     file_name = doc.metadata['source']
        #     content = doc.page_content
        #     print(file_name)
        #     print(content)

        # doc = file_instance.load_file(f"/home/tom/work/python_project/test_sample/exynos_ai_studio_verifier/source/source_files/run_enntest.py")
        # print(doc[0].metadata["source"])
        # print(doc[0].page_content)

        start = time.time()
        openai_instance = OpenAISession(c_ctrl_params=ctrl_params)

        for doc in docs:
            print("===================================================================================================")
            print(f"\n\nFile Name:")
            filename = doc.metadata["source"]
            print(filename)

            chunks = openai_instance.split_text(string_content=doc.page_content, f_limit_max_token=max_token,
                                                f_limit_min_token=min_token, use_defined_algorithm=use_hybrid)

            for idx, chunk in enumerate(chunks):
                print(
                    f"[Start]....................................................................................................{doc.metadata['source']}   {idx + 1}/{len(chunks)}")
                response = openai_instance.chat_completions_stream(
                    system_content=system_prompt,
                    user_content=f"{chunk}\n\nfile name: {filename}: (Chunk {idx + 1}/{len(chunks)})\n{user_prompt}.",
                    temperature=temperature,
                    num_history=-num_history_cnt
                )
                print(response)

        openai_instance.close()
        print(f"\n\n Elapsed Time:  {time.time() - start} s.\n\n")

    else:
        # 전체 파일을 한번에 하나씩 읽어와 읽어와 읽어온 파일의 contents 기반 split 진행
        file_instance = FileManager()
        openai_instance = OpenAISession(c_ctrl_params=ctrl_params)

        start = time.time()

        for root, dirs, files in os.walk(project_dir):
            for filename in files:
                file_path = os.path.join(root, filename)
                m_doc = file_instance.load_file(file_path)
                if m_doc is None:
                    continue

                print(
                    "===================================================================================================")
                print(f"\n\nFile Name:")
                filename = m_doc[0].metadata["source"]
                print(filename)

                chunks = openai_instance.split_text(string_content=m_doc[0].page_content, f_limit_max_token=max_token,
                                                    f_limit_min_token=min_token, use_defined_algorithm=use_hybrid)

                # 아래 2줄은 이전 기억할 범위를 현재 분석하고자 하는 개별 파일 전체에 대해서 기억 하도록 함.
                openai_instance.assistant_message_history = []
                num_history_cnt = len(chunks)

                for idx, chunk in enumerate(chunks):
                    print(
                        f"[Start]....................................................................................................{m_doc[0].metadata['source']}   {idx + 1}/{len(chunks)}")
                    response = openai_instance.chat_completions_stream(
                        system_content=system_prompt,
                        user_content=f"{chunk}\n\nfile name: {filename}: (Chunk {idx + 1}/{len(chunks)})\n{user_prompt}.",
                        temperature=temperature,
                        num_history=num_history_cnt
                    )
                    print(response)

        openai_instance.close()
        print(f"\n\n Elapsed Time:  {time.time() - start} s.\n\n")


def openai_assistant_api():
    temperature = 0.5
    system_prompt = "You are a top-level expert with at least 30 years of experience in program source code.\n"
    include = [".py", ".c", ".cpp", ".zip", ]  # 업로드할 파일 확장자 리스트

    ctrl_params = {
        "llm_key": "",
        "llm_model": "gpt-4o-mini"
    }

    # project_dir = f"{os.getcwd()}/sample_code.py"
    project_dir = f"/home/tom/work/python_project/test_sample/exynos_ai_studio_verifier/source/source_files"

    start = time.time()
    openai_assistants_api_instance = OpenAIAssistant(c_ctrl_params=ctrl_params)

    try:
        file_ids = openai_assistants_api_instance.upload_files(file_paths=project_dir, include=include)
        assistant_id = openai_assistants_api_instance.create_assistant(system_prompt=system_prompt,
                                                                       temperature=temperature)

        if assistant_id:
            print("Static Analysis .............................................")

            user_prompt = "코드의 정적 분석 결과를 **항상** 100점 만점 기준으로 점수와 함께 제공해 줘.\n 각 코드 라인별로 **문제점을 상세히 설명**하고, 어떤 부분을 개선해야 하는지 알려 줘. 단순히 문제점을 나열하는 것이 아니라, **수정된 코드 블록을 제공하고, 문제 발생 이유와 함께 수정된 코드가 왜 더 나은지 설명해 줘.\n각 문제점마다 **추천 수정 코드 예시**도 추가로 제공해 줘.\n분석 결과는 markdown 형식으로 정리해 줘.\n"
            run_id_analysis, thread_id_analysis = openai_assistants_api_instance.start_analysis(assistant_id, file_ids,
                                                                                                analysis_message=user_prompt,
                                                                                                temperature=temperature)

            if openai_assistants_api_instance.wait_for_run_completion(run_id_analysis, thread_id_analysis):
                result = openai_assistants_api_instance.get_run_results(run_id_analysis, thread_id_analysis)
                print("".join(result))

            print("Dynamic Analysis .............................................")
            user_prompt = "Please analyze the code dynamically and track its behavior."
            run_id_analysis, thread_id_analysis = openai_assistants_api_instance.start_analysis(assistant_id, file_ids,
                                                                                                analysis_message=user_prompt,
                                                                                                temperature=temperature)

            if openai_assistants_api_instance.wait_for_run_completion(run_id_analysis, thread_id_analysis):
                result = openai_assistants_api_instance.get_run_results(run_id_analysis, thread_id_analysis)
                print("".join(result))

    except Exception as e:
        """예외 처리 메서드, 파일명과 줄 번호 정보 추가"""
        exc_type, exc_value, exc_tb = sys.exc_info()  # 예외 정보 가져오기
        file_name = exc_tb.tb_frame.f_code.co_filename  # 예외 발생 파일명
        line_number = exc_tb.tb_lineno  # 예외 발생 줄 번호

        print(f"에러 발생 파일: {file_name}, 라인: {line_number}")
        print(f"에러 메시지: {e}")  # 예외 메시지 출력
        traceback.print_exc()  # 상세한 traceback 정보 출력
        sys.exit(1)  # 강제 종료

    openai_assistants_api_instance.close()
    print(f"\n\n Elapsed Time:  {time.time() - start} s.\n\n")


if __name__ == "__main__":
    general_openai_api()
    # openai_assistant_api()





