#!/usr/bin/env python3

from source.head import *

if getattr(sys, 'frozen', False):  # PyInstaller로 패키징된 경우
    BASE_DIR = os.path.dirname(sys.executable)  # 실행 파일이 있는 폴더
    RESOURCE_DIR = sys._MEIPASS  # 임시 폴더(내부 리소스 저장됨)
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    RESOURCE_DIR = BASE_DIR  # 개발 환경에서는 현재 폴더 사용


class LoadDirectoryThread(QThread, FileManager):
    finished_load_project_sig = pyqtSignal(str)  # ret, failed_pairs, memory_profile 전달
    copy_status_sig = pyqtSignal(str, int)  # ret, failed_pairs, memory_profile 전달

    def __init__(self, m_source_dir, BASE_DIR, keyword_filter):
        super().__init__()
        # FileManager.__init__(self)

        self.running = True
        self.filter = keyword_filter

        self.src_dir = m_source_dir.replace("\\", "/")

        unique_id = str(uuid.uuid4())  # 고유한 UUID 생성
        self.target_dir = os.path.join(BASE_DIR, f"root_temp_{unique_id}", os.path.basename(self.src_dir)).replace("\\",
                                                                                                                   "/")

        self.remove_all_directory_sequentially(t_path=BASE_DIR, t_name="root_temp_")

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

        self.remove_create_dir(t_dir=self.target_dir)

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


class CodeAnalysisThreadVersion1(QThread):
    finished_analyze_sig = pyqtSignal(str)
    chunk_analyzed_sig = pyqtSignal(str)
    analysis_progress_sig = pyqtSignal(str)

    EXTENSION_TO_LOADER = {
        ".py": PythonLoader,
        ".ipynb": NotebookLoader,
        ".txt": TextLoader,
        ".json": JSONLoader,
        "default": UnstructuredFileLoader,
    }

    def __init__(self, ctrl_params):
        super().__init__()

        self.cnt_1 = 0
        self.running = True

        self.llm = ctrl_params["llm_model"]
        self.output_language = ctrl_params["language"]
        self.max_token_limit = ctrl_params["max_limit_token"]
        self.project_dir = ctrl_params["project_src_file"]
        self.user_contents = ctrl_params["user_contents"]
        self.user_prompt = ctrl_params["user_prompt"]
        self.system_prompt = f'{ctrl_params["system_prompt"]}. respond in {self.output_language}'
        self.num_history_cnt = ctrl_params["num_history_cnt"]
        self.temperature = ctrl_params["temperature"]

        self.OPENAI_API_KEY = ctrl_params["llm_key"]
        self.OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
        self.HEADERS = {
            "Authorization": f"Bearer {self.OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }
        # 세션 생성
        self.session = requests.Session()
        self.session.headers.update(self.HEADERS)

    def parsed_load_file(self, file_path):
        """
        주어진 파일 경로에 맞는 로더를 사용하여 문서를 로드합니다.
        :param file_path: 파일 경로
        :return: 로드된 문서 리스트
        """
        file_extension = os.path.splitext(file_path)[1].lower()  # 파일 확장자 추출

        # 해당 확장자에 맞는 로더를 찾음
        if file_extension in self.EXTENSION_TO_LOADER:
            loader_cls = self.EXTENSION_TO_LOADER[file_extension]
        else:
            loader_cls = self.EXTENSION_TO_LOADER["default"]

        # 파일에 맞는 로더 생성
        loader = DirectoryLoader(os.path.dirname(file_path), glob=os.path.basename(file_path), loader_cls=loader_cls)

        # 파일 로드
        return loader.load()

    def parsed_load_files(self, project_dir):
        """
        주어진 디렉토리에서 모든 파일을 찾아, 해당 파일 확장자에 맞는 로더를 사용하여 문서를 로드하고 결합합니다.
        :param project_dir: 프로젝트 디렉토리 경로 또는 파일 경로
        :return: 결합된 문서 리스트
        """
        all_docs = []

        # project_dir이 파일인지 폴더인지 확인
        if os.path.isfile(project_dir):
            # 파일이 주어진 경우
            docs = self.parsed_load_file(project_dir)
            all_docs.extend(docs)  # 로드된 문서들을 결합

        elif os.path.isdir(project_dir):
            # project_dir이 폴더인 경우
            for root, dirs, files in os.walk(project_dir):
                for file_name in files:
                    file_path = os.path.join(root, file_name)
                    docs = self.parsed_load_file(file_path)
                    all_docs.extend(docs)  # 로드된 문서들을 결합

        else:
            raise ValueError(f"주어진 경로가 유효한 파일 또는 디렉토리가 아닙니다: {project_dir}")

        return all_docs

    def recursive_summarization(self, summaries):
        """
        요약된 내용이 너무 크다면 다시 청크 단위로 분할하여 재귀적으로 요약하는 함수
        """
        if not summaries:
            return None

        # 요약 결과가 컨텍스트 제한 내라면 그대로 반환
        total_size = sum(len(s) for s in summaries)

        # 요약된 내용의 크기가 제한을 초과하지 않으면 그대로 반환
        if total_size <= self.max_token_limit:
            return "\n\n".join(summaries)

        self.cnt_1 += 1
        self.analysis_progress_sig.emit(f"Summarized output is too large, summarizing again...{self.cnt_1}")

        # 새로 분할하여 다시 요약 요청
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.max_token_limit, chunk_overlap=100)

        # 재귀 호출을 위해 새로운 청크 생성
        new_chunks = text_splitter.split_text("\n\n".join(summaries))

        # 분할된 청크들을 한 번에 처리할 수 있도록 다루기
        new_summarized_parts = []
        for idx, chunk in enumerate(new_chunks):
            payload = {
                "model": self.llm,
                "messages": [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": f"Summarize: {chunk}"}
                ],
                "temperature": self.temperature
            }

            response = self._send_message(payload)

            if response:
                summarized_text = response["choices"][0]["message"]["content"]
                new_summarized_parts.append(summarized_text)
            else:
                self.analysis_progress_sig.emit(f"Failed to summarize chunk {idx + 1}")

        # 새로운 요약된 부분의 총 크기 계산
        new_total_size = sum(len(s) for s in new_summarized_parts)

        # 요약된 부분의 크기가 줄어들지 않으면 더 이상 재귀를 하지 않음
        print("size, ", new_total_size, total_size)
        if new_total_size >= total_size:
            self.analysis_progress_sig.emit("Summarized output size did not decrease, stopping recursion.")
            return "\n\n".join(new_summarized_parts)

        # 여전히 크기가 크다면 다시 재귀 호출
        return self.recursive_summarization(new_summarized_parts)

    def run(self):
        self.analysis_progress_sig.emit("Read all File Data...")

        # 프로젝트 디렉토리 내 모든 파일을 확인하고, 파일 확장자에 맞는 로더를 선택하여 파일을 로드
        if self.project_dir is None:
            all_docs = [{"metadata": {"source": "user_input"}, "page_content": self.user_contents}]
            file_structure = "\n".join([doc['metadata']['source'] for doc in all_docs])
        else:
            all_docs = self.parsed_load_files(project_dir=self.project_dir)
            file_structure = "\n".join([doc.metadata['source'] for doc in all_docs])

        # 초기 파일 구조 전달
        init_message = {
            "model": f"{self.llm}",
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user",
                 "content": f"The overall code files and folder structure of the project are as follows:\n\n{file_structure}\n\nRemember this structure."}
            ],
        }
        self._send_message(init_message)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.max_token_limit, chunk_overlap=100)
        previous_responses = []

        for doc in all_docs:
            if self.project_dir is None:
                file_name = doc['metadata']['source']
                content = doc['page_content']
            else:
                file_name = doc.metadata['source']
                content = doc.page_content

            self.analysis_progress_sig.emit(f"{file_name} Chunking...")
            chunks = text_splitter.split_text(content)

            file_responses = []
            for idx, chunk in enumerate(chunks):
                context_messages = [{"role": "system", "content": self.system_prompt}]
                for prev in previous_responses[-self.num_history_cnt:]:
                    context_messages.append({"role": "assistant", "content": prev})

                context_messages.append({
                    "role": "user",
                    "content": f"{chunk}\n\nfile name: {file_name} (Chunk {idx + 1}/{len(chunks)})\n{self.user_prompt}."
                })

                payload = {"model": self.llm, "messages": context_messages, "temperature": self.temperature}
                response = self._send_message(payload)

                if response:
                    result = response["choices"][0]["message"]["content"]
                    file_responses.append(result)
                    msg_progress = f"Finished Chunk Analysis: {file_name} (Chunk {idx + 1}/{len(chunks)})"
                    self.analysis_progress_sig.emit(msg_progress)
                else:
                    self.analysis_progress_sig.emit(f"Failed to process chunk {idx + 1}")

            previous_responses.extend(file_responses)

        summarize_chunk_data = "\n\n".join(previous_responses)

        # self.chunk_analyzed_sig.emit(summarize_chunk_data)

        # # 요약 데이터가 너무 크다면 다시 분할
        final_result = summarize_chunk_data
        # self.analysis_progress_sig.emit("Wait for Summarizing...")
        # final_result = self.recursive_summarization(summaries=summarize_chunk_data)
        self.finished_analyze_sig.emit(final_result)

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

                # print("test\n\n", response.json())
                return response.json()
            else:
                # print(f"Fail to API Call: {response.status_code}, {response.text}")
                return None
        except requests.RequestException as e:
            # print(f"Fail to Requesst: {e}")
            return None


class CodeAnalysisThreadVersion2(QThread):
    finished_analyze_sig = pyqtSignal(str)
    chunk_analyzed_sig = pyqtSignal(str)
    analysis_progress_sig = pyqtSignal(str)

    def __init__(self, ctrl_params):
        super().__init__()
        self.result = None
        self.running = True
        self.ctrl_params = ctrl_params
        self.openai_instance = None

    def openai_standard_api(self):
        try:
            all_docs = []
            src_path = self.ctrl_params["project_src_file"]
            max_token = self.ctrl_params["max_limit_token"]
            min_token = self.ctrl_params["min_limit_token"]
            system_prompt = self.ctrl_params["system_prompt"]
            user_prompt = self.ctrl_params["user_prompt"]
            user_contents = self.ctrl_params["user_contents"]
            temperature = self.ctrl_params["temperature"]
            num_history_cnt = self.ctrl_params["num_history_cnt"]
            language = self.ctrl_params["language"]

            # FileManager에서 반환하는 객체와 유사한 구조 생성
            Document = namedtuple("Document", ["metadata", "page_content"])

            # if src_path is not None:
            if src_path is None:
                src_path = ''

            file_instance = FileManager()
            if file_instance.isdir_check(m_path=src_path):
                # print("dir")
                all_docs = file_instance.parsed_load_files(self.ctrl_params["project_src_file"])
                file_structure = "\n".join([doc.metadata['source'] for doc in all_docs])

            elif file_instance.isfile_check(m_path=src_path):
                # print("file")
                docs = file_instance.parsed_load_file(src_path)
                all_docs.extend(docs)  # 로드된 문서들을 결합
                # file_structure = "\n".join([doc.metadata['source'] for doc in all_docs])

            else:  # src_path가 None인 경우
                # print("user content")
                # noinspection PyArgumentList
                all_docs = [Document(metadata={"source": "user_input_context"}, page_content=user_contents)]
                # file_structure = "\n".join([doc.metadata["source"] for doc in all_docs])

            self.openai_instance = OpenAISession(c_ctrl_params=self.ctrl_params)

            previous_responses = []
            for doc in all_docs:
                if not self.running:
                    return previous_responses

                filename = doc.metadata["source"]

                use_dynamic_chunk_size = False
                if self.ctrl_params["use_dynamic_chunk_size"] and filename.endwith(".py"):
                    use_dynamic_chunk_size = True

                chunks = self.openai_instance.split_text(string_content=doc.page_content, f_limit_max_token=max_token,
                                                         f_limit_min_token=min_token,
                                                         use_defined_algorithm=use_dynamic_chunk_size)

                for idx, chunk in enumerate(chunks):
                    if not self.running:
                        return previous_responses

                    self.analysis_progress_sig.emit(f" {idx+1}/{len(chunks)*len(all_docs)}   '{os.path.basename(filename)}'  Analyzing...")

                    response = self.openai_instance.chat_completions_all_together(
                        system_content=system_prompt,
                        user_content=f"{chunk}\n\nfile name: {filename}: (Chunk {idx + 1}/{len(chunks)})\n{user_prompt}. response in {language}",
                        temperature=temperature,
                        num_history=-num_history_cnt
                    )
                    previous_responses.append(response)

            return previous_responses

        except Exception as e:
            handle_exception(e)

        finally:
            # OpenAI 세션이 None이 아니면 리소스를 해제
            if self.openai_instance is not None:
                try:
                    self.openai_instance.close()
                except Exception as e:
                    handle_exception(e)
                self.openai_instance = None

    def openai_assistant_api(self):
        language = self.ctrl_params["language"]
        temperature = self.ctrl_params["temperature"]
        system_prompt = self.ctrl_params["system_prompt"]
        project_dir = self.ctrl_params["project_src_file"]
        user_prompt = f'{self.ctrl_params["user_prompt"]}. response in {language}'

        include = [".py", ".c", ".cpp", ".zip", ]  # 업로드할 파일 확장자 리스트

        start = time.time()
        openai_assistants_api_instance = OpenAIAssistant(c_ctrl_params=self.ctrl_params)

        try:
            file_ids = openai_assistants_api_instance.upload_files(file_paths=project_dir, include=include)
            assistant_id = openai_assistants_api_instance.create_assistant(system_prompt=system_prompt,
                                                                           temperature=temperature)

            if assistant_id:
                # print("Static Analysis .............................................")
                run_id_analysis, thread_id_analysis = openai_assistants_api_instance.start_analysis(assistant_id,
                                                                                                    file_ids,
                                                                                                    analysis_message=user_prompt,
                                                                                                    temperature=temperature)

                if openai_assistants_api_instance.wait_for_run_completion(run_id_analysis, thread_id_analysis):
                    result = openai_assistants_api_instance.get_run_results(run_id_analysis, thread_id_analysis)
                    print("".join(result))

                    return result

                # print("Dynamic Analysis .............................................")
                # user_prompt = "Please analyze the code dynamically and track its behavior."
                # run_id_analysis, thread_id_analysis = openai_assistants_api_instance.start_analysis(assistant_id,
                #                                                                                     file_ids,
                #                                                                                     analysis_message=user_prompt,
                #                                                                                     temperature=temperature)
                #
                # if openai_assistants_api_instance.wait_for_run_completion(run_id_analysis, thread_id_analysis):
                #     result = openai_assistants_api_instance.get_run_results(run_id_analysis, thread_id_analysis)
                #     print("".join(result))

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

    def run(self):
        # print("llm_model:", self.ctrl_params["llm_model"])
        # print("temperature", self.ctrl_params["temperature"])
        # print("language:", {self.ctrl_params['language']})
        # print("num_history_cnt:", self.ctrl_params["num_history_cnt"])
        # print(f"use_dynamic_chunk_size: {self.ctrl_params['use_dynamic_chunk_size']} --> only support .py file")
        # print("max_limit_token:", self.ctrl_params["max_limit_token"])
        # print("min_limit_token:", self.ctrl_params["min_limit_token"])
        # print("project_src_file:", self.ctrl_params["project_src_file"])
        # print("system_prompt:\n", self.ctrl_params["system_prompt"])
        # print("user_prompt:\n", self.ctrl_params["user_prompt"])

        if self.ctrl_params["use_assistant_api"]:
            self.result = self.openai_assistant_api()  # file을 통채로 openai에 던져 주고 알아서 분석하라고 함.
        else:
            self.result = self.openai_standard_api()    # file의 contents를 파싱, 청킹, 분석 요청 일련의 과정 수행

        analysis_result = "\n\n".join(self.result)
        self.finished_analyze_sig.emit(analysis_result)

    def stop(self):
        self.running = False
        # print("called code analsysis stop", self.openai_instance)

        # OpenAI 세션을 안전하게 종료
        if self.openai_instance is not None:
            try:
                self.openai_instance.close()
            except Exception as e:
                handle_exception(e)

            # 세션이 종료되었으면 None으로 설정
            self.openai_instance = None

        self.quit()  # Quit the event loop
        self.wait()  # Wait for the thread to finish



class ProjectMainWindow(QtWidgets.QMainWindow, FileManager):
    def __init__(self):
        super().__init__()

        # 실행 파일이 있는 폴더에 저장할 실제 JSON 파일 경로
        control_parameter_path = os.path.join(BASE_DIR, "control_parameter.json")

        # 만약 실행 폴더에 control_parameter.json이 없으면, 임시 폴더에서 복사
        if not os.path.exists(control_parameter_path):
            original_path = os.path.join(RESOURCE_DIR, "source", "control_parameter.json")
            shutil.copyfile(original_path, control_parameter_path)

        _, self.CONFIG_PARAMS = self.json_load_f(control_parameter_path, use_encoding=False)

        self.llm_analyze_instance = None
        self.t_load_project = None
        self.tree_view = None
        self.last_selected_index = None
        self.file_model = None
        self.work_progress = None

        self.llm_radio_buttons = []  # 라디오 버튼을 저장할 리스트
        """ for main frame & widget """
        self.mainFrame_ui = None
        self.widget_ui = None
        self.dialog = None

        # 기존 UI 로드
        rt = load_module_func(module_name="source.ui_designer.main_frame")
        self.mainFrame_ui = rt.Ui_MainWindow()
        self.mainFrame_ui.setupUi(self)
        self.mainFrame_ui.explorer_frame.setMinimumWidth(300)  # 최소 너비 설정
        # self.mainFrame_ui.explorer_frame.hide()
        self.mainFrame_ui.explore_pushButton.setText("Hide")

        self.setupGPTModels()
        self.setDefaultPrompt()
        self.setDefaultUserContent()
        self.remove_all_directory_sequentially(t_path=BASE_DIR, t_name="root_temp_")

        # 탐색기 뷰 추가
        if self.tree_view is None:
            self.file_model = QFileSystemModel()

            self.tree_view = QtWidgets.QTreeView()
            self.tree_view.setModel(self.file_model)
            self.tree_view.clicked.connect(self.on_item_selected)

            use_style = True
            if use_style:
                # 스타일 시트 설정
                self.tree_view.setStyleSheet("""
                            QTreeView::item:selected {
                                background-color: red; /* 선택된 항목 배경색 */
                                color: white; /* 선택된 항목 텍스트 색상 */
                            }
                            QTreeView::item {
                                background-color: transparent; /* 기본 항목 배경색 */
                                color: black; /* 기본 항목 텍스트 색상 */
                            }
                        """)

            # 초기화 시 루트를 설정하지 않음
            self.tree_view.setHeaderHidden(False)  # 헤더는 보이게 설정

            # 가로 스크롤바 항상 보이도록 설정
            self.tree_view.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)

            # 수평 스크롤바가 스크롤 단위별로 동작하도록 설정
            self.tree_view.setHorizontalScrollMode(QtWidgets.QAbstractItemView.ScrollPerPixel)

            # 수평 크기 조정 가능하도록 설정
            self.tree_view.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
            self.tree_view.header().setSectionResizeMode(QtWidgets.QHeaderView.Interactive)  # 컬럼 크기 조정 가능
            self.tree_view.header().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)

            # 스크롤 영역에 추가
            self.mainFrame_ui.explorer_scrollArea.setWidget(self.tree_view)

        self.mainFrame_ui.tabWidget.setTabVisible(2, False)
        self.mainFrame_ui.tabWidget.setTabVisible(3, False)
        self.setWindowTitle(self.CONFIG_PARAMS["Version"])

    def closeEvent(self, event):
        answer = QtWidgets.QMessageBox.question(self,
                                                "Confirm Exit...",
                                                "Are you sure you want to exit?\nAll data will be lost.",
                                                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)

        if answer == QtWidgets.QMessageBox.Yes:
            find_and_stop_qthreads()
            stop_all_threads()

            event.accept()
        else:
            event.ignore()

    # def normalOutputWritten(self, text):
    #     cursor = self.mainFrame_ui.logtextbrowser.textCursor()
    #     cursor.movePosition(QtGui.QTextCursor.End)

    #     # 기본 글자 색상 설정
    #     color_format = cursor.charFormat()
    #     color_format.setForeground(QtCore.Qt.red if "><" in text else QtCore.Qt.black)

    #     cursor.setCharFormat(color_format)
    #     cursor.insertText(text)

    #     # 커서를 최신 위치로 업데이트
    #     self.mainFrame_ui.logtextbrowser.setTextCursor(cursor)
    #     self.mainFrame_ui.logtextbrowser.ensureCursorVisible()

    def cleanPromptBrowser(self):
        self.mainFrame_ui.prompt_window.clear()

    def getLanguage(self):
        language = "English"

        if self.mainFrame_ui.korean_radioButton.isChecked():
            language = "Korean"

        return language

    def getLLMPrompt(self):
        user_text = self.mainFrame_ui.prompt_window.toPlainText()  # QTextBrowser에서 전체 텍스트 가져오기
        # combined_text = "".join(text)
        system_text = self.mainFrame_ui.systemlineEdit.text()  # QTextBrowser에서 전체 텍스트 가져오기

        return system_text, user_text

    def getUserContents(self):
        text = self.mainFrame_ui.user_textEdit.toPlainText()

        return text

    def getSummaryResult(self):
        text = self.mainFrame_ui.llmresult_textEdit.toPlainText()  # QTextBrowser에서 전체 텍스트 가져오기
        # combined_text = "".join(text)

        return text

    def getChunkResult(self):
        text = self.mainFrame_ui.chunk_textEdit.toPlainText()  # QTextBrowser에서 전체 텍스트 가져오기
        # combined_text = "".join(text)

        return text

    def save_prompt(self):
        system_prompt, user_prompt = self.getLLMPrompt()

        self.CONFIG_PARAMS["prompt"]["user"] = [user_prompt]
        self.CONFIG_PARAMS["prompt"]["system"] = [system_prompt]
        # control_parameter_path = os.path.join(BASE_DIR, "source", "control_parameter.json")
        control_parameter_path = os.path.join(BASE_DIR, "control_parameter.json")
        self.json_dump_f(file_path=control_parameter_path, data=self.CONFIG_PARAMS, use_encoding=False)

    def get_prompt(self):
        user_text = self.CONFIG_PARAMS["prompt"]["user"]
        self.mainFrame_ui.prompt_window.setText("".join(user_text))

        system_text = self.CONFIG_PARAMS["prompt"]["system"]
        self.mainFrame_ui.systemlineEdit.setText("".join(system_text))

    def connectSlotSignal(self):
        """ sys.stdout redirection """
        # sys.stdout = EmittingStream(textWritten=self.normalOutputWritten)
        self.mainFrame_ui.prompt_clear_pushButton.clicked.connect(self.cleanPromptBrowser)

        # self.mainFrame_ui.actionOn.triggered.connect(self.log_browser_ctrl)
        # self.mainFrame_ui.actionOff.triggered.connect(self.log_browser_ctrl)

        self.mainFrame_ui.explore_pushButton.clicked.connect(self.explore_window_ctrl)

        self.mainFrame_ui.actionOpen_Fold.triggered.connect(self.open_directory)

        self.mainFrame_ui.analyze_pushButton.clicked.connect(self.start_analyze)

        # self.mainFrame_ui.deselectpushButton.clicked.connect(self.deselect_file_dir)

        self.mainFrame_ui.save_pushButton.clicked.connect(self.save_prompt)

        self.mainFrame_ui.get_pushButton.clicked.connect(self.get_prompt)

        self.mainFrame_ui.codeview_pushButton.clicked.connect(self.view_code)

    def setupGPTModels(self):
        row = 0  # 그리드 레이아웃의 첫 번째 행
        for index, name in enumerate(self.CONFIG_PARAMS["llm_company"][self.CONFIG_PARAMS["select_llm"]]["llm_models"]):
            radio_button = QRadioButton(name)  # 라디오 버튼 생성
            if index == 0:  # 첫 번째 요소는 기본적으로 체크되도록 설정
                radio_button.setChecked(True)
            self.mainFrame_ui.modelgridLayout.addWidget(radio_button, row, 0)  # 그리드 레이아웃에 추가
            self.llm_radio_buttons.append(radio_button)
            radio_button.clicked.connect(self.getSelectedModel)
            row += 1  # 행 번호 증가

    def setDefaultPrompt(self):
        user_prompt = "".join(self.CONFIG_PARAMS["prompt"]["user"])  # 리스트 요소를 줄바꿈(\n)으로 합치기
        self.mainFrame_ui.prompt_window.setText(user_prompt)

        system_prompt = "".join(self.CONFIG_PARAMS["prompt"]["system"])  # 리스트 요소를 줄바꿈(\n)으로 합치기
        self.mainFrame_ui.systemlineEdit.setText(system_prompt)

    def setDefaultUserContent(self):
        text = self.CONFIG_PARAMS["example_content"]
        self.mainFrame_ui.user_textEdit.setPlainText("".join(text))

    def getSelectedModel(self):
        # 선택된 라디오 버튼이 무엇인지 확인
        for radio_button in self.llm_radio_buttons:
            if radio_button.isChecked():  # 선택된 버튼을 확인                
                PRINT_(f"Selected GPT model: {radio_button.text()}")  # 선택된 버튼의 텍스트 출력
                return radio_button.text()

    def view_code(self):
        """Handle double-click event on a file in the QTreeView."""
        if self.last_selected_index is None:
            return

        file_path = self.file_model.filePath(self.last_selected_index)
        if os.path.isdir(file_path):
            PRINT_("Select File")
            return

        rt = load_module_func(module_name="source.ui_designer.fileedit")

        self.dialog = QtWidgets.QDialog()
        ui = rt.Ui_Dialog()
        ui.setupUi(self.dialog)
        self.dialog.setWindowTitle(f"{file_path}")  # 원하는 제목을 설정
        ui.fileedit_save.hide()

        # 시그널 슬롯 연결 람다 사용해서 직접 인자를 넘기자...........
        ui.fileedit_cancel.clicked.connect(self.dialog.close)

        # QSyntaxHighlighter를 통해 ':'로 끝나는 줄을 파란색으로 강조 표시
        # highlighter = ColonLineHighlighter(ui.textEdit.document())

        try:
            with open(file_path, "r", encoding="utf-8") as file:
                line = file.readlines()  # 모든 줄을 리스트로 반환

        except Exception as e:
            PRINT_(f"파일을 여는 중 오류 발생: {e}")
            PRINT_(traceback.format_exc())  # 전체 오류 트레이스백 출력
            sys.exit(1)  # 프로그램 종료 (1은 오류 코드, 0은 정상 종료)

        model_config_str = "".join(line)

        ui.textEdit.setPlainText(model_config_str)  # 변환된 문자열 설정
        self.dialog.setWindowModality(QtCore.Qt.ApplicationModal)
        self.dialog.show()

    def explore_window_ctrl(self, always_show=False):
        if always_show:
            self.mainFrame_ui.explorer_frame.show()
            self.mainFrame_ui.explore_pushButton.setText("Hide")
            return

        if self.mainFrame_ui.explorer_frame.isVisible():
            self.mainFrame_ui.explorer_frame.hide()
            self.mainFrame_ui.explore_pushButton.setText("Show")
        else:
            self.mainFrame_ui.explorer_frame.show()
            self.mainFrame_ui.explore_pushButton.setText("Hide")

    def ctrl_meta_info(self, show=True):
        if not show:
            # 파일 크기(1), 파일 형식(2), 날짜 수정(3) 컬럼 숨기기
            self.tree_view.setColumnHidden(1, True)  # 크기 숨김
            self.tree_view.setColumnHidden(2, True)  # 형식 숨김
            self.tree_view.setColumnHidden(3, True)  # 날짜 숨김

    def deselect_file_dir(self):
        # PRINT_(f"Deselected file/folder")

        # 기존 선택 초기화
        self.tree_view.clearSelection()  # 기존 선택된 항목 해제
        self.tree_view.setCurrentIndex(QModelIndex())  # 현재 인덱스 초기화

    def on_item_selected(self, index: QtCore.QModelIndex):
        """파일 또는 폴더 선택/해제 (토글 기능)"""
        if not index.isValid():
            return

        # 클릭한 항목이 이전에 선택한 것과 같다면 -> 선택 해제
        if self.last_selected_index == index:
            self.tree_view.clearSelection()  # 선택 해제
            self.tree_view.setCurrentIndex(QtCore.QModelIndex())  # 현재 선택 초기화
            self.last_selected_index = None  # 저장된 선택 해제
            self.mainFrame_ui.user_textEdit.setEnabled(True)

        else:
            # 새로운 항목 선택
            self.last_selected_index = index
            # 예제 UI 요소 (텍스트 입력 활성화)
            self.mainFrame_ui.user_textEdit.setEnabled(False)

    def finished_load_thread(self, m_dir=None):
        # 작업 진행 표시가 있으면 닫기
        if self.work_progress is not None:
            self.work_progress.close()

        # 선택된 항목 초기화
        self.last_selected_index = None  # 저장된 선택 해제
        self.deselect_file_dir()  # 폴더나 파일 선택 해제 함수 호출

        # m_dir이 None인 경우, 종료
        if m_dir is None:
            return

        # 선택한 폴더를 탐색기에서 갱신
        self.file_model.setRootPath(m_dir)  # 새 루트 경로 설정

        # 새로운 디렉토리 경로에 대한 인덱스를 가져오기
        index = self.file_model.index(m_dir)

        # 선택한 폴더가 유효한지 확인
        if index.isValid():
            PRINT_("valid directory index.", m_dir)

            # 파일 시스템 모델을 새로 설정하고 루트 경로를 갱신
            self.tree_view.setModel(self.file_model)  # 새 모델 설정

            # 루트 경로 설정 (디렉토리 변경)
            self.tree_view.setRootIndex(index)  # 새로운 루트 경로 설정

            # 선택 항목 초기화: 선택된 항목 없도록 설정
            self.tree_view.clearSelection()  # 기존 선택 항목 해제
            self.tree_view.setCurrentIndex(QtCore.QModelIndex())  # 선택된 항목 초기화

            # 모델을 명시적으로 갱신 (UI 업데이트)
            self.tree_view.viewport().update()

            # 탐색기 메타 정보 표시
            self.ctrl_meta_info(show=True)

            # 탐색기 뷰가 이미 레이아웃에 추가된 경우 중복 추가 방지
            if self.tree_view.parent() is None:
                self.mainFrame_ui.explorer_verticalLayout.addWidget(self.tree_view)

            # 탐색기 윈도우 표시 조정
            self.explore_window_ctrl(always_show=True)

        else:
            PRINT_("Error: Invalid directory index.", m_dir)

    def x_finished_load_thread(self, m_dir=None):
        if self.work_progress is not None:
            self.work_progress.close()

        # 선택된 항목 초기화
        # self.on_item_selected(QtCore.QModelIndex())
        self.last_selected_index = None  # 저장된 선택 해제
        self.deselect_file_dir()

        if m_dir is None:
            return

        # 선택한 폴더를 탐색기에서 갱신
        self.file_model.setRootPath(m_dir)  # 루트 경로 설정

        # QFileSystemModel이 이미 동일한 경로에 대해 캐시된 데이터를 가지고 있으면 갱신되지 않음
        index = self.file_model.index(m_dir)  # m_dir의 인덱스를 가져옴

        # 선택한 폴더가 유효한지 확인
        if index.isValid():
            PRINT_("valid directory index.", m_dir)

            # 모델을 새로 설정하고 루트 인덱스를 다시 설정
            self.tree_view.setModel(None)  # 기존 모델 제거
            self.tree_view.setModel(self.file_model)  # 모델을 새로 설정

            # self.tree_view.clicked.connect(self.on_item_selected)

            # 루트 경로 재설정
            self.tree_view.setRootIndex(index)  # 새로운 루트로 설정

            # 명시적으로 모델 갱신
            self.tree_view.viewport().update()

            self.ctrl_meta_info(show=True)

            # explorer_verticalLayout에 이미 추가된 경우 다시 추가하지 않음
            if self.tree_view.parent() is None:
                self.mainFrame_ui.explorer_verticalLayout.addWidget(self.tree_view)

            self.explore_window_ctrl(always_show=True)

        else:
            PRINT_("Error: Invalid directory index.", m_dir)

    def update_progressbar_label(self, file_name, value):
        if self.work_progress is not None:
            self.work_progress.onProgressTextChanged(text=file_name)
            self.work_progress.onCountChanged(value=value % self.work_progress.getProgressBarMaximumValue())

    def close_progress_dialog(self):
        #  QDialog 인 경우
        if self.work_progress is not None:
            PRINT_(self.work_progress, "close work_progress dialog")
            self.work_progress.deleteLater()
            self.work_progress = None

        #  QThread 인 경우
        if self.t_load_project is not None:
            PRINT_(self.t_load_project, "close t_load_project thread")
            if self.t_load_project.isRunning():
                self.t_load_project.stop()

            self.t_load_project.deleteLater()
            self.t_load_project = None

        if self.llm_analyze_instance is not None:
            PRINT_(self.llm_analyze_instance, "close llm_analyze_instance thread")
            if self.llm_analyze_instance.isRunning():
                self.llm_analyze_instance.stop()

            self.llm_analyze_instance.deleteLater()
            self.llm_analyze_instance = None

    def open_directory(self):
        m_dir = QFileDialog.getExistingDirectory(self, "Select Directory")

        if not m_dir:
            return

        self.last_selected_index = None  # 저장된 선택 해제

        if self.mainFrame_ui.popctrl_radioButton.isChecked():
            modal_display = False
        else:
            modal_display = True

        self.work_progress = ProgressDialog(modal=modal_display, message="Loading Selected Project Files", show=True,
                                            remove_percent_sign=True)
        self.work_progress.progress_stop_sig.connect(self.close_progress_dialog)

        self.t_load_project = LoadDirectoryThread(m_source_dir=m_dir, BASE_DIR=BASE_DIR,
                                                  keyword_filter=self.CONFIG_PARAMS["filter"])
        self.t_load_project.finished_load_project_sig.connect(self.finished_load_thread)
        self.t_load_project.copy_status_sig.connect(self.update_progressbar_label)

        self.t_load_project.start()

        self.work_progress.show_progress()

    @staticmethod
    def saveTestResult(message):
        # 현재 날짜 및 시간을 'YYYYMMDD_HHMMSS' 형식으로 생성
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"result_{timestamp}.md"

        file_path = os.path.join(BASE_DIR, "Result", file_name).replace("\\", "/")

        # 파일 저장
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(message)

    def llm_analyze_result(self, summary_message):
        self.mainFrame_ui.tabWidget.setCurrentIndex(1)

        # summary = f"# [Summary Result]\n\n{summary_message}\n\n\n"
        # self.mainFrame_ui.llmresult_textEdit.setMarkdown(summary)
        #
        # detailed_summary = "# " + self.getChunkResult()
        # overall_report = f"{summary}\n\n\n{detailed_summary}\n\n-End-"

        summary = f"# [Analysis Result]\n\n{summary_message}\n\n\n"
        self.mainFrame_ui.llmresult_textEdit.setMarkdown(summary)
        detailed_summary = ""
        overall_report = f"{summary}\n\n\n{detailed_summary}\n\n-End-"

        self.saveTestResult(message=overall_report)

        if self.work_progress is not None:
            self.work_progress.close()

    def chunking_result(self, chunk_data):
        summary = f"# [Detailed Results]\n\n{chunk_data}\n\n\n"
        self.mainFrame_ui.chunk_textEdit.setMarkdown(summary)

    def start_analyze(self):
        selected_indexes = self.tree_view.selectedIndexes()

        file_path = None
        user_contents = ""

        if selected_indexes:
            file_path = self.file_model.filePath(selected_indexes[0])
        else:
            answer = QtWidgets.QMessageBox.question(self,
                                                    "Information ...",
                                                    "분석할 코드 폴더를 선택하지 않았습니다.\n Input Contents(code, text ...) 내용으로 진행 할 까요?",
                                                    QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)

            if answer == QtWidgets.QMessageBox.No:
                return

            user_contents = self.getUserContents()
            if not user_contents.split():
                answer = QtWidgets.QMessageBox.information(self,
                                                           "Information ...",
                                                           "분석할 Contents가 존재하지 않습니다",
                                                           QtWidgets.QMessageBox.Yes)
                return

        # 폴더가 없으면 생성
        self.mainFrame_ui.llmresult_textEdit.clear()
        self.mainFrame_ui.chunk_textEdit.clear()
        self.mainFrame_ui.embed_textEdit.clear()
        self.mainFrame_ui.tabWidget.setCurrentIndex(1)

        result_dir = os.path.join(BASE_DIR, "Result").replace("\\", "/")
        os.makedirs(result_dir, exist_ok=True)

        llm_model = self.getSelectedModel()
        system_prompt, user_prompt = self.getLLMPrompt()
        language = self.getLanguage()

        llm_key = os.getenv("OPENAI_API_KEY")
        if llm_key is None:
            llm_key = "".join(self.CONFIG_PARAMS["llm_company"][self.CONFIG_PARAMS["select_llm"]]["key"])

        timeout = int(self.mainFrame_ui.timeoutlineEdit.text())

        PRINT_("[Info] LLM Model")
        PRINT_(f"-->{llm_model}")
        PRINT_("[Info] Using System Prompt")
        PRINT_(f"-->{system_prompt}")
        PRINT_("[Info] Using User Prompt")
        PRINT_(f"-->{user_prompt}")
        PRINT_("[Info] Time Out")
        PRINT_(f"-->{timeout} s")

        ctrl_params = {
            "project_src_file": file_path,
            "user_contents": user_contents,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "llm_model": llm_model,
            "llm_key": llm_key,
            "max_limit_token": self.CONFIG_PARAMS["llm_company"][self.CONFIG_PARAMS["select_llm"]]["max_limit_token"],
            "min_limit_token": self.CONFIG_PARAMS["llm_company"][self.CONFIG_PARAMS["select_llm"]]["min_limit_token"],
            "use_dynamic_chunk_size": self.CONFIG_PARAMS["llm_company"][self.CONFIG_PARAMS["select_llm"]]["use_dynamic_chunk_size"],
            "temperature": self.CONFIG_PARAMS["llm_company"][self.CONFIG_PARAMS["select_llm"]]["temperature"],
            "num_history_cnt": self.CONFIG_PARAMS["llm_company"][self.CONFIG_PARAMS["select_llm"]]["num_history_cnt"],
            "timeout": int(timeout),
            "language": language,
            "use_assistant_api": self.CONFIG_PARAMS["llm_company"][self.CONFIG_PARAMS["select_llm"]]["use_assistant_api"]
        }

        if self.mainFrame_ui.popctrl_radioButton.isChecked():
            modal_display = False
        else:
            modal_display = True

        self.work_progress = ProgressDialog(modal=modal_display,
                                            message="Analyzing Selected Project Files",
                                            show=True,
                                            remove_percent_sign=True,
                                            progress_increment_by_self_cnt=True
                                            )
        self.work_progress.progress_stop_sig.connect(self.close_progress_dialog)

        # self.llm_analyze_instance = CodeAnalysisThreadVersion1(ctrl_params=ctrl_params)
        self.llm_analyze_instance = CodeAnalysisThreadVersion2(ctrl_params=ctrl_params)

        # self.llm_analyze_instance = RequestLLMThread(ctrl_params=ctrl_params)

        self.llm_analyze_instance.finished_analyze_sig.connect(self.llm_analyze_result)
        self.llm_analyze_instance.chunk_analyzed_sig.connect(self.chunking_result)
        # 분석 진행 상태를 업데이트하는 신호를 연결
        self.llm_analyze_instance.analysis_progress_sig.connect(self.work_progress.onProgressTextChanged)

        self.llm_analyze_instance.start()

        self.work_progress.show_progress()


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)  # QApplication 생성 (필수)

    app.setStyle("Fusion")
    ui = ProjectMainWindow()
    ui.showMaximized()
    ui.connectSlotSignal()

    sys.exit(app.exec_())
