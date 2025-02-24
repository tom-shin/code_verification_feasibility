#!/usr/bin/env python3

from source.__init__ import *

if getattr(sys, 'frozen', False):  # PyInstaller로 패키징된 경우
    BASE_DIR = os.path.dirname(sys.executable)  # 실행 파일이 있는 폴더
    RESOURCE_DIR = sys._MEIPASS  # 임시 폴더(내부 리소스 저장됨)
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    RESOURCE_DIR = BASE_DIR  # 개발 환경에서는 현재 폴더 사용


class ProjectMainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        # 실행 파일이 있는 폴더에 저장할 실제 JSON 파일 경로
        control_parameter_path = os.path.join(BASE_DIR, "control_parameter.json")

        # 만약 실행 폴더에 control_parameter.json이 없으면, 임시 폴더에서 복사
        if not os.path.exists(control_parameter_path):
            original_path = os.path.join(RESOURCE_DIR, "source", "control_parameter.json")
            shutil.copyfile(original_path, control_parameter_path)

        _, self.CONFIG_PARAMS = json_load_f(control_parameter_path, use_encoding=False)

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
        cleanup_root_temp_folders(BASE_DIR=BASE_DIR)

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
        json_dump_f(file_path=control_parameter_path, data=self.CONFIG_PARAMS, use_encoding=False)

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
                                            unknown_max_limit=True)
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

        summary = f"# [Summary Result]\n\n{summary_message}\n\n\n"
        self.mainFrame_ui.llmresult_textEdit.setMarkdown(summary)

        detailed_summary = "# " + self.getChunkResult()
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
            "max_token_limit": self.CONFIG_PARAMS["llm_company"][self.CONFIG_PARAMS["select_llm"]]["max_limit_token"],
            "timeout": int(timeout),
            "language": language
        }

        if self.mainFrame_ui.popctrl_radioButton.isChecked():
            modal_display = False
        else:
            modal_display = True

        self.work_progress = ProgressDialog(modal=modal_display,
                                            message="Analyzing Selected Project Files",
                                            show=True,
                                            unknown_max_limit=True,
                                            on_count_changed_params_itself=True
                                            )
        self.work_progress.progress_stop_sig.connect(self.close_progress_dialog)

        self.llm_analyze_instance = CodeAnalysisThread(ctrl_params=ctrl_params)
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
