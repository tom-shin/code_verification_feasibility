#!/usr/bin/env python3

from source.__init__ import *

if getattr(sys, 'frozen', False):
    BASE_DIR = sys._MEIPASS
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 일반 Python 스크립트 실행 시


def load_module_func(module_name):
    mod = __import__(f"{module_name}", fromlist=[module_name])
    return mod


class EmittingStream(QObject):
    textWritten = pyqtSignal(str)

    def write(self, text):
        self.textWritten.emit(str(text))

    def flush(self):
        pass


class Project_MainWindow(QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()

        self.tree_view = None
        self.file_model = None
        self.work_progress = None

        self.llm_radio_buttons = []  # 라디오 버튼을 저장할 리스트
        """ for main frame & widget """
        self.mainFrame_ui = None
        self.widget_ui = None

        # 기존 UI 로드
        rt = load_module_func(module_name="source.ui_designer.main_frame")
        self.mainFrame_ui = rt.Ui_MainWindow()
        self.mainFrame_ui.setupUi(self)
        self.mainFrame_ui.explorer_frame.setMinimumWidth(300)  # 최소 너비 설정
        # self.mainFrame_ui.explorer_verticalLayout.setStretch(0, 1)  # QTreeView가 수평으로 확장될 수 있도록 설정
        self.mainFrame_ui.explorer_frame.hide()
        self.mainFrame_ui.explore_pushButton.setText("Show")

        self.setupGPTModels()
        self.setDefaultPrompt()

        # 탐색기 뷰 추가
        if self.tree_view is None:
            self.file_model = QFileSystemModel()

            self.tree_view = QtWidgets.QTreeView()
            self.tree_view.setModel(self.file_model)
            self.tree_view.setRootIndex(self.file_model.index(QtCore.QDir.rootPath()))
            self.tree_view.setHeaderHidden(True)  # 헤더 감추기

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

        self.setWindowTitle(Version)

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

    def normalOutputWritten(self, text):
        cursor = self.mainFrame_ui.logtextbrowser.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)

        # 기본 글자 색상 설정
        color_format = cursor.charFormat()
        color_format.setForeground(QtCore.Qt.red if "><" in text else QtCore.Qt.black)

        cursor.setCharFormat(color_format)
        cursor.insertText(text)

        # 커서를 최신 위치로 업데이트
        self.mainFrame_ui.logtextbrowser.setTextCursor(cursor)
        self.mainFrame_ui.logtextbrowser.ensureCursorVisible()

    def cleanLogBrowser(self):
        self.mainFrame_ui.logtextbrowser.clear()

    def log_browser_ctrl(self):
        sender = self.sender()
        if sender:
            if sender.objectName() == "actionOff":
                self.mainFrame_ui.logtextbrowser.hide()
            else:
                self.mainFrame_ui.logtextbrowser.show()

    def connectSlotSignal(self):
        """ sys.stdout redirection """
        # sys.stdout = EmittingStream(textWritten=self.normalOutputWritten)
        self.mainFrame_ui.log_clear_pushButton.clicked.connect(self.cleanLogBrowser)

        self.mainFrame_ui.actionOn.triggered.connect(self.log_browser_ctrl)
        self.mainFrame_ui.actionOff.triggered.connect(self.log_browser_ctrl)

        self.mainFrame_ui.explore_pushButton.clicked.connect(self.explore_window_ctrl)

        self.mainFrame_ui.actionOpen_Fold.triggered.connect(self.open_directory)

        self.mainFrame_ui.analyze_pushButton.clicked.connect(self.start_analyze)

        self.mainFrame_ui.deselectpushButton.clicked.connect(self.deselect_file_dir)

        # Connect double-click signal to handler
        # self.tree_view.doubleClicked.connect(self.file_double_clicked)

    def setupGPTModels(self):
        row = 0  # 그리드 레이아웃의 첫 번째 행
        for index, name in enumerate(keyword["gpt_models"]):
            radio_button = QRadioButton(name)  # 라디오 버튼 생성
            if index == 0:  # 첫 번째 요소는 기본적으로 체크되도록 설정
                radio_button.setChecked(True)
            self.mainFrame_ui.modelgridLayout.addWidget(radio_button, row, 0)  # 그리드 레이아웃에 추가
            self.llm_radio_buttons.append(radio_button)
            radio_button.clicked.connect(self.getSelectedModel)
            row += 1  # 행 번호 증가

    def setDefaultPrompt(self):
        prompt = "\n".join(keyword["pre_prompt"])  # 리스트 요소를 줄바꿈(\n)으로 합치기
        self.mainFrame_ui.logtextbrowser.setText(prompt)

    def getSelectedModel(self):
        # 선택된 라디오 버튼이 무엇인지 확인
        for radio_button in self.llm_radio_buttons:
            if radio_button.isChecked():  # 선택된 버튼을 확인
                print(f"Selected GPT model: {radio_button.text()}")  # 선택된 버튼의 텍스트 출력

    # def file_double_clicked(self, index):
    #     """Handle double-click event on a file in the QTreeView."""
    #     file_path = self.file_model.filePath(index)
    #
    #     if self.file_model.isDir(index):
    #         PRINT_("Double-clicked directory:", file_path)
    #         # Handle directory (e.g., open it, show contents, etc.)
    #     else:
    #         PRINT_("Double-clicked file:", file_path)
    #         try:
    #             with open(file_path, "r", encoding="utf-8") as file:
    #                 while True:
    #                     line = file.readline()  # 한 줄씩 읽기
    #                     if not line:
    #                         break  # 더 이상 읽을 줄이 없으면 종료
    #                     cleaned_sentence = ANSI_ESCAPE.sub('', line)
    #                     print(cleaned_sentence)  # 각 줄을 처리
    #
    #         except FileNotFoundError:
    #             PRINT_("Error: The file was not found.")
    #         except PermissionError:
    #             PRINT_("Error: Permission denied.")
    #         except Exception as e:
    #             PRINT_(f"An unexpected error occurred: {e}")

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

    def finished_open_dir(self):
        if self.work_progress is not None:
            self.work_progress.close()

    def ctrl_meta_info(self, show=True):
        if not show:
            # 파일 크기(1), 파일 형식(2), 날짜 수정(3) 컬럼 숨기기
            self.tree_view.setColumnHidden(1, True)  # 크기 숨김
            self.tree_view.setColumnHidden(2, True)  # 형식 숨김
            self.tree_view.setColumnHidden(3, True)  # 날짜 숨김

    def open_directory(self):
        m_dir = QFileDialog.getExistingDirectory(self, "Select Directory")

        if not m_dir:
            return

        # PRINT_("Open: ", m_dir)

        self.deselect_file_dir()

        # 선택한 폴더를 탐색기에서 갱신
        self.file_model.setRootPath(m_dir)  # 초기 루트 경로 설정
        self.tree_view.setRootIndex(self.file_model.index(m_dir))
        self.ctrl_meta_info(show=True)

        # 기존에 explorer_verticalLayout에 추가된 경우 다시 추가하지 않음
        if self.tree_view.parent() is None:
            self.mainFrame_ui.explorer_verticalLayout.addWidget(self.tree_view)

        self.explore_window_ctrl(always_show=True)

    def deselect_file_dir(self):
        PRINT_(f"Deselected file/folder")

        # 기존 선택 초기화
        self.tree_view.clearSelection()  # 기존 선택된 항목 해제
        self.tree_view.setCurrentIndex(QModelIndex())  # 현재 인덱스 초기화

    def start_analyze(self):
        selected_indexes = self.tree_view.selectedIndexes()
        if selected_indexes:
            file_path = self.file_model.filePath(selected_indexes[0])
            PRINT_(f"Selected file/folder: {file_path}")
        else:
            PRINT_("No file or folder selected.")

    def check_all_scenario(self):
        sender = self.sender()
        check = False

        if sender:
            if sender.objectName() == "all_check_scenario":
                check = True
            elif sender.objectName() == "all_uncheck_scenario":
                check = False

    def save_result(self):
        pass


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)  # QApplication 생성 (필수)

    app.setStyle("Fusion")
    ui = Project_MainWindow()
    ui.showMaximized()
    ui.connectSlotSignal()

    sys.exit(app.exec_())
