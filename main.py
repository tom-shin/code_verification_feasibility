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
        self.t_open_dir_instance = None
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

        # 탐색기 뷰 추가
        if self.tree_view is None:
            self.file_model = QFileSystemModel()
            self.file_model.setRootPath(QtCore.QDir.rootPath())  # 초기 루트 경로 설정

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

    @staticmethod
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

    @staticmethod
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

    def closeEvent(self, event):
        answer = QtWidgets.QMessageBox.question(self,
                                                "Confirm Exit...",
                                                "Are you sure you want to exit?\nAll data will be lost.",
                                                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)

        if answer == QtWidgets.QMessageBox.Yes:
            event.accept()
            self.find_and_stop_qthreads()
            self.stop_all_threads()
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

    def connectSlotSignal(self):
        """ sys.stdout redirection """
        sys.stdout = EmittingStream(textWritten=self.normalOutputWritten)
        self.mainFrame_ui.log_clear_pushButton.clicked.connect(self.cleanLogBrowser)

        self.mainFrame_ui.actionOn.triggered.connect(self.log_browser_ctrl)
        self.mainFrame_ui.actionOff.triggered.connect(self.log_browser_ctrl)

        self.mainFrame_ui.explore_pushButton.clicked.connect(self.explore_window_ctrl)

        self.mainFrame_ui.actionOpen_Fold.triggered.connect(self.open_directory)

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

    def log_browser_ctrl(self):
        sender = self.sender()
        if sender:
            if sender.objectName() == "actionOff":
                self.mainFrame_ui.logtextbrowser.hide()
            else:
                self.mainFrame_ui.logtextbrowser.show()

    def finished_open_dir(self):
        if self.t_open_dir_instance is not None:
            self.t_open_dir_instance.stop()

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

        PRINT_("Open: ", m_dir)

        self.t_open_dir_instance = OpenDir_WorkerThread(dirPath=m_dir)
        self.t_open_dir_instance.finished_open_dir_sig.connect(self.finished_open_dir)

        # 선택한 폴더를 탐색기에서 갱신
        self.tree_view.setRootIndex(self.file_model.index(m_dir))
        self.ctrl_meta_info(show=True)

        # 기존에 explorer_verticalLayout에 추가된 경우 다시 추가하지 않음
        if self.tree_view.parent() is None:
            self.mainFrame_ui.explorer_verticalLayout.addWidget(self.tree_view)

        self.explore_window_ctrl(always_show=True)

    def start_analyze(self):
        pass

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
