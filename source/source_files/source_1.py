import threading
import time
import re
import platform

from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import QObject, QThread

from .. import *
from ..ui_designer import main_widget


class Load_Target_Dir_Thread(QtCore.QThread):
    send_scenario_update_ui_sig = QtCore.pyqtSignal(int, str)
    send_finish_scenario_update_ui_sig = QtCore.pyqtSignal()

    def __init__(self, file_path, grand_parent):
        super().__init__()
        self.file_path = file_path
        self.grand_parent = grand_parent

    def run(self):
        for cnt, test_path in enumerate(self.file_path):
            self.send_scenario_update_ui_sig.emit(cnt, test_path)

        self.send_finish_scenario_update_ui_sig.emit()


class Template_Working_Thread(QThread):
    output_signal = pyqtSignal(str, tuple, int)
    finish_output_signal = pyqtSignal(bool)
    error_signal = pyqtSignal(str, tuple)
    send_max_progress_cnt = pyqtSignal(int)

    def __init__(self, parent=None, grand_parent=None):
        super().__init__()
        self.parent = parent
        self.grand_parent = grand_parent
        self._running = True

    def run(self):        
        pass        

    def stop(self):
        self._running = False
        self.quit()
        self.wait(3000)


class default_source_class(QObject):
    send_sig_delete_all_sub_widget = pyqtSignal()

    def __init__(self, parent, grand_parent, progress_ctrl=True):
        super().__init__()

        self.parent = parent
        self.grandparent = grand_parent
        self.progress_ctrl = progress_ctrl

        self.start_evaluation_time = None
        self.end_evaluation_time = None
        self.insert_widget_thread = None
        self.added_scenario_widgets = None
        self.all_test_path = []        
        self.progressBar = None
        self.user_error_fmt = None
        self.template_thread = None
        self.save_progress = None
        self.result_thread = None

        self.send_sig_delete_all_sub_widget.connect(self.update_all_sub_widget)

    def open_file(self):
        self.parent.mainFrame_ui.scenario_path_lineedit.setText("directory to open")

        self.clear_sub_widget()

    def clear_sub_widget(self):
        while self.parent.mainFrame_ui.formLayout.count():
            item = self.parent.mainFrame_ui.formLayout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.setParent(None)

        self.send_sig_delete_all_sub_widget.emit()

    def update_all_sub_widget(self):
        if self.progress_ctrl:
            self.progressBar = ModalLess_ProgressDialog(message="Loading Scenario")
        else:
            self.progressBar = Modal_ProgressDialog(message="Loading Scenario")

        self.progressBar.setProgressBarMaximum(max_value=len(self.all_test_path))

        self.added_scenario_widgets = []
        self.insert_widget_thread = Load_Target_Dir_Thread(self.all_test_path, self.parent)
        self.insert_widget_thread.send_scenario_update_ui_sig.connect(self.insert_widget_progress_status)
        self.insert_widget_thread.send_finish_scenario_update_ui_sig.connect(self.finish_insert_widget_progress_status)

        self.insert_widget_thread.start()

        self.progressBar.showModal_less()

    def insert_widget_progress_status(self, cnt, test_path):
        if self.progressBar is not None:
            rt = main_widget  # load_module_func(module_name="common.ui_designer.main_widget")
            widget_ui = rt.Ui_Form()
            widget_instance = QtWidgets.QWidget()
            widget_ui.setupUi(widget_instance)

            scenario = os.path.basename(test_path)
            widget_ui.scenario_checkBox.setText(f"{scenario}")
            widget_ui.pathlineEdit.setText(f"{test_path}")
            widget_ui.contexts_textEdit.setMinimumHeight(200)
            self.parent.mainFrame_ui.formLayout.setWidget(cnt, QtWidgets.QFormLayout.FieldRole,
                                                          widget_instance)

            self.added_scenario_widgets.append((widget_ui, widget_instance))
            self.progressBar.onCountChanged(value=cnt)

            widget_ui.open_terminal_pushButton.hide()

    def finish_insert_widget_progress_status(self):
        if self.progressBar is not None:
            self.progressBar.close()
            print(f"총 테스트 할 OP 갯수: {len(self.added_scenario_widgets)}")

    def update_test_result(self, output_result, sub_widget, executed_cnt):

        def highlight_text(output, words_to_highlight):
            output = output.replace("\n", "<br>")
            found_highlight = False
            # HTML 형식으로 텍스트를 변경하기 위한 함수
            for word in words_to_highlight:
                if re.search(f'({re.escape(word)})', output):  # 변환될 단어가 있는지 확인
                    found_highlight = True  # 변환할 단어가 있으면 True로 설정

                # re.escape로 단어에 특수 문자가 있을 경우에도 처리
                output = re.sub(f'({re.escape(word)})', r'<span style="color:red;">\1</span>', output)

            return found_highlight, output

        found_highlight, colored_output_result = highlight_text(output_result, self.user_error_fmt)

        sub_widget[0].contexts_textEdit.setHtml(colored_output_result)

        if found_highlight:
            sub_widget[0].lineEdit.setText("Fail")
        else:
            sub_widget[0].lineEdit.setText("Pass")

        # sub_widget[0].contexts_textEdit.setText(output_result)
        if self.progressBar is not None:
            self.progressBar.onCountChanged(value=executed_cnt)

    def error_update_test_result(self, error_message, sub_widget):
        if self.progressBar is not None:
            self.progressBar.close()

        sub_widget[0].contexts_textEdit.setText(error_message)
        sub_widget[0].lineEdit.setText("Error")

    def finish_update_test_result(self, normal_stop):
        if self.progressBar is not None:
            self.progressBar.close()

            self.end_evaluation_time = time.time()
            elapsed_time = self.end_evaluation_time - self.start_evaluation_time
            days = elapsed_time // (24 * 3600)
            remaining_secs = elapsed_time % (24 * 3600)
            hours = remaining_secs // 3600
            remaining_secs %= 3600
            minutes = remaining_secs // 60
            seconds = remaining_secs % 60

            total_time = f"{int(days)}day {int(hours)}h {int(minutes)}m {int(seconds)}s"
            msg_box = QtWidgets.QMessageBox()

            if normal_stop:
                msg_box.setWindowTitle("Test Done...")
                msg_box.setText(f"All Test Done !\nSave Button to store result data\nElapsed time: {total_time}")
            else:
                msg_box.setWindowTitle("Stop Test...")
                msg_box.setText(f"User forcibly terminated !")

            msg_box.setStandardButtons(QtWidgets.QMessageBox.Yes)
            # Always show the message box on top
            if platform.system() == "Windows":
                msg_box.setWindowFlags(msg_box.windowFlags() | QtCore.Qt.WindowStaysOnTopHint)

            # 메시지 박스를 최상단에 표시
            answer = msg_box.exec_()

    def set_max_progress_cnt(self, max_cnt):
        if self.progressBar is not None:
            self.progressBar.setProgressBarMaximum(max_value=max_cnt)

    def stop_analyze(self):
        if self.progressBar is not None:
            self.progressBar.close()

        if self.template_thread is not None:
            self.template_thread.stop()

    def op_analyze(self):
        if self.added_scenario_widgets is None:
            return

        if len(self.added_scenario_widgets) == 0:
            return

        check = False
        for cnt, target_widget in enumerate(self.added_scenario_widgets):
            if target_widget[0].scenario_checkBox.isChecked():
                check = True
                break

        if not check:
            msg_box = QtWidgets.QMessageBox()  # QMessageBox 객체 생성
            msg_box.setWindowTitle("Check Test Target")  # 대화 상자 제목
            msg_box.setText(
                "test target directory are required.\nMark target directory")
            msg_box.setStandardButtons(QtWidgets.QMessageBox.Yes)  # Yes/No 버튼 추가

            if platform.system() == "Windows":
                msg_box.setWindowFlags(msg_box.windowFlags() | QtCore.Qt.WindowStaysOnTopHint)  # 항상 위에 표시

            answer = msg_box.exec_()  # 대화 상자를 실행하고 사용자의 응답을 반환
            if answer == QtWidgets.QMessageBox.Yes:
                return

        self.start_evaluation_time = time.time()

        if self.parent.mainFrame_ui.popctrl_radioButton.isChecked():
            self.progressBar = ModalLess_ProgressDialog(message="Analyzing OP Model", show=True)
        else:
            self.progressBar = Modal_ProgressDialog(message="Analyzing OP Model", show=True)

        self.user_error_fmt = [fmt.strip() for fmt in self.grand_parent.error_lineedit.text().split(",")]

        self.template_thread = Template_Working_Thread(self, self.grand_parent)
        self.template_thread.output_signal.connect(self.update_test_result)
        self.template_thread.error_signal.connect(self.error_update_test_result)
        self.template_thread.send_max_progress_cnt.connect(self.set_max_progress_cnt)

        self.progressBar.send_user_close_event.connect(self.stop_analyze)
        self.template_thread.finish_output_signal.connect(self.finish_update_test_result)

        self.template_thread.start()

        self.progressBar.showModal_less()

    def select_all_scenario(self, check):
        if self.added_scenario_widgets is None or len(self.added_scenario_widgets) == 0:
            return

        for scenario_widget, scenario_widget_instance in self.added_scenario_widgets:
            scenario_widget.scenario_checkBox.setChecked(check)

    def save_analyze_result(self, basedir):

        def save_analyze_result_thread():
            count = 0
            for cnt, target_widget in enumerate(self.added_scenario_widgets):
                if target_widget[0].scenario_checkBox.isChecked():
                    date = GetCurrentDate()
                    pass_fail = target_widget[0].lineEdit.text()
                    test_dir = os.path.basename(target_widget[0].pathlineEdit.text())

                    # html로 저장 (color)
                    filename = f"{date}_{pass_fail}_{test_dir}"

                    test_result = target_widget[0].contexts_textEdit.toHtml()
                    file_path = os.path.join(basedir, filename)
                    save2html(file_path=file_path, data=test_result)

                    test_result = target_widget[0].contexts_textEdit.toPlainText()
                    file_path = os.path.join(basedir, filename)
                    save2txt(file_path=file_path, data=test_result)

                    count += 1
                    self.save_progress.onCountChanged(value=count)

        if self.added_scenario_widgets is None or len(self.added_scenario_widgets) == 0:
            return

        count = 0
        for cnt, target_widget in enumerate(self.added_scenario_widgets):
            if target_widget[0].scenario_checkBox.isChecked():
                count += 1
        if count == 0:
            return

        if self.parent.mainFrame_ui.popctrl_radioButton.isChecked():
            self.save_progress = ModalLess_ProgressDialog(message="Saving Result")
        else:
            self.save_progress = Modal_ProgressDialog(message="Saving Result")

        self.save_progress.setProgressBarMaximum(max_value=count)

        self.result_thread = threading.Thread(target=save_analyze_result_thread, daemon=True)
        self.result_thread.start()

        self.save_progress.showModal_less()
        self.save_progress.close()

        msg_box = QtWidgets.QMessageBox()  # QMessageBox 객체 생성
        msg_box.setWindowTitle("Save Result")  # 대화 상자 제목
        msg_box.setText(
            "All test result are saved.               \n")
        msg_box.setStandardButtons(QtWidgets.QMessageBox.Yes)  # Yes/No 버튼 추가

        if platform.system() == "Windows":
            msg_box.setWindowFlags(msg_box.windowFlags() | QtCore.Qt.WindowStaysOnTopHint)  # 항상 위에 표시

        answer = msg_box.exec_()  # 대화 상자를 실행하고 사용자의 응답을 반환
