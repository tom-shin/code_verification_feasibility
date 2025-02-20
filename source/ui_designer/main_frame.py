# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'main_frame.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1344, 953)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout_8 = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout_8.setObjectName("horizontalLayout_8")
        self.explorer_frame = QtWidgets.QFrame(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(50)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.explorer_frame.sizePolicy().hasHeightForWidth())
        self.explorer_frame.setSizePolicy(sizePolicy)
        self.explorer_frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.explorer_frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.explorer_frame.setObjectName("explorer_frame")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.explorer_frame)
        self.verticalLayout.setObjectName("verticalLayout")
        self.explorer_scrollArea = QtWidgets.QScrollArea(self.explorer_frame)
        self.explorer_scrollArea.setWidgetResizable(True)
        self.explorer_scrollArea.setObjectName("explorer_scrollArea")
        self.scrollAreaWidgetContents_2 = QtWidgets.QWidget()
        self.scrollAreaWidgetContents_2.setGeometry(QtCore.QRect(0, 0, 73, 843))
        self.scrollAreaWidgetContents_2.setObjectName("scrollAreaWidgetContents_2")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.scrollAreaWidgetContents_2)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.explorer_scrollArea.setWidget(self.scrollAreaWidgetContents_2)
        self.verticalLayout.addWidget(self.explorer_scrollArea)
        self.deselectpushButton = QtWidgets.QPushButton(self.explorer_frame)
        self.deselectpushButton.setObjectName("deselectpushButton")
        self.verticalLayout.addWidget(self.deselectpushButton)
        self.horizontalLayout_8.addWidget(self.explorer_frame)
        self.frame = QtWidgets.QFrame(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame.sizePolicy().hasHeightForWidth())
        self.frame.setSizePolicy(sizePolicy)
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.verticalLayout_8 = QtWidgets.QVBoxLayout(self.frame)
        self.verticalLayout_8.setObjectName("verticalLayout_8")
        self.tabWidget = QtWidgets.QTabWidget(self.frame)
        self.tabWidget.setObjectName("tabWidget")
        self.tab_1 = QtWidgets.QWidget()
        self.tab_1.setObjectName("tab_1")
        self.gridLayout = QtWidgets.QGridLayout(self.tab_1)
        self.gridLayout.setObjectName("gridLayout")
        self.verticalLayout_19 = QtWidgets.QVBoxLayout()
        self.verticalLayout_19.setObjectName("verticalLayout_19")
        self.label_2 = QtWidgets.QLabel(self.tab_1)
        self.label_2.setObjectName("label_2")
        self.verticalLayout_19.addWidget(self.label_2)
        self.prompt_window = QtWidgets.QTextBrowser(self.tab_1)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.prompt_window.sizePolicy().hasHeightForWidth())
        self.prompt_window.setSizePolicy(sizePolicy)
        self.prompt_window.setLineWidth(1)
        self.prompt_window.setReadOnly(False)
        self.prompt_window.setObjectName("prompt_window")
        self.verticalLayout_19.addWidget(self.prompt_window)
        self.gridLayout.addLayout(self.verticalLayout_19, 2, 0, 1, 1)
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.explore_pushButton = QtWidgets.QPushButton(self.tab_1)
        self.explore_pushButton.setObjectName("explore_pushButton")
        self.horizontalLayout_4.addWidget(self.explore_pushButton)
        self.popctrl_radioButton = QtWidgets.QRadioButton(self.tab_1)
        self.popctrl_radioButton.setObjectName("popctrl_radioButton")
        self.horizontalLayout_4.addWidget(self.popctrl_radioButton)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_4.addItem(spacerItem)
        self.analyze_pushButton = QtWidgets.QPushButton(self.tab_1)
        self.analyze_pushButton.setObjectName("analyze_pushButton")
        self.horizontalLayout_4.addWidget(self.analyze_pushButton)
        self.prompt_clear_pushButton = QtWidgets.QPushButton(self.tab_1)
        self.prompt_clear_pushButton.setObjectName("prompt_clear_pushButton")
        self.horizontalLayout_4.addWidget(self.prompt_clear_pushButton)
        self.get_pushButton = QtWidgets.QPushButton(self.tab_1)
        self.get_pushButton.setObjectName("get_pushButton")
        self.horizontalLayout_4.addWidget(self.get_pushButton)
        self.save_pushButton = QtWidgets.QPushButton(self.tab_1)
        self.save_pushButton.setObjectName("save_pushButton")
        self.horizontalLayout_4.addWidget(self.save_pushButton)
        self.gridLayout.addLayout(self.horizontalLayout_4, 3, 0, 1, 1)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.frame_2 = QtWidgets.QFrame(self.tab_1)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame_2.sizePolicy().hasHeightForWidth())
        self.frame_2.setSizePolicy(sizePolicy)
        self.frame_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_2.setObjectName("frame_2")
        self.verticalLayout_17 = QtWidgets.QVBoxLayout(self.frame_2)
        self.verticalLayout_17.setObjectName("verticalLayout_17")
        self.label_3 = QtWidgets.QLabel(self.frame_2)
        self.label_3.setObjectName("label_3")
        self.verticalLayout_17.addWidget(self.label_3)
        self.scrollArea = QtWidgets.QScrollArea(self.frame_2)
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setObjectName("scrollArea")
        self.scrollAreaWidgetContents = QtWidgets.QWidget()
        self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 880, 531))
        self.scrollAreaWidgetContents.setObjectName("scrollAreaWidgetContents")
        self.verticalLayout_11 = QtWidgets.QVBoxLayout(self.scrollAreaWidgetContents)
        self.verticalLayout_11.setObjectName("verticalLayout_11")
        self.user_textEdit = QtWidgets.QTextEdit(self.scrollAreaWidgetContents)
        self.user_textEdit.setObjectName("user_textEdit")
        self.verticalLayout_11.addWidget(self.user_textEdit)
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)
        self.verticalLayout_17.addWidget(self.scrollArea)
        self.horizontalLayout.addWidget(self.frame_2)
        self.frame_3 = QtWidgets.QFrame(self.tab_1)
        self.frame_3.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_3.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_3.setObjectName("frame_3")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.frame_3)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.groupBox = QtWidgets.QGroupBox(self.frame_3)
        self.groupBox.setObjectName("groupBox")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout(self.groupBox)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.modelgridLayout = QtWidgets.QGridLayout()
        self.modelgridLayout.setObjectName("modelgridLayout")
        self.verticalLayout_5.addLayout(self.modelgridLayout)
        self.verticalLayout_4.addWidget(self.groupBox)
        self.groupBox_2 = QtWidgets.QGroupBox(self.frame_3)
        self.groupBox_2.setObjectName("groupBox_2")
        self.verticalLayout_16 = QtWidgets.QVBoxLayout(self.groupBox_2)
        self.verticalLayout_16.setObjectName("verticalLayout_16")
        self.korean_radioButton = QtWidgets.QRadioButton(self.groupBox_2)
        self.korean_radioButton.setChecked(True)
        self.korean_radioButton.setObjectName("korean_radioButton")
        self.verticalLayout_16.addWidget(self.korean_radioButton)
        self.english_radioButton = QtWidgets.QRadioButton(self.groupBox_2)
        self.english_radioButton.setObjectName("english_radioButton")
        self.verticalLayout_16.addWidget(self.english_radioButton)
        self.verticalLayout_4.addWidget(self.groupBox_2)
        spacerItem1 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_4.addItem(spacerItem1)
        self.gridLayout_3 = QtWidgets.QGridLayout()
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.label = QtWidgets.QLabel(self.frame_3)
        self.label.setObjectName("label")
        self.gridLayout_3.addWidget(self.label, 0, 0, 1, 1)
        self.timeoutlineEdit = QtWidgets.QLineEdit(self.frame_3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.timeoutlineEdit.sizePolicy().hasHeightForWidth())
        self.timeoutlineEdit.setSizePolicy(sizePolicy)
        self.timeoutlineEdit.setObjectName("timeoutlineEdit")
        self.gridLayout_3.addWidget(self.timeoutlineEdit, 0, 1, 1, 1)
        self.verticalLayout_4.addLayout(self.gridLayout_3)
        self.horizontalLayout.addWidget(self.frame_3)
        self.gridLayout.addLayout(self.horizontalLayout, 0, 0, 1, 1)
        self.line = QtWidgets.QFrame(self.tab_1)
        self.line.setLineWidth(1)
        self.line.setMidLineWidth(20)
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.gridLayout.addWidget(self.line, 1, 0, 1, 1)
        self.tabWidget.addTab(self.tab_1, "")
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.verticalLayout_13 = QtWidgets.QVBoxLayout(self.tab_2)
        self.verticalLayout_13.setObjectName("verticalLayout_13")
        self.frame_6 = QtWidgets.QFrame(self.tab_2)
        self.frame_6.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_6.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_6.setObjectName("frame_6")
        self.verticalLayout_14 = QtWidgets.QVBoxLayout(self.frame_6)
        self.verticalLayout_14.setObjectName("verticalLayout_14")
        self.scrollArea_4 = QtWidgets.QScrollArea(self.frame_6)
        self.scrollArea_4.setWidgetResizable(True)
        self.scrollArea_4.setObjectName("scrollArea_4")
        self.scrollAreaWidgetContents_5 = QtWidgets.QWidget()
        self.scrollAreaWidgetContents_5.setGeometry(QtCore.QRect(0, 0, 1159, 809))
        self.scrollAreaWidgetContents_5.setObjectName("scrollAreaWidgetContents_5")
        self.verticalLayout_15 = QtWidgets.QVBoxLayout(self.scrollAreaWidgetContents_5)
        self.verticalLayout_15.setObjectName("verticalLayout_15")
        self.llmresult_textEdit = QtWidgets.QTextEdit(self.scrollAreaWidgetContents_5)
        self.llmresult_textEdit.setObjectName("llmresult_textEdit")
        self.verticalLayout_15.addWidget(self.llmresult_textEdit)
        self.scrollArea_4.setWidget(self.scrollAreaWidgetContents_5)
        self.verticalLayout_14.addWidget(self.scrollArea_4)
        self.verticalLayout_13.addWidget(self.frame_6)
        self.tabWidget.addTab(self.tab_2, "")
        self.tab_3 = QtWidgets.QWidget()
        self.tab_3.setObjectName("tab_3")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.tab_3)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.frame_4 = QtWidgets.QFrame(self.tab_3)
        self.frame_4.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_4.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_4.setObjectName("frame_4")
        self.verticalLayout_6 = QtWidgets.QVBoxLayout(self.frame_4)
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.scrollArea_2 = QtWidgets.QScrollArea(self.frame_4)
        self.scrollArea_2.setWidgetResizable(True)
        self.scrollArea_2.setObjectName("scrollArea_2")
        self.scrollAreaWidgetContents_3 = QtWidgets.QWidget()
        self.scrollAreaWidgetContents_3.setGeometry(QtCore.QRect(0, 0, 1159, 809))
        self.scrollAreaWidgetContents_3.setObjectName("scrollAreaWidgetContents_3")
        self.verticalLayout_7 = QtWidgets.QVBoxLayout(self.scrollAreaWidgetContents_3)
        self.verticalLayout_7.setObjectName("verticalLayout_7")
        self.chunk_textEdit = QtWidgets.QTextEdit(self.scrollAreaWidgetContents_3)
        self.chunk_textEdit.setObjectName("chunk_textEdit")
        self.verticalLayout_7.addWidget(self.chunk_textEdit)
        self.scrollArea_2.setWidget(self.scrollAreaWidgetContents_3)
        self.verticalLayout_6.addWidget(self.scrollArea_2)
        self.verticalLayout_3.addWidget(self.frame_4)
        self.tabWidget.addTab(self.tab_3, "")
        self.tab_4 = QtWidgets.QWidget()
        self.tab_4.setObjectName("tab_4")
        self.verticalLayout_9 = QtWidgets.QVBoxLayout(self.tab_4)
        self.verticalLayout_9.setObjectName("verticalLayout_9")
        self.frame_5 = QtWidgets.QFrame(self.tab_4)
        self.frame_5.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_5.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_5.setObjectName("frame_5")
        self.verticalLayout_10 = QtWidgets.QVBoxLayout(self.frame_5)
        self.verticalLayout_10.setObjectName("verticalLayout_10")
        self.scrollArea_3 = QtWidgets.QScrollArea(self.frame_5)
        self.scrollArea_3.setWidgetResizable(True)
        self.scrollArea_3.setObjectName("scrollArea_3")
        self.scrollAreaWidgetContents_4 = QtWidgets.QWidget()
        self.scrollAreaWidgetContents_4.setGeometry(QtCore.QRect(0, 0, 1159, 809))
        self.scrollAreaWidgetContents_4.setObjectName("scrollAreaWidgetContents_4")
        self.verticalLayout_12 = QtWidgets.QVBoxLayout(self.scrollAreaWidgetContents_4)
        self.verticalLayout_12.setObjectName("verticalLayout_12")
        self.embed_textEdit = QtWidgets.QTextEdit(self.scrollAreaWidgetContents_4)
        self.embed_textEdit.setObjectName("embed_textEdit")
        self.verticalLayout_12.addWidget(self.embed_textEdit)
        self.scrollArea_3.setWidget(self.scrollAreaWidgetContents_4)
        self.verticalLayout_10.addWidget(self.scrollArea_3)
        self.verticalLayout_9.addWidget(self.frame_5)
        self.tabWidget.addTab(self.tab_4, "")
        self.verticalLayout_8.addWidget(self.tabWidget)
        self.horizontalLayout_8.addWidget(self.frame)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1344, 21))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionOn = QtWidgets.QAction(MainWindow)
        self.actionOn.setObjectName("actionOn")
        self.actionOff = QtWidgets.QAction(MainWindow)
        self.actionOff.setObjectName("actionOff")
        self.actionOpen_Result_Excel = QtWidgets.QAction(MainWindow)
        self.actionOpen_Result_Excel.setObjectName("actionOpen_Result_Excel")
        self.actionOpen_Fold = QtWidgets.QAction(MainWindow)
        self.actionOpen_Fold.setObjectName("actionOpen_Fold")
        self.menuFile.addAction(self.actionOpen_Result_Excel)
        self.menuFile.addAction(self.actionOpen_Fold)
        self.menubar.addAction(self.menuFile.menuAction())

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.deselectpushButton.setText(_translate("MainWindow", "Deselect"))
        self.label_2.setText(_translate("MainWindow", "Input Your Prompt"))
        self.explore_pushButton.setText(_translate("MainWindow", "Hide"))
        self.popctrl_radioButton.setText(_translate("MainWindow", "pop control"))
        self.analyze_pushButton.setText(_translate("MainWindow", "Analyze"))
        self.prompt_clear_pushButton.setText(_translate("MainWindow", "Clear Prompt"))
        self.get_pushButton.setText(_translate("MainWindow", "Get Prompt"))
        self.save_pushButton.setText(_translate("MainWindow", "Save Prompt"))
        self.label_3.setText(_translate("MainWindow", "Input Contents(code, text ...)"))
        self.groupBox.setTitle(_translate("MainWindow", "LLM Models"))
        self.groupBox_2.setTitle(_translate("MainWindow", "Language (Answer)"))
        self.korean_radioButton.setText(_translate("MainWindow", "Korean"))
        self.english_radioButton.setText(_translate("MainWindow", "English"))
        self.label.setText(_translate("MainWindow", "Time Out [sec.]"))
        self.timeoutlineEdit.setText(_translate("MainWindow", "300"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_1), _translate("MainWindow", "Control Window"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("MainWindow", "Summary Result"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_3), _translate("MainWindow", "Summarize Chunks Result"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_4), _translate("MainWindow", "Summarize Embedded Result"))
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.actionOn.setText(_translate("MainWindow", "On"))
        self.actionOff.setText(_translate("MainWindow", "Off"))
        self.actionOpen_Result_Excel.setText(_translate("MainWindow", "Open File..."))
        self.actionOpen_Fold.setText(_translate("MainWindow", "Open Fold..."))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
