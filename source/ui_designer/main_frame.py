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
        self.scrollAreaWidgetContents_2.setGeometry(QtCore.QRect(0, 0, 69, 432))
        self.scrollAreaWidgetContents_2.setObjectName("scrollAreaWidgetContents_2")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.scrollAreaWidgetContents_2)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.explorer_scrollArea.setWidget(self.scrollAreaWidgetContents_2)
        self.verticalLayout.addWidget(self.explorer_scrollArea)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem)
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
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.gridLayout = QtWidgets.QGridLayout(self.tab_2)
        self.gridLayout.setObjectName("gridLayout")
        self.frame_2 = QtWidgets.QFrame(self.tab_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame_2.sizePolicy().hasHeightForWidth())
        self.frame_2.setSizePolicy(sizePolicy)
        self.frame_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_2.setObjectName("frame_2")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.frame_2)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.scrollArea = QtWidgets.QScrollArea(self.frame_2)
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setObjectName("scrollArea")
        self.scrollAreaWidgetContents = QtWidgets.QWidget()
        self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 1046, 580))
        self.scrollAreaWidgetContents.setObjectName("scrollAreaWidgetContents")
        self.verticalLayout_11 = QtWidgets.QVBoxLayout(self.scrollAreaWidgetContents)
        self.verticalLayout_11.setObjectName("verticalLayout_11")
        self.formLayout = QtWidgets.QFormLayout()
        self.formLayout.setObjectName("formLayout")
        self.verticalLayout_11.addLayout(self.formLayout)
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)
        self.gridLayout_2.addWidget(self.scrollArea, 0, 0, 1, 1)
        self.gridLayout.addWidget(self.frame_2, 2, 0, 1, 1)
        self.frame_3 = QtWidgets.QFrame(self.tab_2)
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
        spacerItem1 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_4.addItem(spacerItem1)
        self.gridLayout.addWidget(self.frame_3, 2, 1, 1, 1)
        self.tabWidget.addTab(self.tab_2, "")
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.tab)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.groupBox_7 = QtWidgets.QGroupBox(self.tab)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.groupBox_7.setFont(font)
        self.groupBox_7.setObjectName("groupBox_7")
        self.verticalLayout_13 = QtWidgets.QVBoxLayout(self.groupBox_7)
        self.verticalLayout_13.setObjectName("verticalLayout_13")
        self.history_table = QtWidgets.QGridLayout()
        self.history_table.setObjectName("history_table")
        self.label_14 = QtWidgets.QLabel(self.groupBox_7)
        self.label_14.setAlignment(QtCore.Qt.AlignCenter)
        self.label_14.setObjectName("label_14")
        self.history_table.addWidget(self.label_14, 0, 3, 1, 1)
        self.label_5 = QtWidgets.QLabel(self.groupBox_7)
        self.label_5.setAlignment(QtCore.Qt.AlignCenter)
        self.label_5.setObjectName("label_5")
        self.history_table.addWidget(self.label_5, 0, 0, 1, 1)
        self.label_15 = QtWidgets.QLabel(self.groupBox_7)
        self.label_15.setAlignment(QtCore.Qt.AlignCenter)
        self.label_15.setObjectName("label_15")
        self.history_table.addWidget(self.label_15, 0, 4, 1, 1)
        self.label_8 = QtWidgets.QLabel(self.groupBox_7)
        self.label_8.setAlignment(QtCore.Qt.AlignCenter)
        self.label_8.setObjectName("label_8")
        self.history_table.addWidget(self.label_8, 0, 2, 1, 1)
        self.label_6 = QtWidgets.QLabel(self.groupBox_7)
        self.label_6.setAlignment(QtCore.Qt.AlignCenter)
        self.label_6.setObjectName("label_6")
        self.history_table.addWidget(self.label_6, 0, 1, 1, 1)
        self.verticalLayout_13.addLayout(self.history_table)
        self.verticalLayout_3.addWidget(self.groupBox_7)
        spacerItem2 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_3.addItem(spacerItem2)
        self.tabWidget.addTab(self.tab, "")
        self.verticalLayout_8.addWidget(self.tabWidget)
        self.logtextbrowser = QtWidgets.QTextBrowser(self.frame)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.logtextbrowser.sizePolicy().hasHeightForWidth())
        self.logtextbrowser.setSizePolicy(sizePolicy)
        self.logtextbrowser.setReadOnly(False)
        self.logtextbrowser.setObjectName("logtextbrowser")
        self.verticalLayout_8.addWidget(self.logtextbrowser)
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.explore_pushButton = QtWidgets.QPushButton(self.frame)
        self.explore_pushButton.setObjectName("explore_pushButton")
        self.horizontalLayout_4.addWidget(self.explore_pushButton)
        self.deselectpushButton = QtWidgets.QPushButton(self.frame)
        self.deselectpushButton.setObjectName("deselectpushButton")
        self.horizontalLayout_4.addWidget(self.deselectpushButton)
        self.popctrl_radioButton = QtWidgets.QRadioButton(self.frame)
        self.popctrl_radioButton.setObjectName("popctrl_radioButton")
        self.horizontalLayout_4.addWidget(self.popctrl_radioButton)
        spacerItem3 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_4.addItem(spacerItem3)
        self.analyze_pushButton = QtWidgets.QPushButton(self.frame)
        self.analyze_pushButton.setObjectName("analyze_pushButton")
        self.horizontalLayout_4.addWidget(self.analyze_pushButton)
        self.save_pushButton = QtWidgets.QPushButton(self.frame)
        self.save_pushButton.setObjectName("save_pushButton")
        self.horizontalLayout_4.addWidget(self.save_pushButton)
        self.log_clear_pushButton = QtWidgets.QPushButton(self.frame)
        self.log_clear_pushButton.setObjectName("log_clear_pushButton")
        self.horizontalLayout_4.addWidget(self.log_clear_pushButton)
        self.verticalLayout_8.addLayout(self.horizontalLayout_4)
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
        self.groupBox.setTitle(_translate("MainWindow", "LLM Models"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("MainWindow", "Result Screen"))
        self.groupBox_7.setTitle(_translate("MainWindow", "History"))
        self.label_14.setText(_translate("MainWindow", "Repository"))
        self.label_5.setText(_translate("MainWindow", "Image"))
        self.label_15.setText(_translate("MainWindow", "Released to ThunderSoft"))
        self.label_8.setText(_translate("MainWindow", "TAG"))
        self.label_6.setText(_translate("MainWindow", "Version"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("MainWindow", "Reserved"))
        self.explore_pushButton.setText(_translate("MainWindow", "Hide"))
        self.deselectpushButton.setText(_translate("MainWindow", "Deselect"))
        self.popctrl_radioButton.setText(_translate("MainWindow", "pop control"))
        self.analyze_pushButton.setText(_translate("MainWindow", "Analyze"))
        self.save_pushButton.setText(_translate("MainWindow", "Save"))
        self.log_clear_pushButton.setText(_translate("MainWindow", "Prompt Clear"))
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
