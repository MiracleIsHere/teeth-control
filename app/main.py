import calendar
import copy
import json
import os
import shutil
import sys
import time
from random import random
from threading import Thread

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QThread, QTimer, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (QAction, QApplication, QCheckBox, QDialog,
                             QDialogButtonBox, QFileDialog, QFrame, QGroupBox,
                             QHBoxLayout, QLabel, QLineEdit, QMainWindow,
                             QMenu, QMessageBox, QSpinBox, QStatusBar,
                             QTabWidget, QVBoxLayout, QWidget)

import GrabCut
import matcher
import utils
from TeethCare import *


# folders paths
NO_CAMERA = r"\app\no-camera.svg"
TUTORIAL_PATH = r"\app\tutorial.pdf"
IMGS_PATH = r"\app\imgs"
CONFIG_PATH = r"\app\config.json"

# default config body
CONFIG = {
    'critical_loss': 25,
    'stack_size': 1,
    "stream_path": None,
    "stream_period": [48, True],
    "contours": {},
    "ROI": [0, 0, 0, 0],
    "match_treshold": 75
}

# thread for capturing
class Thread(QThread):
    changePixmap = pyqtSignal(QImage, np.ndarray,bool)
    path = '0'
    dims = [1920, 1080]
    roi = [0, 0, 0, 0]

    def run(self):
        # open stream
        cap = cv2.VideoCapture(self.path)
        reseted = False
        while True:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                reseted = True

            # frame is exists => resize and send to main thread
            if ret:
                rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                cv2.rectangle(
                    rgbImage, (self.roi[0], self.roi[1], self.roi[2], self.roi[3]), (0, 0, 255), 2)
                h, w, ch = rgbImage.shape
                bytesPerLine = ch * w
                convertToQtFormat = QImage(
                    rgbImage.data, w, h, bytesPerLine, QImage.Format_RGB888)
                p = convertToQtFormat.scaled(
                    self.dims[0], self.dims[1], Qt.KeepAspectRatio)
                self.changePixmap.emit(p, frame,reseted)
                reseted = False
            _ = cv2.waitKey(20)


# frame for displaying video stream
class VideoFrame(QFrame):

    def __init__(self, parent=None):
        super(VideoFrame, self).__init__()
        self.parent = parent
        self.frame = None
        self.label = QLabel()
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.setAlignment(Qt.AlignCenter)
        self.setLayout(layout)
        self.setFrameShape(QFrame.StyledPanel)
        self.initThread()
        self.label.setPixmap(QPixmap(os.getcwd()+NO_CAMERA))

    def initThread(self):
        self.th = Thread(self)
        self.th.changePixmap.connect(self.setImage)

    # start thread
    def startThread(self, path, dims, roi):
        self.th.path = path
        self.th.dims = dims
        self.th.roi = roi
        self.th.start()
        self.timeCount, self.frameCount = time.time(), 0

    # set frame recieved from thread
    @pyqtSlot(QImage, np.ndarray,bool)
    def setImage(self, image, frame,reseted):
        if reseted:
            self.frameCount = 0
            self.timeCount = 0

        self.label.setPixmap(QPixmap.fromImage(image))
        self.frame = frame
        self.frameCount += 1
        dif_time = time.time() - self.timeCount
        stream_period = self.parent.data['config']['stream_period']
        if (stream_period[1] and dif_time >= stream_period[0]) or (not stream_period[1] and self.frameCount >= stream_period[0]):
            self.timeCount, self.frameCount = time.time(), 0
            self.parent.startProcessFrameThread(utils.crop(frame, self.th.roi))

# widget for displaying result picture
class PictureFrame(QWidget):
    def __init__(self):
        super(PictureFrame, self).__init__()
        self.label = QLabel()
        layout = QHBoxLayout()
        layout.addWidget(self.label)
        layout.setAlignment(Qt.AlignCenter)
        self.setLayout(layout)

    def setIMG(self, img):
        self.label.setPixmap(QPixmap.fromImage(img))


class ResetDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent=parent)

        self.setWindowTitle("Reset")
        self.layout = QVBoxLayout()

        label = QLabel()
        label.setText(
            "All settings will be deleted! The system will be restarted!")

        QBtn = QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        self.buttonBox = QDialogButtonBox(QBtn)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        self.accepted.connect(self.acceptedEvent)

        self.layout.addWidget(label)
        self.layout.addWidget(self.buttonBox)
        self.setLayout(self.layout)

    # delete default folders an restart
    def acceptedEvent(self):
        shutil.rmtree(os.getcwd()+IMGS_PATH, ignore_errors=True)
        os.remove(
            os.getcwd()+CONFIG_PATH) if os.path.exists(os.getcwd()+CONFIG_PATH) else None
        os.startfile(__file__)
        sys.exit()


class ROIDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent=parent)

        w, h = parent.width(), parent.height()
        self.parent = parent
        roi = parent.data["config"]["ROI"]

        self.setWindowTitle("ROI")
        self.layout = QVBoxLayout()

        gStartBox = QGroupBox("Start")
        vbox = QVBoxLayout()
        gStartBox.setLayout(vbox)

        startX = QSpinBox()
        startX.setPrefix("X: ")
        startX.setRange(0, w)
        startX.setValue(roi[0])
        vbox.addWidget(startX)

        startY = QSpinBox()
        startY.setPrefix("Y: ")
        startY.setRange(0, h)
        startY.setValue(roi[1])
        vbox.addWidget(startY)

        gEndBox = QGroupBox("End")
        vbox = QVBoxLayout()
        gEndBox.setLayout(vbox)

        endX = QSpinBox()
        endX.setPrefix("X: ")
        endX.setRange(0, w)
        endX.setValue(roi[2])
        vbox.addWidget(endX)

        endY = QSpinBox()
        endY.setPrefix("Y: ")
        endY.setRange(0, h)
        endY.setValue(roi[3])
        vbox.addWidget(endY)

        QBtn = QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        self.buttonBox = QDialogButtonBox(QBtn)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        self.accepted.connect(lambda: self.acceptedEvent(
            [startX.value(), startY.value(), endX.value(), endY.value()]))

        self.layout.addWidget(gStartBox)
        self.layout.addWidget(gEndBox)
        self.layout.addWidget(self.buttonBox)
        self.setLayout(self.layout)

    # update roi
    def acceptedEvent(self, roi):
        if roi[0] < roi[2] and roi[1] < roi[3]:
            self.parent.data["config"]["ROI"] = roi
        else:
            QMessageBox.question(
                self, 'Повідомлення', 'The result is not saved. Coordinates set incorrectly.', QMessageBox.Ok)


class StreamDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent=parent)

        self.parent = parent
        isStreaming = parent.isStreaming
        stream_path = parent.data["config"]["stream_path"]
        stream_period = parent.data["config"]["stream_period"]
        match_treshold = parent.data["config"]["match_treshold"]
        critical_loss = parent.data["config"]["critical_loss"]
        stack_size = parent.data["config"]["stack_size"]

        self.setWindowTitle("Stream")

        self.layout = QVBoxLayout()

        self.line = QLineEdit()
        self.line.setPlaceholderText("Stream path")
        if stream_path:
            self.line.setText(stream_path)
        if isStreaming:
            self.line.setToolTip(
                'Stream from a new source is possible after a restart')
        self.layout.addWidget(self.line)

        self.matchBox = QSpinBox()
        self.matchBox.setToolTip(
            'The minimum % similarity of the bucket template with the image at which the area of the bucket teeth is calculated')
        self.matchBox.setSuffix(' % similarity')
        self.matchBox.setRange(10, 100)
        self.matchBox.setValue(match_treshold)
        self.layout.addWidget(self.matchBox)

        freqGBox = QGroupBox("Frequency of image capture")
        vbox = QVBoxLayout()
        freqGBox.setLayout(vbox)

        self.frameBox = QCheckBox('Sec')
        self.frameBox.setChecked(stream_period[1])
        self.frameBox.stateChanged.connect(self.changeMod)
        vbox.addWidget(self.frameBox)

        self.freqBox = QSpinBox()
        self.freqBox.setPrefix("Every ")
        self.freqBox.setSuffix(' s' if stream_period[1] else ' frames')
        self.freqBox.setRange(5, 3600)
        self.freqBox.setValue(stream_period[0])
        vbox.addWidget(self.freqBox)

        self.layout.addWidget(freqGBox)

        calculusGBox = QGroupBox("Calculations")
        vbox = QVBoxLayout()
        calculusGBox.setLayout(vbox)

        self.stackBox = QSpinBox()
        self.stackBox.setPrefix("Queue with ")
        self.stackBox.setSuffix(' calculations')
        self.stackBox.setRange(1, 10)
        self.stackBox.setValue(stack_size)
        self.stackBox.setToolTip(
            'The value of the tooth area will be calculated when filling the queue as the average of the values of the elements in the queue')
        vbox.addWidget(self.stackBox)

        self.criticalBox = QSpinBox()
        self.criticalBox.setSuffix(' % losses')
        self.criticalBox.setRange(1, 100)
        self.criticalBox.setValue(critical_loss)
        self.criticalBox.setToolTip(
            'A warning about damage to the teeth will be issued when the difference with the specified value is reached')
        vbox.addWidget(self.criticalBox)

        self.layout.addWidget(calculusGBox)

        QBtn = QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        self.buttonBox = QDialogButtonBox(QBtn)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)

        self.accepted.connect(lambda: self.acceptedEvent(
            self.line.text().strip() or None, self.frameBox.isChecked(), self.freqBox.value(), self.matchBox.value(), self.stackBox.value(), self.criticalBox.value()))

        self.layout.addWidget(self.buttonBox)
        self.setLayout(self.layout)

    def changeMod(self):
        self.freqBox.setSuffix(
            ' s' if self.frameBox.isChecked() else ' frames')

    # update settings
    def acceptedEvent(self, path, checked, freq, treshold, size, critical):
        self.parent.data["config"]["stream_path"] = path
        self.parent.data["config"]["stream_period"] = [freq, checked]
        self.parent.data["config"]["match_treshold"] = treshold
        self.parent.data["config"]["critical_loss"] = critical
        self.parent.data["config"]["stack_size"] = size


class FrameProcessThread(QThread):
    resultSignal = pyqtSignal(np.ndarray, list, str)
    frame = None
    imgs = {}
    contours = {}
    accuracy = 0.75

    def run(self):
        #image processing
        frame = cv2.GaussianBlur(self.frame, (5, 5), 1.41)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        #get paths of all buckets
        EX_imgs = [key for key in self.contours.keys() if key.endswith('_EX')]
        IN_imgs = [key for key in self.contours.keys() if key.endswith('_IN')]


        # template matching for buckets
        EX_res = [[matcher.multiscale_template_matching(frame, self.imgs[ex_suf], cv2.TM_CCOEFF_NORMED, np.linspace(
            0.8, 1.3, 7)[::-1], self.imgs[ex_suf+'_MASK'].copy()), ex_suf] for ex_suf in EX_imgs]
        IN_res = [[matcher.multiscale_template_matching(frame, self.imgs[in_suf], cv2.TM_CCOEFF_NORMED, np.linspace(
            0.8, 1.3, 7)[::-1], self.imgs[in_suf+'_MASK'].copy()), in_suf] for in_suf in IN_imgs]

        # filter buckets by corel
        EX_res_filtered = list(
            filter(lambda x: x[0][0][1] >= self.accuracy, EX_res))
        IN_res_filtered = list(
            filter(lambda x: x[0][0][1] >= self.accuracy, IN_res))

        # find max
        EX_max = max(EX_res_filtered or [
                     [[[0, 0], [0], [0], [0, 0]]]], key=lambda x: x[0][0][1])
        IN_max = max(IN_res_filtered or [
                     [[[0, 0], [0], [0], [0, 0]]]], key=lambda x: x[0][0][1])

        # get with the highest area
        bucket_res = max([EX_max, IN_max], key=lambda x: x[0][3][0]*x[0][3][1])

        # bucket detected
        if bucket_res[0][3][0] != 0:
            # resize to restore max corel
            resized_bucket = utils.resize(
                frame, width=int(frame.shape[1] * bucket_res[0][2]))

            maxLoc = bucket_res[0][0][3]
            th, tw = bucket_res[0][3]

            #get bucket pic
            resized_bucket = utils.crop(
                resized_bucket, (maxLoc[0], maxLoc[1], maxLoc[0]+tw, maxLoc[1]+th))

            # find teeth line
            line_res = [matcher.multiscale_template_matching(resized_bucket, self.imgs[bucket_res[1]+'_LINE'], cv2.TM_CCOEFF_NORMED, np.linspace(
                0.8, 1.3, 7)[::-1], self.imgs[bucket_res[1]+'_LINE_MASK'].copy()), bucket_res[1]+'_LINE_MASK']
            offset = 0 if bucket_res[1].endswith('_EX') else 0.1

            # teeth found
            if line_res[0][0][1] >= self.accuracy - offset:
                # print('line',line_res,self.accuracy)
                # resize to restore max corel
                resized_line = utils.resize(resized_bucket, width=int(
                    resized_bucket.shape[1] * line_res[0][2]))

                maxLoc = line_res[0][0][3]
                th, tw = line_res[0][3]
                resized_line = utils.crop(
                    resized_line, (maxLoc[0], maxLoc[1], maxLoc[0]+tw, maxLoc[1]+th))

                # find teeth positions
                res = []
                for coords in self.contours[bucket_res[1]]:
                    res.append(utils.detectTooth(
                        resized_line, coords[0], bucket_res[1].endswith('EX')))

                self.resultSignal.emit(resized_line, res, bucket_res[1])


class MainWindow(QMainWindow):
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        # defaults and settings
        self.frame, self.prev_check_res, self.time = None, None, None
        self.res = self.prev_check_res
        self.isStreaming = False
        self.data = {'config': initConfigFile(
            copy.deepcopy(CONFIG)), "imgs": {}}
        refSquare = {}
        for buck_name, sizes in self.data['config']['contours'].items():
            refSquare[buck_name] = [size[1] for size in sizes]
        self.teeth_system = TeethCare(
            refSquare, self.data['config']['stack_size'], self.data['config']['critical_loss'])
        self.loadImages()
        self.threads = {}
        self.createMenuBar()
        self.initUI()
        QTimer.singleShot(1000, lambda: self.startStream())

    # analyse bucket thread
    def startProcessFrameThread(self, frame):
        if len(self.data['imgs']) != 0:
            thr = FrameProcessThread()
            thr.frame, thr.imgs, thr.contours, thr.accuracy = frame, self.data[
                'imgs'], self.data['config']['contours'], self.data['config']['match_treshold']/100
            thr.resultSignal.connect(self.setFrameProcessResult)
            thr.finished.connect(lambda: self.threads.pop(
                id(thr), None))
            thr.finished.connect(thr.deleteLater)
            self.threads[id(thr)] = thr
            thr.start()

    # calculate teeth area and display results
    @pyqtSlot(np.ndarray, list, str)
    def setFrameProcessResult(self, img, sizes, bucket_name):
        # adding calculated area
        # check if critical
        self.teeth_system.queue_size, self.teeth_system.critical_loss = self.data[
            'config']['stack_size'], self.data['config']['critical_loss']
        refSquare = {}
        for buck_name, _sizes in self.data['config']['contours'].items():
            refSquare[buck_name] = [size[1] for size in _sizes]
            if buck_name == bucket_name:
                base_coords = [tuple(size[0]) for size in _sizes]
        self.teeth_system.square_ref = refSquare
        self.teeth_system.addReview(bucket_name, [size[1] for size in sizes], [
                                    size[0] for size in sizes],img)
        res, img = self.teeth_system.isCriticalDamaged(bucket_name)

        # if queue is full display image
        if res != None:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            for idx, tooth in enumerate(res):
                # x, y, w, h = tooth[2]
                bx, by, bw, bh = base_coords[idx]
                color = (200, 0, 0) if tooth[0] else (0, 200, 0)
                cv2.rectangle(img, (bx, by), (bx+bw, by+bh), color, 1)
                cv2.putText(img, str(
                    max(int(tooth[1]),0)), (bx, by+bh), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

                h, w, ch = img.shape
                bytesPerLine = ch * w
                convertToQtFormat = QImage(
                    img.data, w, h, bytesPerLine, QImage.Format_RGB888)
                p = convertToQtFormat.scaled(
                    self.width()//2, self.height()//2, Qt.KeepAspectRatio)

                self.tabwidget.widget(1).setIMG(p)

            damaged = [tooth[0] for tooth in res]
            self.res = (random(), 250, 0) if any(
                damaged) else (random(), 0, 250)






    def loadImages(self):
        for file_name in self.data['config']['contours'].keys():
            self.data['imgs'][file_name] = cv2.imread(
                f'{os.getcwd()+IMGS_PATH}\\{file_name}.jpg', cv2.IMREAD_GRAYSCALE)
            for suf in ['_LINE', '_MASK', '_LINE_MASK']:
                self.data['imgs'][file_name+suf] = cv2.imread(
                    f'{os.getcwd()+IMGS_PATH}\\{file_name+suf}.jpg', cv2.IMREAD_GRAYSCALE)

    def openDialog(self, obj, name):
        if obj.exec_():
            print(f"{name}. done!")
        else:
            print(f"{name}. failed!")

    # init video stream
    def startStream(self):
        path = self.data['config']['stream_path']
        roi = self.data['config']['ROI']
        if (not self.isStreaming and path):
            self.tabwidget.widget(0).startThread(
                path, [self.width()-130, self.height()-130], roi)
            self.startAction.setEnabled(False)
            self.isStreaming = True
            self.checkTimer = QTimer(self)
            self.checkTimer.setInterval(1000)
            self.checkTimer.timeout.connect(self.updateStatusBar)
            self.checkTimer.start()
        else:
            QMessageBox.question(self, 'Message',
                                 "Stream is not specified", QMessageBox.Ok)

    def createMenuBar(self):
        menuBar = self.menuBar()

        functionsMenu = QMenu("&Features", self)
        menuBar.addMenu(functionsMenu)

        streamAction = QAction("&Stream", self)
        streamAction.triggered.connect(
            lambda: self.openDialog(StreamDialog(parent=self), 'Stream'))
        self.startAction = QAction("&Start", self)
        self.startAction.triggered.connect(self.startStream)
        resetAction = QAction("&Reset", self)
        resetAction.triggered.connect(
            lambda: self.openDialog(ResetDialog(parent=self), 'Reset'))
        roiAction = QAction("&ROI", self)
        roiAction.triggered.connect(lambda: self.openDialog(ROIDialog(parent=self), 'ROI') or self.updateVideoFrameParams() if self.data['config']['ROI'] != [0, 0, 0, 0] else QMessageBox.question(self, 'Message',
                                                                                                                                                                                                        "The ROI has not been predefined", QMessageBox.Ok))
        reloadAction = QAction("&Restart", self)
        reloadAction.triggered.connect(
            lambda: dumpJSON(os.getcwd()+CONFIG_PATH, self.data["config"]) or os.startfile(__file__) or sys.exit())
        functionsMenu.addAction(streamAction)
        functionsMenu.addAction(roiAction)
        functionsMenu.addAction(self.startAction)
        functionsMenu.addAction(resetAction)
        functionsMenu.addAction(reloadAction)

        bucketMenu = QMenu("&Bucket", self)
        menuBar.addMenu(bucketMenu)

        addBucketMenu = QMenu("&Add", self)
        bucketMenu.addMenu(addBucketMenu)
        deleteBucketMenu = QMenu("&Delete", self)
        bucketMenu.addMenu(deleteBucketMenu)

        upAddBucketction = QAction("&Down", self)
        upAddBucketction.triggered.connect(
            lambda: self.bucketSizing('_EX') or self.updateVideoFrameParams() if self.isStreaming else QMessageBox.question(self, 'Повідомлення', 'Розпочніть трансляцію', QMessageBox.Ok))
        downAddBucketction = QAction("Up", self)
        downAddBucketction.triggered.connect(
            lambda: self.bucketSizing('_IN') or self.updateVideoFrameParams() if self.isStreaming else QMessageBox.question(self, 'Повідомлення', 'Розпочніть трансляцію', QMessageBox.Ok))
        addBucketMenu.addAction(upAddBucketction)
        addBucketMenu.addAction(downAddBucketction)

        upDeleteBucketAction = QAction("&Down", self)
        downDeleteBucketAction = QAction("&Up", self)
        deleteBucketMenu.addAction(upDeleteBucketAction)
        upDeleteBucketAction.triggered.connect(
            lambda: self.fileDialogExec('_EX'))
        deleteBucketMenu.addAction(downDeleteBucketAction)
        downDeleteBucketAction.triggered.connect(
            lambda: self.fileDialogExec('_IN'))

        helpAction = QAction("&Help", self)
        helpAction.triggered.connect(lambda: os.startfile(os.getcwd()+TUTORIAL_PATH) if os.path.exists(os.getcwd(
        )+TUTORIAL_PATH) else QMessageBox.question(self, 'Message', "I can't help you", QMessageBox.Ok))
        menuBar.addAction(helpAction)

    def updateStatusBar(self):
        if not self.time:
            self.time = time.time()
            return
        diff_time = time.time() - self.time
        if (self.prev_check_res) != (self.res):
            _, r, g = self.res
            self.prev_check_res = self.res
            self.time = time.time()
            self.statusBar.setStyleSheet(
                f"background-color: rgb({r}, {g}, 0);")
        self.statusBar.showMessage(
            f'The last check was carried out {int(diff_time)} seconds ago.')

    def initUI(self):
        self.setWindowTitle(
            "Excavator teeth condition control system ")

        self.tabwidget = QTabWidget()
        self.tabwidget.addTab(VideoFrame(parent=self), "Stream")
        self.tabwidget.addTab(PictureFrame(), "Calculations")
        self.setCentralWidget(self.tabwidget)

        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)

    def closeEvent(self, event):
        dumpJSON(os.getcwd()+CONFIG_PATH, self.data["config"])
        event.accept()

    def bucketSizing(self, suffix):
        #get roi_points 1
        #get roi 2
        #select line roi_points 3
        #clean roi - roi_gc 4
        #clean line_roi_gc 5
        #clean teeth gc 6
        # convert all gc 7
        #get squares 8
        #save all 9


        show_help = len(self.data['config']['contours']) == 0

        masks, teeth_contours = [], []

        frame = self.tabwidget.widget(0).frame

        h, w, _ = frame.shape
        frame = cv2.GaussianBlur(frame, (5, 5), 1.41)

        if type(frame) != None:
            # select bucket
            if show_help:
                QMessageBox.question(self, 'Message', "Capture a bucket", QMessageBox.Ok)
            roi_points = cv2.selectROI(r"Bucket region", frame)
            cv2.destroyAllWindows()
            if roi_points != (0, 0, 0, 0):
                #select teeth line

                if show_help:
                    QMessageBox.question(self, 'Message', "Capture a teeth line", QMessageBox.Ok)
                roi = utils.crop(frame, roi_points)
                line_roi_points = cv2.selectROI(r"Teeth region", roi)
                cv2.destroyAllWindows()
                if type(line_roi_points) != (0, 0, 0, 0):
                    if show_help:
                        QMessageBox.question(self, 'Message', "Mark a bucket", QMessageBox.Ok)

                    # grabcut bucket
                    line_roi = utils.crop(roi, line_roi_points)
                    roi_gc = GrabCut.App().run(
                        roi, (0, 0, roi.shape[1]-1, roi.shape[0]-1))
                    cv2.destroyAllWindows()
                    if type(roi_gc) != None:
                        if show_help:
                            QMessageBox.question(self, 'Message', "Mark each tooth", QMessageBox.Ok)
                        # grabcut teeth line
                        line_roi_gc = utils.crop(roi_gc, line_roi_points)
                        teeth_gc = GrabCut.App().run(
                            line_roi_gc, (0, 0, line_roi_gc.shape[1]-1, line_roi_gc.shape[0]-1))
                        cv2.destroyAllWindows()
                        if type(teeth_gc) != None:
                            for gc in [roi_gc, line_roi_gc, teeth_gc]:
                                gc_copy = gc.copy()
                                gc_copy[gc_copy > 0] = 255
                                masks.append(gc_copy)
                            # get teeth area
                            contours, _ = cv2.findContours(cv2.cvtColor(
                                masks[2], cv2.COLOR_BGR2GRAY), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                            teeth_contours = [[cv2.boundingRect(
                                cnt), cv2.contourArea(cnt)] for cnt in contours]
                            teeth_contours.sort(key=lambda x: x[0][0])

                            name = str(calendar.timegm(time.gmtime()))
                            stuff = {suffix: cv2.cvtColor(
                                roi, cv2.COLOR_BGR2GRAY),
                                suffix+'_LINE': cv2.cvtColor(
                                line_roi, cv2.COLOR_BGR2GRAY),
                                suffix+'_MASK': cv2.cvtColor(
                                masks[0], cv2.COLOR_BGR2GRAY), suffix+'_LINE_MASK': cv2.cvtColor(
                                masks[1], cv2.COLOR_BGR2GRAY)}
                            # save imgs and masks
                            for suf, pic in stuff.items():
                                cv2.imwrite(
                                    f"{os.getcwd()+IMGS_PATH}\\{name+suf}.jpg", pic)
                                self.data['imgs'][name+suf] = pic
                            self.data['config']['contours'][name +
                                                            suffix] = teeth_contours
                            # update roi
                            old_roi = self.data['config']['ROI']
                            if old_roi == [0, 0, 0, 0]:
                                self.data['config']['ROI'] = (
                                    max(0, roi_points[0] - 150), 0, min(roi_points[2]+150, w), min(h, roi_points[3]+150))
                            else:
                                self.data['config']['ROI'] = (max(min(old_roi[0], roi_points[0]-150), 0), 0, min(
                                    max(roi_points[2]+150, old_roi[2]), w), min(max(roi_points[3], roi_points[3]+150), h))
        else:
            QMessageBox.question(
                self, 'Message', 'The result has not been saved. It is necessary to pass all stages.', QMessageBox.Ok)

    def fileDialogExec(self, suffix):
        file_dialog = QFileDialog(self)
        file_dialog.setFileMode(QFileDialog.ExistingFiles)
        file_dialog.setNameFilters([f"{suffix} (*{suffix}.jpg)"])
        file_dialog.setDirectory(os.getcwd() + IMGS_PATH)

        # delete all derivitievs
        if file_dialog.exec_():
            for path_name in file_dialog.selectedFiles():
                os.remove(path_name) if os.path.exists(path_name) else None
                directory, img_name_type = path_name.rsplit('/', 1)
                img_name, format = img_name_type.rsplit('.', 1)
                self.data['config']['contours'].pop(img_name, None)
                for suf in ['_MASK', '_LINE', '_LINE_MASK', ]:
                    name = img_name+suf+'.'+format
                    new_path_name = directory + '\\'+name
                    os.remove(new_path_name) if os.path.exists(
                        new_path_name) else None
                    self.data['imgs'].pop(img_name+suf, None)
        else:
            pass

    def updateVideoFrameParams(self):
        self.tabwidget.widget(0).th.roi = self.data['config']['ROI']


def dumpJSON(path, json_dict):
    with open(path, 'w') as f:
        f.write(json.dumps(json_dict))


def initConfigFile(config):
    if not os.path.exists(os.getcwd()+CONFIG_PATH):
        with open(os.getcwd()+CONFIG_PATH, 'w') as f:
            f.write(json.dumps(config))
    else:
        with open(os.getcwd()+CONFIG_PATH) as f:
            config = json.load(f)
    return config


def initIMGFolder():
    if not os.path.exists(os.getcwd()+IMGS_PATH):
        os.makedirs(os.getcwd()+IMGS_PATH)


if __name__ == '__main__':
    initIMGFolder()
    app = QApplication(sys.argv)
    window = MainWindow()
    window.showMaximized()
    app.exec_()
