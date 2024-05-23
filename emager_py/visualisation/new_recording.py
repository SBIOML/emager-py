# importing libraries
from PyQt5.QtWidgets import *
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QMainWindow, QFileDialog
from PyQt5.QtGui import *
from PyQt5.QtCore import QThread, QObject, pyqtSignal
import sys
import numpy as np
import time
import SensorLib
from PyQt5.QtWidgets import QApplication
import csv
from playsound import playsound
from threading import Thread
import os
import keyboard
import math
from collections import deque

PORT_NAME = "COM4"

PATH = "C:/Users/Anne-Sophie/Desktop/STAGE/Data_record/Nouveau"
AUDIO = "audio/mixkit-achievement-bell-601.wav"
#AUDIO_2 = "audio/mixkit-achievement-bell-600.wav"


#SELECT DATASET
DB = 3
DB = 3

#parameters
NB_GESTES = {1:6, 2:5, 3:11} #NB of gestures i each subdataset
GESTES = NB_GESTES[DB]

#Useful for the laelling
START = 0
if DB == 2:
    START += NB_GESTES[1]
if DB == 3:
    START += NB_GESTES[2]



#creates the sequence neut -> transition -> hold -> etc.
states = [0, 1, 2, 3] * GESTES + [0]
#states.append(0)
REC_TIMES = {0: 2000, 1: 1000, 2: 2000, 3: 1000}
LISTE_GESTES = [_ for _ in range(GESTES)]


def play_sound():
    playsound(AUDIO)

class MyWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.STOP = False   #useful for the stop button, becomes true when the stop button is pressed

        #initial parameters
        self.sessionNb = "001"
        self.repNb = "000"
        self.userID = "000"
         # Users can record multiple sessions
        self.gestureNb = None  # gesture 0 to 5
        self.arm = "right"  # choose the arm used for data recording
        self.filepath = PATH  # Path to save the data
        self.initUI()

        # Creates Sensor Object

        self.sensor = SensorLib.HDSensor(PORT_NAME, 1500000)
        self.sensor.open()

        # Creates variable to store gesture number
        self.current_gesture = 0

        self.gestures = []

    # method for creating widgets
    def initUI(self):
        self.setGeometry(200, 20, 1200, 1000)
        # User ID
        self.label_userID = QtWidgets.QLabel(self)
        self.label_userID.setText("User ID")
        self.label_userID.move(50, 10)
        self.dropDown_userID = QtWidgets.QComboBox(self)
        self.dropDown_userID.addItems([""] + [f"{_:03}" for _ in range(20)])
        self.dropDown_userID.move(25, 35)
        self.dropDown_userID.currentTextChanged.connect(self.userID_changed)

        # Session Number
        self.label_repNb = QtWidgets.QLabel(self)
        self.label_repNb.setText("Repetition")
        self.label_repNb.move(40, 150)
        self.dropDown_repNb = QtWidgets.QComboBox(self)
        self.dropDown_repNb.addItems(
            [""] + [f"{_:03}" for _ in range(0, 10)])
        self.dropDown_repNb.move(25, 175)
        self.dropDown_repNb.currentTextChanged.connect(self.repNb_changed)
        #display the progression of reps made
        self.rep_label = QtWidgets.QLabel(self)
        self.rep_label.setText(f"{int(self.repNb)}/10")
        self.rep_label.move(40, 200)
#
        # Select Path Button
        self.button_path = QtWidgets.QPushButton(self)
        self.button_path.setText("Select Path")
        self.button_path.clicked.connect(self.path)
        self.button_path.move(25, 320)
        #display the path selected
        self.path_sel_lab = QLabel(self)
        self.path_sel_lab.setText("")
        self.path_sel_lab.move(25, 370)
        self.path_sel_lab.resize(250, 20)

        # Arm used
        self.label_arm = QtWidgets.QLabel(self)
        self.label_arm.setText("Arm Used")
        self.label_arm.move(50, 80)
        self.dropDown_arm = QtWidgets.QComboBox(self)
        self.dropDown_arm.addItems(["", "right", "left"])
        self.dropDown_arm.move(25, 105)
        self.dropDown_arm.currentTextChanged.connect(self.arm_changed)

        # Session num
        self.label_session = QtWidgets.QLabel(self)
        self.label_session.setText("Session num")
        self.label_session.move(50, 240)
        self.dropDown_session = QtWidgets.QComboBox(self)
        self.dropDown_session.addItems([f"{_:03}" for _ in range(1, 10)])
        self.dropDown_session.move(25, 265)
        self.dropDown_session.currentTextChanged.connect(self.session_changed)

        # creating push button
        self.btn = QPushButton('Start (space)', self)
        self.btn.move(20, 700)
        self.btn.clicked.connect(self.doAction)
        self.btn.setShortcut(' ')

        #Status label
        self.status_label = QtWidgets.QLabel(self)
        self.status_label.setText("")
        self.status_label.move(20, 750)
        self.status_label.resize(200, 20)

        #and stop button (you can )
        self.stop_btn = QPushButton('Stop', self)
        self.stop_btn.setText("Stop (p)")
        self.stop_btn.move(20, 750)
        self.stop_btn.clicked.connect(self.stop)
        self.stop_btn.setShortcut("p")


        # setting window

        self.setWindowTitle("EMaGer 2.0 Recording")

        self.p_bars = []
        self.neutral_lab = []
        self.label_gest = []

        #showing all the images
        self.display_img()

        # showing all the widgets
        self.show()

    #change the session numb manually
    def session_changed(self, s):
        self.sessionNb = f"{int(s):03}"

    # selects the User ID, Session Nb, Arm, Nb of rep and length of rep
    def userID_changed(self, s):
        if s == "":
            self.userID = None
        else:
            self.userID = s
    def repNb_changed(self, s):
        if s == "":
            self.repNb = None
        else:
            self.repNb = s
        self.rep_label.setText(f"{int(self.repNb)}/10")
    def arm_changed(self, s):
        if s == "":
            self.arm = None
        else:
            self.arm = s


    def path(self):
        self.filepath = str(QFileDialog.getExistingDirectory(self, "Select Directory"))
        self.path_sel_lab.setText(self.filepath)

    #updates the right progress bar
    def update_pbar(self, p_bar, low, high, queue, final_time):
        time_now = 0
        while time_now < final_time:
            p_bar.setValue(low + int((high - low) * time_now / final_time))
            time_now = queue.popleft()

    def display_img(self):
        # Creating the required number of gestures
        self.p_bars.clear()
        self.neutral_lab.clear()
        self.label_gest.clear()
        esp = 850 / (GESTES)
        #if 10 or more images
        if GESTES>=10:
            esp = esp*2


        for num in range(GESTES):
            y_pos = int(60 + num * esp) #position of the image
            self.p_bars.append(QProgressBar(self))  # create progress bars for each image
            self.p_bars[-1].setGeometry(430, y_pos, 500, 25)

            self.label_gest.append(QtWidgets.QLabel(self))
            self.label_gest[-1].move(282, int(y_pos - esp / 4))

            #create 2 rows if there are 10 or more images
            if GESTES >= 10:
                self.p_bars[-1].setGeometry(430, y_pos, 300, 25)
                if num > GESTES/2:
                    y_pos = int(60 + (num-math.ceil(GESTES/2)) * esp)
                    self.p_bars[-1].setGeometry(850, y_pos, 300, 25)
                    self.label_gest[-1].move(700, int(y_pos - esp / 4))

            #create the image
            pix = QPixmap(f'photos_gestes/db{DB}/OBGesture{DB}-{num}')
            # miroir the images if the user uses their left hand
            if self.arm == "left":
                pix = QPixmap(f'photos_gestes/db{DB}/OBGesture{DB}-{num}').transformed(QTransform().scale(-1, 1))
            pix = pix.scaledToHeight(120)
            self.label_gest[-1].setPixmap(pix)
            self.label_gest[-1].resize(pix.width(), pix.height())

        #create the neutral label and progress bar (bottom of the screen)
        self.neutral_lab = (QtWidgets.QLabel(self))
        self.neutral_lab.setText("Neutral")
        self.neutral_lab.resize(150, 25)
        self.neutral_lab.setFont(QFont('Arial', 16))
        self.neutral_lab.setStyleSheet("QLabel { font-weight: bold; color : black; }")
        self.neutral_lab.move(250, 920)
        self.neutral_pbar = QProgressBar(self)
        self.neutral_pbar.setGeometry(400, 920, 500, 25)

    # when button is pressed this method is being called
    def doAction(self):

        self.STOP = False #reinitialize any stop button that might have been pressed

        #if invalid settings
        if None in [self.userID, self.repNb, self.arm, self.filepath]:
            self.status_label.setText("Wrong settings X(")
            self.status_label.setStyleSheet('color : red')
        else:
            self.status_label.setText("Lets a gooo X) ")
            self.status_label.setStyleSheet('color : black')

            #set all progress bars to value = 0
            self.neutral_pbar.setValue(0)
            for gest in range(GESTES):
                self.p_bars[gest].setValue(0)

            #reset useful variables
            idx = 0     #current index of the data array
            full_data = np.zeros((GESTES*6000+2000, 64))    #data
            gesture_performed = np.zeros((GESTES*6000+2000,2))  #labels of the data
            pbar_values = {0: (33, 100), 1:(0, 33), 2:(33, 100), 3:{0, 33}} #max and min values of the pbar

            self.sensor.clear_buffer()
            print("Buffer Cleared")

            gest_now = 0
            last_gest = 0

            #loop over all states
            for num, state in enumerate(states):
                if not self.STOP:
                    #select the right progess bar and gesture depending on current state
                    if state == 0:
                        pbar_now = self.neutral_pbar
                        #gest_now += 1
                    if state == 1:  #change gesture

                        last_gest += 1
                        gest_now = last_gest
                        pbar_now = self.p_bars[gest_now - 1]

                        #the following comment changes the color of the progress bar depending on the state of recording
                        #pbar_now.setStyleSheet("QProgressBar::chunk "
                                               # "{"
                                               # "background-color: yellow;"
                                               # "}")
                    # else:
                    #     pbar_now.setStyleSheet("QProgressBar::chunk "
                    #                            "{"
                    #                            "background-color: green;"
                    #                            "}")
                    # if state == 2:  #keeps the same gest num
                    #     pbar_now = self.p_bars[gest_now-1]
                    if state == 3:  #switches to neutral
                        pbar_now = self.neutral_pbar

                    #how long to record
                    final_time = REC_TIMES[state]

                    #update the progress bar to the right values
                    low, high = pbar_values[state]

                    #low and high are the progressbar values for the current states
                    if num == 0:
                        low, high = 0, 100
                    # elif num == 1:
                    #     low, high = 75, 100
                    elif num == len(states) - 2:
                        low, high = 0, 33
                    elif num == len(states)-1:
                        low, high = 33, 100


                    self.record_data(final_time, full_data, idx, low, high, pbar_now)
                    QApplication.processEvents()

                    if not self.STOP :
                        if state == 0:
                            # self.pbar_now.setStyleSheet("QProgressBar::chunk "
                            #                   "{"
                            #                   "background-color: yellow;"
                            #                   "}")
                            gesture_performed[idx:idx + final_time, :] = np.append((np.ones((final_time, 1)) * 0), np.ones((final_time, 1)) * state, axis=1)  # hold
                        else:
                            gesture_performed[idx:idx+final_time, :] = np.append((np.ones((final_time, 1)) * (gest_now+START)), np.ones((final_time, 1)) * state, axis=1)  # hold
                        idx += final_time

                        if state == 0 or state == 2:
                            thread = Thread(target=play_sound)
                            thread.start()

            if not self.STOP:
                # Save as CSV and change the repetition number
                data_matrix = np.append(full_data[0:idx, :],gesture_performed[0:idx, :], axis=1)
                self.save_data(data_matrix)
                self.change_rep()

#change the session number after one session is done.
    def change_rep(self):
        self.repNb = f"{int(self.repNb) + 1:03}"
        self.rep_label.setText(f"{int(self.repNb)}/10")

#saving the data with the right filename and directory
    def save_data(self, data_matrix):
        csv_filename = f"{self.userID}-{self.sessionNb}-{self.repNb}-{self.arm}.csv"

        #will check if the directory exists, creates it if not
        csv_dir = self.filepath + f"/{self.userID:03}"
        try:
            os.mkdir(csv_dir)
        except FileExistsError:
            pass

        csv_dir += f"/session_{self.sessionNb}"
        try:
            os.mkdir(csv_dir)
        except FileExistsError:
            pass

        csv_dir += f"/DB{DB}"
        try:
            os.mkdir(csv_dir)
        except FileExistsError:
            pass

        #final path and saving it
        csv_path = csv_dir + "/" + csv_filename

        with open(csv_path, 'w', newline='') as f:
            # create the csv writer
            writer = csv.writer(f)
            # write rows to the csv file
            writer.writerows(data_matrix)
            # close the file
            f.close()

    def record_data(self, final_time, full_data, idx, low, high, pbar_now):
        time_now = 0
        while time_now < final_time:
            if not keyboard.is_pressed('p'): #pausing
                timer = time.time()
                data = np.array(self.sensor.read_full_buffer())

                time_now += data.shape[0]
                if time_now > final_time:  # if it has read too much data
                     extra = time_now - final_time
                     full_data[idx + time + time_now - extra] = data[:-extra]
                else:
                    full_data[idx + time_now - data.shape[0]: idx + time_now] = data
                # #if time_now % 10 == 0:
                #print("Record time: ", time.time()-timer)
                #time_now = time.time()
                #timer = time.time()
                #print(timer-time_now)

                pbar_now.setValue(low + int((high - low) * time_now / final_time))
            else:
                self.STOP = True
                break

    def stop(self):
        self.STOP = True

# main method
if __name__ == '__main__':
    # create pyqt5 app
    App = QApplication(sys.argv)

    # create the instance of our Window
    window = MyWindow()

    # start the app
    sys.exit(App.exec())
