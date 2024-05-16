from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox
from PyQt5.QtGui import QPixmap, QFont
import sys
import csv
import serial
import numpy as np
import time


# Deals with the sensor
def reorder(data, mask, match_result):
    '''
    Looks for mask/template matching in data array and reorders
    :param data: (numpy array) - 1D data input
    :param mask: (numpy array) - 1D mask to be matched
    :param match_result: (int) - Expected result of mask-data convolution matching
    :return: (numpy array) - Reordered data array
    '''
    data_lsb = data & np.ones(128, dtype=np.int8)
    mask_match = np.convolve(mask, np.append(data_lsb, data_lsb), 'valid')
    try:
        offset = np.where(mask_match == match_result)[0][0]-3
    except IndexError:
        return None
    return np.roll(data, -offset)
class HDSensor(object):
    '''
    Sensor object for data logging from HD EMG sensor
    '''
    def __init__(self, serialpath, BR):
        '''
        Initialize HDSensor object, open serial communication to specified port using PySerial API
        :param serialpath: (str) - Path to serial port
        :param BR: (int) - Com port baudrate
        '''
        self.ser = serial.Serial(serialpath, BR, timeout=1)
        self.ser.close()

        self.bytes_to_read = 128
                        ### ^ Number of bytes in message (i.e. channel bytes + header/tail bytes)
        self.mask = np.array([0, 2] + [0, 1]*63)
                        ### ^ Template mask for template matching on input data
        self.channelMap = [10,22,12,24,13,26,7,28,1,30,59,32,53,34,48,36] + \
                          [62,16,14,21,11,27,5,33,63,39,57,45,51,44,50,40] + \
                          [8,18,15,19,9,25,3,31,61,37,55,43,49,46,52,38] + \
                          [6,20,4,17,2,23,0,29,60,35,58,41,56,47,54,42]


            # [i for i in range(27, 32)] + [0, 1, 2] + [i for i in range(23, 27)] + \
            #                [i for i in range(3, 7)] + [22, 21, 20, 19] + [10, 9, 8, 7] + [18, 17, 16, 15, 14, 13, 12, 11]
        #                 ### ^ Channel map to hardware sensor obtained from lab tests, needed to reorder channels

    def clear_buffer(self):
        '''
        Clear the serial port input buffer.
        :return: None
        '''
        self.ser.reset_input_buffer()
        return

    def close(self):
        self.ser.close()
        return

    def open(self):
        self.ser.open()
        return

    def read(self, readtime, feedback=False, savetxt=False, savepath=None):
        '''
        Read the incoming data in com port for a given time.
        :param readtime: (int) - reading time period (seconds)
        :param feedback: (bool) - print notice upon receiving corrupted data
        :param savetxt: (bool) - save read data to csv
        :param savepath: (str) - path for saved data
        :return: (list of lists) - list of channels' listed data points (e.g. 64xN for 64 channels of N data points)
        '''
        data = [[] for i in range(64)]

                                ### ^ 64 = n_channels
        self.open()
        self.clear_buffer()

        start_time = time.time()
        while(time.time() - start_time) < readtime:
            data_packet = reorder(list(self.ser.read(self.bytes_to_read)), self.mask, 63)
            if data_packet is not None:
                samples = [int.from_bytes(bytes([data_packet[i*2], data_packet[i*2+1]]), 'big', signed=True) for i in range(64)]
                                            ### ^ Iterating over byte pairs in line, 64 => n_channels, 2 bytes per ch.
                # print(data)
                for i, d in enumerate(data):
                    d += [samples[i]]
                        ### ^ Separating recorded data to respective channels
            elif feedback:
                print('Corrupted data. Dropped packet.')
            else:
                pass
        self.close()

        data_remap = []
        for i in self.channelMap:
             data_remap += [data[i]]
        #                 ### ^ Remapping data channels

        if savetxt:
            np.savetxt(savepath, data_remap, delimiter=',', fmt='%s')

        return data_remap #data_remap

    def sample(self):
        '''
        Sample 1 message from com port (1 sample from each channel), retry until valid reception in case of
        corrupted data.
        :return: (list) - containing the 64 samples (1 for each channel)
        '''
        #self.open()
        self.clear_buffer()
        while(True):
            data_packet = reorder(list(self.ser.read(128)), self.mask, 63)
            if data_packet is not None:
                sample = [int.from_bytes(bytes([data_packet[i * 2], data_packet[i * 2 + 1]]), 'big', signed=True) for i in
                            range(64)]      ### ^ Iterating over byte pairs in line, 32 => n_channels, 2 bytes per ch.
                # sample = [sample[i] for i in self.channelMap]
                #                             ### ^ Remapping data channels
                #self.close()
                return sample


# deals with the GUI
class MyWindow(QMainWindow):
    def __init__(self):
        super(MyWindow, self).__init__()
        self.initUI()
        self.userID = None  # Users are identified through a unique ID
        self.sessionNb = None  # Users can record multiple sessions
        self.gestureNb = None  # gesture 0 to 5
        self.arm = None  # choose the arm used for data recording
        self.recordTime = None  # length of a recording in seconds
        self.repetitionNb = 1  # number of repetition for each gesture
        self.filepath = None  # Path to save the data
        self.trialNb = np.array([0, 0, 0, 0, 0, 0], dtype=int)  # repetition number for one gesture!
        self.trialNb_str = np.array(["000", "000", "000", "000", "000", "000"])

    def initUI(self):  # initialize all of the GUI
        self.setGeometry(200, 100, 600, 700)
        self.setWindowTitle("64-Channel EMG Data Collection")

        # User ID
        self.label_userID = QtWidgets.QLabel(self)
        self.label_userID.setText("User ID")
        self.label_userID.move(50,10)
        self.dropDown_userID = QtWidgets.QComboBox(self)
        self.dropDown_userID.addItems([" ","000", "001", "002", "003", "004", "005", "006", "007", "008", "009", "010","011"])
        self.dropDown_userID.move(25, 35)
        self.dropDown_userID.currentTextChanged.connect(self.userID_changed)

        # Session Number
        self.label_sessionNb = QtWidgets.QLabel(self)
        self.label_sessionNb.setText("Session Number")
        self.label_sessionNb.move(190, 10)
        self.dropDown_sessionNb = QtWidgets.QComboBox(self)
        self.dropDown_sessionNb.addItems([" ", "000", "001", "002", "003","004","005","006","007","008","009","010"])
        self.dropDown_sessionNb.move(185, 35)
        self.dropDown_sessionNb.currentTextChanged.connect(self.sessionNb_changed)

        # Arm used
        self.label_arm = QtWidgets.QLabel(self)
        self.label_arm.setText("Arm Used")
        self.label_arm.move(50, 80)
        self.dropDown_arm = QtWidgets.QComboBox(self)
        self.dropDown_arm.addItems([" ", "right", "left"])
        self.dropDown_arm.move(25, 105)
        self.dropDown_arm.currentTextChanged.connect(self.arm_changed)

        # Number of repetition
        self.label_repetition = QtWidgets.QLabel(self)
        self.label_repetition.setText("Nb. of repetition")
        self.label_repetition.move(30, 150)
        self.dropDown_repetition = QtWidgets.QComboBox(self)
        self.dropDown_repetition.addItems(["0", "1", "3", "5", "10"])
        self.dropDown_repetition.move(25, 175)
        self.dropDown_repetition.currentTextChanged.connect(self.repetitionNb_changed)

        # Recording time (sec)
        self.label_recordTime = QtWidgets.QLabel(self)
        self.label_recordTime.setText("Rec. Time (sec)")
        self.label_recordTime.move(190, 150)
        self.dropDown_recordTime = QtWidgets.QComboBox(self)
        self.dropDown_recordTime.addItems(["1", "3", "5"])
        self.dropDown_recordTime.move(185, 175)
        self.dropDown_recordTime.currentTextChanged.connect(self.recordTime_changed)

        ## Gesture ##
        self.label_gesture = QtWidgets.QLabel(self)
        self.label_gesture.setText("Gesture")
        self.label_gesture.move(372, 10)
        # 000 closed fist
        self.label_gesture0 = QtWidgets.QLabel(self)  # creates label for image
        self.label_gesture0.move(350, 40)
        self.pix_gesture0 = QPixmap('OBgesture0.jpg')  # loads the image
        self.pix_gesture0 = self.pix_gesture0.scaledToHeight(90)
        self.label_gesture0.setPixmap(self.pix_gesture0)  # puts  the image at the location of the label
        self.label_gesture0.resize(self.pix_gesture0.width(), self.pix_gesture0.height())
        self.button0 = QtWidgets.QRadioButton(self)  # creates the checkbox
        self.button0.setGeometry(QtCore.QRect(450, 75, 95, 20))
        self.button0.toggled.connect(self.action0)  # links the checkbox to an action
        self.label_count0 = QtWidgets.QLabel(self)
        self.label_count0.setText("0/0")
        self.label_count0.move(500, 70)

        # 001 thumbs up
        self.label_gesture1 = QtWidgets.QLabel(self)
        self.label_gesture1.move(350, 150)
        self.pix_gesture1 = QPixmap('OBgesture1.jpg')
        self.pix_gesture1 = self.pix_gesture1.scaledToHeight(90)
        self.label_gesture1.setPixmap(self.pix_gesture1)
        self.label_gesture1.resize(self.pix_gesture1.width(), self.pix_gesture1.height())
        self.button1 = QtWidgets.QRadioButton(self)
        self.button1.setGeometry(QtCore.QRect(450, 185, 95, 20))
        self.button1.toggled.connect(self.action1)  # links the checkbox to an action
        self.label_count1 = QtWidgets.QLabel(self)
        self.label_count1.setText("0/0")
        self.label_count1.move(500, 180)

        # 002 tripod grip
        self.label_gesture2 = QtWidgets.QLabel(self)
        self.label_gesture2.move(350, 260)
        self.pix_gesture2 = QPixmap('OBgesture2.jpg')
        self.pix_gesture2 = self.pix_gesture2.scaledToHeight(90)
        self.label_gesture2.setPixmap(self.pix_gesture2)
        self.label_gesture2.resize(self.pix_gesture2.width(), self.pix_gesture2.height())
        self.button2 = QtWidgets.QRadioButton(self)
        self.button2.setGeometry(QtCore.QRect(450, 295, 95, 20))
        self.button2.toggled.connect(self.action2)  # links the checkbox to an action
        self.label_count2 = QtWidgets.QLabel(self)
        self.label_count2.setText("0/0")
        self.label_count2.move(500, 290)

        # 003 neutral hand
        self.label_gesture3 = QtWidgets.QLabel(self)
        self.label_gesture3.move(350, 370)
        self.pix_gesture3 = QPixmap('OBgesture3.jpg')
        self.pix_gesture3 = self.pix_gesture3.scaledToHeight(90)
        self.label_gesture3.setPixmap(self.pix_gesture3)
        self.label_gesture3.resize(self.pix_gesture3.width(), self.pix_gesture3.height())
        self.button3 = QtWidgets.QRadioButton(self)
        self.button3.setGeometry(QtCore.QRect(450, 405, 95, 20))
        self.button3.toggled.connect(self.action3)  # links the checkbox to an action
        self.label_count3 = QtWidgets.QLabel(self)
        self.label_count3.setText("0/0")
        self.label_count3.move(500, 400)

        # 004 fine pinch
        self.label_gesture4 = QtWidgets.QLabel(self)
        self.label_gesture4.move(350, 480)
        self.pix_gesture4 = QPixmap('OBgesture4.jpg')
        self.pix_gesture4 = self.pix_gesture4.scaledToHeight(90)
        self.label_gesture4.setPixmap(self.pix_gesture4)
        self.label_gesture4.resize(self.pix_gesture4.width(), self.pix_gesture4.height())
        self.button4 = QtWidgets.QRadioButton(self)
        self.button4.setGeometry(QtCore.QRect(450, 515, 95, 20))
        self.button4.toggled.connect(self.action4)  # links the checkbox to an action
        self.label_count4 = QtWidgets.QLabel(self)
        self.label_count4.setText("0/0")
        self.label_count4.move(500, 510)

        # 005 one finger
        self.label_gesture5 = QtWidgets.QLabel(self)
        self.label_gesture5.move(350, 590)
        self.pix_gesture5 = QPixmap('OBgesture5.jpg')
        self.pix_gesture5 = self.pix_gesture5.scaledToHeight(90)
        self.label_gesture5.setPixmap(self.pix_gesture5)
        self.label_gesture5.resize(self.pix_gesture5.width(), self.pix_gesture5.height())
        self.button5 = QtWidgets.QRadioButton(self)
        self.button5.setGeometry(QtCore.QRect(450, 625, 95, 20))
        self.button5.toggled.connect(self.action5)  # links the checkbox to an action
        self.label_count5 = QtWidgets.QLabel(self)
        self.label_count5.setText("0/0")
        self.label_count5.move(500, 620)

        # Create Run button
        self.button_record = QtWidgets.QPushButton(self)
        self.button_record.setText("Record")
        self.button_record.resize(200,80)
        self.button_record.setFont(QFont('Times', 15))
        self.button_record.clicked.connect(self.record)
        self.button_record.move(52, 260)

        # Select Path Button
        self.button_path = QtWidgets.QPushButton(self)
        self.button_path.setText("Select Path")
        self.button_path.clicked.connect(self.path)
        self.button_path.move(185, 105)

        # Create Undo Button
        self.button_undo = QtWidgets.QPushButton(self)
        self.button_undo.setText("Undo")
        self.button_undo.resize(50, 20)
        self.button_undo.setFont(QFont('Times', 8))
        self.button_undo.clicked.connect(self.undo)
        self.button_undo.move(275, 260)

        # Create a status label
        self.label_status = QtWidgets.QLabel(self)
        self.label_status.setText("Check settings")
        self.label_status.move(52, 350)

    # selects the User ID, Session Nb, Arm, Nb of rep and length of rep
    def userID_changed(self, s):
        self.userID = s
    def sessionNb_changed(self, s):
        self.sessionNb = s
    def arm_changed(self, s):
        self.arm = s
    def repetitionNb_changed(self, s):
        self.repetitionNb = int(s)
        self.label_count0.setText("0/" + str(self.repetitionNb))
        self.label_count1.setText("0/" + str(self.repetitionNb))
        self.label_count2.setText("0/" + str(self.repetitionNb))
        self.label_count3.setText("0/" + str(self.repetitionNb))
        self.label_count4.setText("0/" + str(self.repetitionNb))
        self.label_count5.setText("0/" + str(self.repetitionNb))
    def recordTime_changed(self, s):
        self.recordTime = int(s)
    def updateCountScreen(self):
        self.label_count0.setText(str(self.trialNb[0]) + "/" + str(self.repetitionNb))
        self.label_count1.setText(str(self.trialNb[1]) + "/" + str(self.repetitionNb))
        self.label_count2.setText(str(self.trialNb[2]) + "/" + str(self.repetitionNb))
        self.label_count3.setText(str(self.trialNb[3]) + "/" + str(self.repetitionNb))
        self.label_count4.setText(str(self.trialNb[4]) + "/" + str(self.repetitionNb))
        self.label_count5.setText(str(self.trialNb[5]) + "/" + str(self.repetitionNb))

    # selects the gesture
    def action0(self, selected):
        if selected:
            self.gestureNb = "000"
    def action1(self, selected):
        if selected:
            self.gestureNb = "001"
    def action2(self, selected):
        if selected:
            self.gestureNb = "002"
    def action3(self, selected):
        if selected:
            self.gestureNb = "003"
    def action4(self, selected):
        if selected:
            self.gestureNb = "004"
    def action5(self, selected):
        if selected:
            self.gestureNb = "005"
    def switch_button(self, current):
        if current == '000':
            self.button0.setChecked(False)
            self.button1.setChecked(True)
        elif current == '001':
            self.button1.setChecked(False)
            self.button2.setChecked(True)
        elif current == '002':
            self.button2.setChecked(False)
            self.button3.setChecked(True)
        elif current == '003':
            self.button3.setChecked(False)
            self.button4.setChecked(True)
        elif current == '004':
            self.button4.setChecked(False)
            self.button5.setChecked(True)
        elif current == '005':
            self.button5.setChecked(False)
            self.button0.setChecked(True)

    # action when the record button is pressed
    def record(self):
        self.label_status.setText("Recording in progress...")
        self.label_status.setStyleSheet("QLabel { font-weight: bold; color : red; }")
        self.label_status.adjustSize()
        self.label_status.repaint()

        print("Recording!")
        time.sleep(0.5)
        sensor = HDSensor('COM13', 1500000)

        # Gets Pts and average them
        data = sensor.read(self.recordTime)  # alternative = sensor.sample()
        data = np.transpose(data)

        # Change status to DONE!
        self.label_status.setText("Done!")
        self.label_status.setStyleSheet("QLabel { font-weight: bold; color : black; }")

        # put data into CSV file
        # aaa-bbb-ccc-ddd.csv --> a = userID, b = session#, c = gesture#, d = trial#
        csv_filename = self.userID + "-" + self.sessionNb + "-" + self.gestureNb + "-" + self.trialNb_str[int(self.gestureNb)] + "-" + self.arm + ".csv"
        csv_path = self.filepath + "/" + csv_filename
        with open(csv_path, 'w', newline='') as f:
            # create the csv writer
            writer = csv.writer(f)
            # write rows to the csv file
            writer.writerows(data)
            # close the file
            f.close()

        # increments the number of rep done for specific gesture
        self.trialNb[int(self.gestureNb)] += 1
        self.trialNb_str[int(self.gestureNb)] = "00" + str(self.trialNb[int(self.gestureNb)])

        # Update the counts on screen
        self.updateCountScreen()

        #Automatically changes the check box
        self.switch_button(self.gestureNb)

    # Gets the user desired filepath
    def path(self):
        self.filepath = str(QFileDialog.getExistingDirectory(self, "Select Directory"))

    def undoEvent(self):
        reply = QMessageBox.question(self, 'Undo', 'Are you sure you want to undo the last trial?',
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            return True
        else:
            return False

    def undo(self):
        user_certain = self.undoEvent()
        if user_certain:
            for _ in range(5):
                self.switch_button(self.gestureNb)
            self.trialNb[int(self.gestureNb)] -= 1
            self.trialNb_str[int(self.gestureNb)] = "00" + str(self.trialNb[int(self.gestureNb)])
            self.updateCountScreen()
            # print(self.trialNb)
        else:
            print("status quo")





def window():
    app = QApplication(sys.argv)
    win = MyWindow()
    win.show()
    sys.exit(app.exec_())

window()