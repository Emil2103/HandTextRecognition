# import matplotlib
# from PyQt5.QtWidgets import QMainWindow, QApplication, QMenu, QMenuBar, QAction, QFileDialog, QPushButton, QTextBrowser
# from PyQt5.QtGui import QIcon, QImage, QPainter, QPen, QBrush
# from PyQt5.QtCore import Qt, QPoint
# import sys
# from PyQt5.QtWidgets import QMainWindow, QTextEdit, QAction, QApplication
# from PyQt5.QtWidgets import (QWidget, QLabel, QLineEdit, QTextEdit, QGridLayout, QApplication)
# import numpy as np
# from tensorflow import keras
# from PIL import Image
#
# class Window(QMainWindow):
#     def __init__(self):
#         super().__init__()
#
#         title = "RECOGNITION"
#         top = 200
#         left = 200
#         width = 540
#         height = 340
#
#         self.drawing = False
#         self.brushSize = 8
#         self.brushColor = Qt.black
#         self.lastPoint = QPoint()
#
#         self.image = QImage(278, 278, QImage.Format_RGB32)
#         self.image.fill(Qt.white)
#
#         self.nameLabel = QLabel(self)
#         self.nameLabel.setText('RES:')
#         self.line = QLineEdit(self)
#
#         self.line.move(360, 168)
#         self.line.resize(99, 42)
#         self.nameLabel.move(290, 170)
#
#         prediction_button = QPushButton('RECOGNITION', self)
#         prediction_button.move(290, 30)
#         prediction_button.resize(230, 33)
#         prediction_button.clicked.connect(self.save)
#         prediction_button.clicked.connect(self.predicting)
#
#         clean_button = QPushButton('CLEAN', self)
#         clean_button.move(290, 100)
#         clean_button.resize(230, 33)
#         clean_button.clicked.connect(self.clear)
#
#         self.setWindowTitle(title)
#         self.setGeometry(top, left, width, height)
#
#     def print_letter(self,result):
#         letters = "ЁАБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"
#         self.line.setText(letters[result])
#         return letters[result]
#
#     def predicting(self):
#         image = keras.preprocessing.image
#         model = keras.models.load_model('Ycpeh.h5')
#         img = image.load_img('res.jpeg', target_size=(28, 28))
#         x = image.img_to_array(img)
#         x = np.expand_dims(x, axis=0)
#         images = np.vstack([x])
#         classes = model.predict(images, batch_size=256)
#         result = int(np.argmax(classes))
#         self.print_letter(result)
#
#     def mousePressEvent(self, event):
#         if event.button() == Qt.LeftButton:
#             self.drawing = True
#             self.lastPoint = event.pos()
#
#     def mouseMoveEvent(self, event):
#         if (event.buttons() & Qt.LeftButton) & self.drawing:
#             painter = QPainter(self.image)
#             painter.setPen(QPen(self.brushColor, self.brushSize, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
#             painter.drawLine(self.lastPoint, event.pos())
#             self.lastPoint = event.pos()
#             self.update()
#
#     def mouseReleaseEvent(self, event):
#
#         if event.button() == Qt.LeftButton:
#             self.drawing = False
#
#     def paintEvent(self, event):
#         canvasPainter = QPainter(self)
#         canvasPainter.drawImage(0, 0, self.image)
#
#     def save(self):
#         self.image.save('res.jpeg')
#
#     def clear(self):
#         self.image.fill(Qt.white)
#         self.update()
#
# if __name__ == "__main__":
#     app = QApplication(sys.argv)
#     window = Window()
#     window.show()
#     app.exec()

from imutils.perspective import four_point_transform
from imutils import contours
import tensorflow as tf
from tensorflow import keras
from keras.layers import *
import keras.backend as K
import matplotlib.pyplot as plt
import imutils
import numpy as np
import cv2
import os
import PIL
from PIL import ImageTk, Image, ImageDraw
import tkinter as tk
from tkinter import *
import string

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

w = 1000
h = 250
center = h / 2
white = (255, 255, 255)
lastx, lasty = None, None

global letters
global words

def letters_extract(image_file: str):
    img = cv2.imread(image_file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    img_erode = cv2.erode(thresh, np.ones((3, 3), np.uint8), iterations=1)

    contours, hierarchy = cv2.findContours(img_erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    output = img.copy()
    letters = []
    for idx, contour in enumerate(contours):
        (x, y, w, h) = cv2.boundingRect(contour)
        if hierarchy[0][idx][3] == 0:
            cv2.rectangle(output, (x, y), (x + w, y + h), (70, 0, 0), 1)
            letter_crop = gray[y:y + h, x:x + w]
            size_max = max(w, h)
            letter_square = 255 * np.ones(shape=[size_max, size_max], dtype=np.uint8)
            if w > h:
                y_pos = size_max//2 - h//2
                letter_square[y_pos:y_pos + h, 0:w] = letter_crop
            elif w < h:
                x_pos = size_max//2 - w//2
                letter_square[0:h, x_pos:x_pos + w] = letter_crop
            else:
                letter_square = letter_crop

            letters.append((x, w, cv2.resize(letter_square, (100, 100), interpolation=cv2.INTER_AREA)))

    # Sort array in place by X-coordinate
    letters.sort(key=lambda x: x[0], reverse=False)
    # cv2.imshow("0", letters[0][2])
    # cv2.imshow("1", letters[1][2])
    # cv2.imshow("2", letters[2][2])
    # cv2.imshow("3", letters[3][2])
    # cv2.imshow("4", letters[4][2])
    # cv2.imshow("5", letters[5][2])
    # cv2.imshow("6", letters[6][2])
    # cv2.imshow("7", letters[7][2])
    # cv2.imshow("8", letters[8][2])
    # cv2.imshow("9", letters[9][2])
    # cv2.imshow("10", letters[10][2])
    # cv2.imshow("11", letters[11][2])
    # cv2.imshow("12", letters[12][2])
    # cv2.imshow("13", letters[13][2])
    return letters


def pred_Pomogi():
    result = []
    image = keras.preprocessing.image
    model = keras.models.load_model('Aboba.hdf5')
    #img = image.load_img('image.png', target_size=(32, 128))
    arr_img = letters_extract('image.jpg')
    for i in range(len(arr_img)):
        rgb = cv2.cvtColor(arr_img[i][2], cv2.COLOR_BGR2RGB)
        x = image.img_to_array(rgb)
        x = np.expand_dims(x, axis=0)
        images = np.vstack([x])
        classes = model.predict(images, batch_size=32)
        count = int(np.argmax(classes))
        letters = "ЁАБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"
        result += letters[count]
    print(result)
    return result

def smoothBrush(event):
    global lastx, lasty
    cv.bind('<B1-Motion>', paint)
    lastx, lasty = event.x, event.y

def paint(event):
    global lastx, lasty
    x, y = event.x, event.y
    cv.create_line((lastx, lasty, x, y), fill='black', width=10, capstyle=ROUND)
    #  --- PIL
    draw.line((lastx, lasty, x, y), fill='black', width=10, joint='curve')
    lastx, lasty = x, y

def convert():
    filename = "image.jpg"
    image.save(filename)
    pred = pred_Pomogi()
    text.insert(tk.END, pred)

def clear():
    cv.delete('all')
    words = ""  # clear words
    letters = []
    draw.rectangle((0, 0, 1000, 250), fill=(255, 255, 255, 0))
    text.delete('1.0', END)

def save():
    image.save('text_dection_1.png')

win = tk.Tk()
win.geometry("1000x450")
win.configure(bg='white')

frame1 = tk.Frame(win, bg='white')

frame1.pack()

cv = Canvas(frame1, bg='white', height=h, width=w)
cv.pack()

image = PIL.Image.new("RGB", (w, h), white)
draw = ImageDraw.Draw(image)

txt = tk.Text(win, bd=3, exportselection=0, bg='WHITE', font='Helvetica',
              padx=10, pady=10, height=5, width=20)

cv.bind('<1>', smoothBrush)
cv.pack(expand=YES, fill=BOTH)

frame2 = tk.Frame(win, bg='white')
frame2.pack()
# Text area
text = tk.Text(frame2, width=71, height=5)
text.pack()

# Clear Button
btnClear = Button(frame2, text="clear", command=clear)
btnClear.pack(padx=20, pady=5)

# Predict Button
btnConvert = Button(frame2, text='convert', command=convert)
btnConvert.pack()

# Save
btnSave = Button(frame2, text='save', command=save)
btnSave.pack()

win.title('Painter')
win.mainloop()
