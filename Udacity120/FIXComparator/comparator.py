# -*- coding: utf-8 -*-
"""
Created on Sat Oct  8 22:44:24 2016

@author: ALEXK
"""

import sys
from PyQt4.QtCore import pyqtSlot
from PyQt4.QtGui import *
 
# Create an PyQT4 application object.
a = QApplication(sys.argv)
 
# The QWidget widget is the base class of all user interface objects in PyQt4.
w = QWidget()

w.setWindowTitle('Compare 2 .fix files')

# Create textbox
textbox = QLineEdit(w)
textbox.move(20, 20)
textbox.resize(280,40)

# Create textbox
textbox2 = QLineEdit(w)
textbox2.move(320, 20)
textbox.resize(280,40)
 
# Set window size.
w.resize(600, 150)
 
# Create a button in the window
button = QPushButton('Click me', w)
button.move(20,80)
 
# Create the actions
@pyqtSlot()
def on_click():
    textbox.setText("Button clicked.")
 
# connect the signals to the slots
button.clicked.connect(on_click)

w.show()
a.exec_()
