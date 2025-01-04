QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

CONFIG += c++17

# You can make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

SOURCES += \
    main.cpp \
    mainwindow.cpp \
    test.cpp

HEADERS += \
    mainwindow.h \
    test.h

FORMS += \
    mainwindow.ui

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target

INCLUDEPATH+=E:\opencv\build\install\include\opencv2\
             E:\opencv\build\install\include
LIBS += -L E:\opencv\build\install\x64\mingw\lib\libopencv_*.a

INCLUDEPATH+=E:\opencv\onnxruntime-win-x64-1.20.1\onnxruntime-win-x64-1.20.1\include

LIBS += -L E:\opencv\onnxruntime-win-x64-1.20.1\onnxruntime-win-x64-1.20.1\lib\onnxruntime.lib
