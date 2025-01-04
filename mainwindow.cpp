#include "mainwindow.h"
#include "ui_mainwindow.h"


MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    timer   = new QTimer(this);
    connect(timer, SIGNAL(timeout()), this, SLOT(readFarme()));  // 时间到，读取当前摄像头信息
    connect(ui->openCamera, SIGNAL(clicked()), this, SLOT(openCamara()));
    connect(ui->closeCamera, SIGNAL(clicked()), this, SLOT(closeCamara()));
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::openCamara()
{
    capture.open(0);    //从摄像头读入视频如果设备只有一个摄像头就传入参数0
    qDebug("open");
    if (!capture.isOpened()) //是否打开摄像头
    {
        qDebug("err");
    }
    timer->start(50);//每50ms获取一帧
}

void MainWindow::readFarme()
{
    capture>>cap; //读取当前帧
    if (!cap.empty()) //当前帧是否捕捉成功
    {
        imag = Mat2QImage(cap);
        imag = imag.scaled(ui->camera->width(), ui->camera->height(),
                           Qt::IgnoreAspectRatio, Qt::SmoothTransformation);//设置图片大小和label的长宽一致

        ui->camera->setPixmap(QPixmap::fromImage(imag));  // 将图片显示到label上
    }
    else
        qDebug("can not ");
}

QImage  MainWindow::Mat2QImage(Mat cvImg)//图片转换
{
    QImage qImg;
    if(cvImg.channels()==3)     //3 channels color image
    {

        cvtColor(cvImg,cvImg,COLOR_BGR2RGB);
        qImg =QImage((const unsigned char*)(cvImg.data),
                      cvImg.cols, cvImg.rows,
                      cvImg.cols*cvImg.channels(),
                      QImage::Format_RGB888);
    }
    else if(cvImg.channels()==1)                    //grayscale image
    {
        qImg =QImage((const unsigned char*)(cvImg.data),
                      cvImg.cols,cvImg.rows,
                      cvImg.cols*cvImg.channels(),
                      QImage::Format_Indexed8);
    }
    else
    {
        qImg =QImage((const unsigned char*)(cvImg.data),
                      cvImg.cols,cvImg.rows,
                      cvImg.cols*cvImg.channels(),
                      QImage::Format_RGB888);
    }
    return qImg;
}

void MainWindow::closeCamara()
{
    timer->stop();
}
