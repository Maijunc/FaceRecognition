#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QDebug>
#include <QTimer>
#include <QImage>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <QMessageBox>

using namespace cv;
using namespace std;

QT_BEGIN_NAMESPACE
namespace Ui {
class MainWindow;
}
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

    QImage  Mat2QImage(Mat cvImg);
private slots:
    void openCamara();      // 打开摄像头
    void readFarme();       // 读取当前帧信息
    void closeCamara();     // 关闭摄像头。

private:
    Ui::MainWindow *ui;

    QTimer          *timer;
    QImage          imag;
    Mat             cap,cap_gray,cap_tmp; //定义一个Mat变量，用于存储每一帧的图像
    VideoCapture    capture; //声明视频读入类
    cv::dnn::Net net;
    cv::Mat preprocessImageForModel(const cv::Mat &frame);
    cv::Mat runModelInference(const cv::Mat &inputBlob);
    void processInferenceResult(Mat& frame, const vector<Mat>& outs);

    float confThreshold = 0.1f;
    float nmsThreshold = 0.1f;
    void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat &frame);
    vector<string> classes;
};
#endif // MAINWINDOW_H
