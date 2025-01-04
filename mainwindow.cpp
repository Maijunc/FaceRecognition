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

    try {
        net = cv::dnn::readNetFromONNX("E:\\opencv\\yolov8n-face-lindevs.onnx");

        // 设置计算后端和目标
        // 检查是否有可用的CUDA设备（即检查是否可以使用GPU进行加速）
        int deviceID = cv::cuda::getCudaEnabledDeviceCount();
        if(deviceID == 1)
        {
            // 如果有可用的CUDA设备，将网络的推理后端设置为CUDA以使用GPU
            this->net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
            this->net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
        }
        else
        {
            // 如果没有检测到CUDA设备，则弹出消息框提示用户当前使用CPU进行推理
            QMessageBox::information(NULL, "warning", QStringLiteral("正在使用CPU推理！\n"), QMessageBox::Yes, QMessageBox::Yes);
        }
        qDebug() << "ONNX 模型加载成功！";
    } catch (const cv::Exception &e) {
        qDebug() << "加载模型失败: " << e.what();
    }
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
        cv::Mat originImg = cap;
        imag = imag.scaled(ui->camera->width(), ui->camera->height(),
                           Qt::IgnoreAspectRatio, Qt::SmoothTransformation);//设置图片大小和label的长宽一致
        ui->camera->setPixmap(QPixmap::fromImage(imag));  // 将图片显示到label上
        // 图像预处理，确保输入符合模型要求
        const cv::Size inputSize(640, 640);
        // 将输入图像转换为神经网络的blob格式，并进行归一化和大小调整
        cv::resize(cap, cap, cv::Size(640, 640));
        cv::Mat blob = cv::dnn::blobFromImage(cap, 1 / 255.0, inputSize, cv::Scalar(0, 0, 0), true);
        // CV_EXPORTS_W Mat blobFromImage(InputArray image, double scalefactor=1.0, const Size& size = Size(),
        //                                const Scalar& mean = Scalar(), bool swapRB=false, bool crop=false,
        //                                int ddepth=CV_32F);

        std::cout << "Blob dimensions: " << blob.dims << std::endl;
        for (int i = 0; i < blob.dims; ++i) {
            std::cout << "Blob dimension " << i << ": " << blob.size[i] << std::endl;
        }

        // 使用模型进行推理
        //cv::Mat result = runModelInference(blob);

        net.setInput(blob);
        std::vector<cv::Mat> outs;
        net.forward(outs, net.getUnconnectedOutLayersNames());

        // 处理推理结果并显示
        // 处理推理结果并显示
        // 假设outs包含检测到的目标数据 (例如，YOLO模型的输出)

        processInferenceResult(cap, outs);

        vector<double> layersTimes;
        double freq = getTickFrequency() / 1000;
        double t = net.getPerfProfile(layersTimes) / freq;
        string label = format("%s Inference time : %.2f ms", "yoloV8", t);
        putText(cap, label, Point(0, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 2);

        // 转换 Mat 为 QImage 进行显示
        // imag = Mat2QImage(cap);
        // 更新界面，显示处理后的图像
        QImage qimg = Mat2QImage(originImg);  // 假设你有Mat2QImage函数来转换图像格式
        qimg = imag.scaled(ui->camera->width(), ui->camera->height(), Qt::IgnoreAspectRatio, Qt::SmoothTransformation);
        ui->result->setPixmap(QPixmap::fromImage(qimg));
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

// 图像预处理：调整大小、归一化等
cv::Mat MainWindow::preprocessImageForModel(const cv::Mat &frame)
{
    cv::Mat resized;
    cv::Mat inputBlob;

    // 例如：调整图像大小到模型所需输入尺寸，例如 224x224
    cv::resize(frame, resized, cv::Size(640, 640));

    // 归一化处理：减去均值，按标准差缩放等（根据模型的训练方式进行调整）
    cv::dnn::blobFromImage(resized, inputBlob, 1.0 / 255, cv::Size(640, 640), cv::Scalar(0, 0, 0), true, false);

    return inputBlob;
}

// 使用加载的模型进行推理
cv::Mat MainWindow::runModelInference(const cv::Mat &inputBlob)
{
    net.setInput(inputBlob);
    return net.forward(); // 获取输出
}

void MainWindow::processInferenceResult(Mat& frame, const vector<Mat>& outs)
{
    vector<int> classIds;
    vector<float> confidences;
    vector<Rect> boxes;

    for (size_t i = 0; i < outs.size(); ++i)
    {
        // Scan through all the bounding boxes output from the network and keep only the
        // ones with high confidence scores. Assign the box's class label as the class
        // with the highest score for the box.
        float* data = (float*)outs[i].data;
        for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
        {
            Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
            Point classIdPoint;
            double confidence;
            // Get the value and location of the maximum score
            minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
            if (confidence > this->confThreshold)
            {
                int centerX = (int)(data[0] * frame.cols);
                int centerY = (int)(data[1] * frame.rows);
                int width = (int)(data[2] * frame.cols);
                int height = (int)(data[3] * frame.rows);
                int left = centerX - width / 2;
                int top = centerY - height / 2;

                classIds.push_back(classIdPoint.x);
                confidences.push_back((float)confidence);
                boxes.push_back(Rect(left, top, width, height));
            }
        }
    }

    // Perform non maximum suppression to eliminate redundant overlapping boxes with
    // lower confidences
    vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, this->confThreshold, this->nmsThreshold, indices);
    for (size_t i = 0; i < indices.size(); ++i)
    {
        int idx = indices[i];
        Rect box = boxes[idx];
        this->drawPred(classIds[idx], confidences[idx], box.x, box.y,
                       box.x + box.width, box.y + box.height, frame);
    }

}

void MainWindow::drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame)   // Draw the predicted bounding box
{
    //Draw a rectangle displaying the bounding box
    rectangle(frame, Point(left, top), Point(right, bottom), Scalar(0, 0, 255), 3);

    //Get the label for the class name and its confidence
    string label = format("%.2f", conf);
    if (!this->classes.empty())
    {
        CV_Assert(classId < (int)this->classes.size());
        label = this->classes[classId] + ":" + label;
    }

    //Display the label at the top of the bounding box
    int baseLine;
    Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    top = max(top, labelSize.height);
    //rectangle(frame, Point(left, top - int(1.5 * labelSize.height)), Point(left + int(1.5 * labelSize.width), top + baseLine), Scalar(0, 255, 0), FILLED);
    putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 255, 0), 1);
}


