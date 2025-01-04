#include "mainwindow.h"
#include "test.h"

#include <QApplication>

int main(int argc, char *argv[])
{
    // QApplication a(argc, argv);
    // MainWindow w;
    // w.show();
    // return a.exec();
    test *myTest = new test();

    cv::dnn::Net net = cv::dnn::readNetFromONNX("F:\\yolo\\yolov5_cpp\\yolov5_cpp\\opencv_yolov5\\best.onnx");
    Mat img = cv::imread("F:\\yolo\\yolov5_cpp\\yolov5_cpp\\opencv_yolov5\\fox.jpg");
    cv::resize(img, img, cv::Size(640, 640));
    Mat blob = cv::dnn::blobFromImage(img, 1.0 / 255.0, cv::Size(640, 640), cv::Scalar(), true);
    net.setInput(blob);
    vector<Mat> netoutput;
    vector<string> out_name = {"output"};
    net.forward(netoutput, out_name);
    Mat result = netoutput[0];
    // print_result(result);
    vector<vector<float>> info = myTest->get_info(result);
    myTest->info_simplify(info);
    vector<vector<vector<float>>> info_split = myTest->split_info(info);
    // cout << " split info" << endl;
    // print_info(info_split[0]);
    // cout << info.size() << " " << info[0].size() << endl;

    for(auto i=0; i < info_split.size(); i++)
    {
        myTest->nms(info_split[i]);
        myTest->draw_box(img, info_split[i]);
    }

    // nms(info_split[0]);
    // cout << "nms" << endl;
    // print_info(info_split[0]);
    // draw_box(img, info_split[0]);
    cv::imshow("test", img);
    cv::waitKey(0);
    return 0;
}
