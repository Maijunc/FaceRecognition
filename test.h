#ifndef TEST_H
#define TEST_H

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <string>
#include <vector>
using cv::Mat;
using std::cout;
using std::endl;
using std::string;
using std::vector;

class test
{
public:
    test();


    void print_result(const Mat &result, float conf = 0.7, int len_data = 15)
    {
        float *pdata = (float *)result.data;
        for (int i = 0; i < result.total() / len_data; i++)
        {
            if (pdata[4] > conf)
            {
                for (int j = 0; j < len_data; j++)
                {
                    cout << pdata[j] << " ";
                }
                cout << endl;
            }
            pdata += len_data;
        }
        return;
    }

    vector<vector<float>> get_info(const Mat &result, float conf = 0.7, int len_data = 15)
    {
        float *pdata = (float *)result.data;
        vector<vector<float>> info;
        for (int i = 0; i < result.total() / len_data; i++)
        {
            if (pdata[4] > conf)
            {
                vector<float> info_line;
                for (int j = 0; j < len_data; j++)
                {
                    // cout << pdata[j] << " ";
                    info_line.push_back(pdata[j]);
                }
                // cout << endl;
                info.push_back(info_line);
            }
            pdata += len_data;
        }
        return info;
    }

    void info_simplify(vector<vector<float>> &info)
    {
        for (auto i = 0; i < info.size(); i++)
        {
            info[i][5] = std::max_element(info[i].cbegin() + 5, info[i].cend()) - (info[i].cbegin() + 5);
            info[i].resize(6);
            float x = info[i][0];
            float y = info[i][1];
            float w = info[i][2];
            float h = info[i][3];
            info[i][0] = x - w / 2.0;
            info[i][1] = y - h / 2.0;
            info[i][2] = x + w / 2.0;
            info[i][3] = y + h / 2.0;
        }
    }

    vector<vector<vector<float>>> split_info(vector<vector<float>> &info)
    {
        vector<vector<vector<float>>> info_split;
        vector<int> class_id;
        for (auto i = 0; i < info.size(); i++)
        {
            if (std::find(class_id.begin(), class_id.end(), (int)info[i][5]) == class_id.end())
            {
                class_id.push_back((int)info[i][5]);
                vector<vector<float>> info_;
                info_split.push_back(info_);
            }
            info_split[std::find(class_id.begin(), class_id.end(), (int)info[i][5]) - class_id.begin()].push_back(info[i]);
        }
        return info_split;
    }

    void nms(vector<vector<float>> &info, float iou = 0.4)
    {
        int counter = 0;
        vector<vector<float>> return_info;
        while (counter < info.size())
        {
            return_info.clear();
            float x1 = 0;
            float x2 = 0;
            float y1 = 0;
            float y2 = 0;
            std::sort(info.begin(), info.end(), [](vector<float> p1, vector<float> p2)
                      { return p1[4] > p2[4]; });
            for (auto i = 0; i < info.size(); i++)
            {
                if (i < counter)
                {
                    return_info.push_back(info[i]);
                    continue;
                }
                if (i == counter)
                {
                    x1 = info[i][0];
                    y1 = info[i][1];
                    x2 = info[i][2];
                    y2 = info[i][3];
                    return_info.push_back(info[i]);
                    continue;
                }
                if (info[i][0] > x2 or info[i][2] < x1 or info[i][1] > y2 or info[i][3] < y1)
                {
                    return_info.push_back(info[i]);
                }
                else
                {
                    float over_x1 = std::max(x1, info[i][0]);
                    float over_y1 = std::max(y1, info[i][1]);
                    float over_x2 = std::min(x2, info[i][2]);
                    float over_y2 = std::min(y2, info[i][3]);
                    float s_over = (over_x2 - over_x1) * (over_y2 - over_y1);
                    float s_total = (x2 - x1) * (y2 - y1) + (info[i][0] - info[i][2]) * (info[i][1] - info[i][3]) - s_over;
                    if (s_over / s_total < iou)
                    {
                        return_info.push_back(info[i]);
                    }
                }
            }
            info = return_info;
            counter += 1;
        }
    }


    void draw_box(Mat &img, const vector<vector<float>> &info)
    {
        for (int i = 0; i < info.size(); i++)
        {
            cv::rectangle(img, cv::Point(info[i][0], info[i][1]), cv::Point(info[i][2], info[i][3]), cv::Scalar(0, 255, 0));
            string label;
            label += class_name[info[i][5]];
            label += "  ";
            std::stringstream oss;
            oss << info[i][4];
            label += oss.str();
            cv::putText(img, label, cv::Point(info[i][0], info[i][1]), 1, 2, cv::Scalar(0, 255, 0), 2);

        }
    }

    const vector<string> class_name = {"cat", "chicken", "cow", "dog", "fox", "goat", "horse", "person", "racoon", "skunk"};

};


#endif // TEST_H
