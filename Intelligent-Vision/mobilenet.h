// =======================================================
// 文件名: mobilenet.h (完整修改版)
// =======================================================
#ifndef MOBILENET_H
#define MOBILENET_H

#include <QObject>
#include <QImage>
#include <QElapsedTimer>
#include <opencv2/core/types.hpp>
#include <vector>
#include <string>
#include <atomic>
#include "rknn_api.h"

// RGA 头文件
#include "rga/RgaApi.h"
#include "rga/im2d.h"

// (结构体定义保持不变)
typedef struct {
    int cls;
    float score;
} mobilenet_result;

typedef struct {
    float value;
    int index;
} element_t;


class MobilenetProcessor : public QObject
{
    Q_OBJECT
public:
    explicit MobilenetProcessor(QObject *parent = nullptr);
    ~MobilenetProcessor();

public slots:
    void init(const QString& modelPath);
    void processFrame(const QImage& frame);

    // --- 【【新：添加这个槽函数】】 ---
    void setTopK(int k);

signals:
    void processingFinished(const QImage& resultImage, qint64 frame_time_ms);

private:
    std::atomic<bool> m_is_busy;
    bool isInitialized;
    std::vector<std::string> m_labels;

    // --- 【【新：添加 TopK 成员变量】】 ---
    int m_topK; // 用于控制显示数量

    // RKNN 相关成员
    rknn_context ctx;
    unsigned char *model_data;
    int model_data_size;
    rknn_input_output_num io_num;
    rknn_tensor_attr *input_attrs;
    rknn_tensor_attr *output_attrs;
    rknn_input inputs[1];
    rknn_output outputs[1];

    int input_width;   // NPU 输入宽度 (224)
    int input_height;  // NPU 输入高度 (224)
    int input_channels;
    void* npu_input_buffer; // NPU (224x224) 缓冲
    size_t npu_input_buffer_size;

    // (显示缓冲)
    int display_width; // 显示宽度 (640)
    int display_height; // 显示高度 (640)
    void* display_buffer_640; // 显示 (640x640) 缓冲
    size_t display_buffer_size;

    // (私有函数保持不变)
    bool loadLabels(const QString& labelPath);
    void softmax(float* array, int size);
    void get_topk_with_indices(float arr[], int size, int k, mobilenet_result* result);
    void quick_sort(element_t arr[], int low, int high);
    int partition(element_t arr[], int low, int high);
    void swap(element_t* a, element_t* b);
};

#endif // MOBILENET_H
