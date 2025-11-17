// =======================================================
// 文件名: npu_processor.h (修改后)
// =======================================================
#ifndef NPUPROCESSOR_H
#define NPUPROCESSOR_H

#include <QObject>
#include <QImage>
#include <QElapsedTimer>
#include <opencv2/core/types.hpp>
#include <vector>
#include <map>
#include <atomic>
#include "rknn_api.h"
#include "tracker_types.h"

// RGA 头文件
#include "rga/RgaApi.h"
#include "rga/im2d.h"

class NPUProcessor : public QObject
{
    Q_OBJECT
public:
    explicit NPUProcessor(QObject *parent = nullptr);
    ~NPUProcessor();

public slots:
    void init(const QString& modelPath);
    void processFrame(const QImage& frame);

    // --- 【【【【新：添加这些槽函数】】】】 ---
    void setConfThreshold(float conf);
    void setNmsThreshold(float nms);
    void setAlertDuration(int seconds);

signals:
    void processingFinished(const QImage& resultImage, qint64 frame_time_ms);
    void alertNeeded(const QImage& alertImage);

private:
    std::atomic<bool> m_is_busy;
    bool isInitialized;
    QElapsedTimer lastAlertTime;
    const qint64 alertCooldownMs = 30000;
    // const qint64 personAlertDurationMs = 5000; // <-- 【移除】
    qint64 m_personAlertDurationMs; // <-- 【【新】】

    std::vector<TrackedObject> tracked_objects;
    int next_track_id = 0;
    std::vector<std::string> classNames;

    // --- (RKNN 相关成员变量) ---
    rknn_context ctx;
    unsigned char *model_data;
    int model_data_size;
    rknn_input_output_num io_num;
    rknn_tensor_attr *input_attrs;
    rknn_tensor_attr *output_attrs; // 【修改】将分配 3 个
    rknn_input inputs[1];
    rknn_output outputs[3]; // 【修改】从 [1] 改为 [3]

    int input_width;
    int input_height;
    int input_channels;

    // NPU 输入缓冲区
    void* npu_input_buffer;
    size_t npu_input_buffer_size;

    // --- 【新】YOLOv5 (3-Output) 特有参数 ---
    std::vector<int> m_strides;
    std::vector<std::vector<cv::Size>> m_anchor_grid;
    // --- 结束修改 ---

    static constexpr int NUM_CLASSES_NPU = 80;
    // static constexpr float CONF_THRESHOLD_NPU = 0.5f; // <-- 【移除】
    // static constexpr float NMS_THRESHOLD_NPU = 0.1f; // <-- 【移除】

    // --- 【【【【新：添加这些成员变量】】】】 ---
    float m_confThreshold;
    float m_nmsThreshold;

    // --- (私有函数) ---
    float calculate_iou(const cv::Rect_<float>& rect1, const cv::Rect_<float>& rect2);
    void updateTracker(const std::vector<cv::Rect_<float>>& detections,
                       const std::vector<int>& classIds,
                       const std::vector<float>& confidences);

    // 【修改】重命名后处理函数
    int postprocess_yolov5_3_output(rknn_output *all_outputs,
                                    const cv::Size& original_img_size,
                                    std::vector<cv::Rect_<float>>& detections,
                                    std::vector<int>& classIds,
                                    std::vector<float>& confidences);

    float sigmoid(float x);
};

#endif // NPUPROCESSOR_H
