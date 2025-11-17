// =======================================================
// 文件名: cpu_processor.h (完整修改版)
// =======================================================
#ifndef CPUPROCESSOR_H
#define CPUPROCESSOR_H

#include <QObject>
#include <QImage>
#include <QElapsedTimer>
#include <opencv2/dnn.hpp>
#include <vector>
#include <map>
#include <atomic> // <-- 确保包含
#include "tracker_types.h"
#include <opencv2/core/types.hpp> // <-- 【【新：确保包含】】

class CPUProcessor : public QObject
{
    Q_OBJECT
public:
    explicit CPUProcessor(QObject *parent = nullptr);
    ~CPUProcessor();

public slots:
    void init(const QString& modelPath);
    void processFrame(const QImage& frame);

    // --- (槽函数不变) ---
    void setConfThreshold(float conf);
    void setNmsThreshold(float nms);
    void setAlertDuration(int seconds);

signals:
    void processingFinished(const QImage& resultImage, qint64 frame_time_ms);
    void alertNeeded(const QImage& alertImage);


private:
    std::atomic<bool> m_is_busy;
    bool isInitialized;
    cv::dnn::Net net;
    std::vector<std::string> classNames;

    float m_confThreshold;
    float m_nmsThreshold;

    int inputWidth = 640;
    int inputHeight = 640;
    std::vector<TrackedObject> tracked_objects;
    int next_track_id = 0;

    QElapsedTimer lastAlertTime;
    const qint64 alertCooldownMs = 30000;
    qint64 m_personAlertDurationMs;


    // --- 【【新：添加YOLOv5-3输出的参数，与NPU版本同步】】 ---
    std::vector<int> m_strides;
    std::vector<std::vector<cv::Size>> m_anchor_grid;
    static constexpr int NUM_CLASSES_CPU = 80;
    float sigmoid(float x);
    // --- 结束添加 ---


    float calculate_iou(const cv::Rect_<float>& rect1, const cv::Rect_<float>& rect2);
    void updateTracker(const std::vector<cv::Rect_<float>>& detections,
                       const std::vector<int>& classIds,
                       const std::vector<float>& confidences);
    cv::Mat format_yolov5(const cv::Mat &source);

    // --- 【【修改：detect 函数现在将处理3个输出】】 ---
    void detect(const cv::Mat &image_blob, cv::Mat &original_image,
                std::vector<cv::Rect_<float>>& detections,
                std::vector<int>& classIds,
                std::vector<float>& confidences);
};

#endif // CPUPROCESSOR_H
