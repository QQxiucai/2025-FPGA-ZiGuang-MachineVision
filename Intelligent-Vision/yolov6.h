// =======================================================
// 文件名: yolov6.h (修改后)
// =======================================================
#ifndef YOLOV6PROCESSOR_H
#define YOLOV6PROCESSOR_H

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

// --- 从官方 postprocess.h 移植过来的常量 ---
#define OBJ_NAME_MAX_SIZE 64
#define OBJ_NUMB_MAX_SIZE 128
#define OBJ_CLASS_NUM 80 // 确保这与您的模型匹配
// #define NMS_THRESH 0.35 // <-- 【移除】
// #define CONF_THRESH 0.5 // <-- 【移除】
// --- 结束移植 ---

class YOLOv6Processor : public QObject
{
    Q_OBJECT
public:
    explicit YOLOv6Processor(QObject *parent = nullptr);
    ~YOLOv6Processor();

public slots:
    // 由 VideoWindow 调用的槽
    void init(const QString& modelPath);
    void processFrame(const QImage& frame);

    // --- 【【【【新：添加这些槽函数】】】】 ---
    void setConfThreshold(float conf);
    void setNmsThreshold(float nms);
    void setAlertDuration(int seconds);

signals:
    // 发送回 VideoWindow 的信号
    void processingFinished(const QImage& resultImage, qint64 frame_time_ms);
    void alertNeeded(const QImage& alertImage);

private:
    // --- 跟踪器和报警逻辑 (与 NPUProcessor 相同) ---
    std::atomic<bool> m_is_busy;
    bool isInitialized;
    QElapsedTimer lastAlertTime;
    const qint64 alertCooldownMs = 30000;
    // const qint64 personAlertDurationMs = 5000; // <-- 【移除】
    qint64 m_personAlertDurationMs; // <-- 【【新】】

    std::vector<TrackedObject> tracked_objects;
    int next_track_id = 0;
    std::vector<std::string> classNames; // 将存储 COCO 标签

    // --- RKNN 相关成员 (与 NPUProcessor 相同) ---
    rknn_context ctx;
    unsigned char *model_data;
    int model_data_size;
    rknn_input_output_num io_num;
    rknn_tensor_attr *input_attrs;
    rknn_tensor_attr *output_attrs;
    rknn_input inputs[1];
    // 【修改】YOLOv6 可能有 6 或 9 个输出
    rknn_output outputs[9]; // 分配足够的空间 (3*box + 3*score + 3*score_sum)

    int input_width;
    int input_height;
    int input_channels;
    void* npu_input_buffer;
    size_t npu_input_buffer_size;

    // --- 【【【【新：添加这些成员变量】】】】 ---
    float m_confThreshold;
    float m_nmsThreshold;

    // --- (私有函数) ---

    // --- 跟踪器 (与 NPUProcessor 相同) ---
    float calculate_iou(const cv::Rect_<float>& rect1, const cv::Rect_<float>& rect2);
    void updateTracker(const std::vector<cv::Rect_<float>>& detections,
                       const std::vector<int>& classIds,
                       const std::vector<float>& confidences);

    // --- 【新】从 postprocess.cc 移植的 YOLOv6 核心函数 ---

    // 1. 主后处理函数 (替换了 postprocess_yolov5_3_output)
    int postprocess_yolov6(rknn_output *all_outputs,
                           const cv::Size& original_img_size,
                           std::vector<cv::Rect_<float>>& detections,
                           std::vector<int>& classIds,
                           std::vector<float>& confidences);

    // 2. 核心解码函数 (根据您的平台，我们使用 process_i8)
    int process_i8(int8_t *box_tensor, int32_t box_zp, float box_scale,
                   int8_t *score_tensor, int32_t score_zp, float score_scale,
                   int8_t *score_sum_tensor, int32_t score_sum_zp, float score_sum_scale,
                   int grid_h, int grid_w, int stride, int dfl_len,
                   std::vector<float> &boxes,
                   std::vector<float> &objProbs,
                   std::vector<int> &classId,
                   float threshold);

    // 3. DFL 解码 (YOLOv6 关键)
    void compute_dfl(float* tensor, int dfl_len, float* box);

    // 4. NMS 辅助函数
    int nms(int validCount, std::vector<float> &outputLocations, std::vector<int> classIds, std::vector<int> &order,
            int filterId, float threshold);
    int quick_sort_indice_inverse(std::vector<float> &input, int left, int right, std::vector<int> &indices);
    float CalculateOverlap(float xmin0, float ymin0, float xmax0, float ymax0, float xmin1, float ymin1, float xmax1, float ymax1);

    // 5. 数学/转换辅助函数
    float sigmoid(float x);
    inline int32_t __clip(float val, float min, float max);
    int8_t qnt_f32_to_affine(float f32, int32_t zp, float scale);
    float deqnt_affine_to_f32(int8_t qnt, int32_t zp, float scale);
};

#endif // YOLOV6PROCESSOR_H
