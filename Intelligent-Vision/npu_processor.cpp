// =======================================================
// 文件名: npu_processor.cpp
// =======================================================
#include "npu_processor.h"
#include <QDebug>
#include <QThread>
#include <QElapsedTimer>
#include <QPainter>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/highgui.hpp>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <sstream>
#include <iomanip>

// RKNN API
#include "rknn_api.h"
// RGA API
#include "rga/RgaApi.h"
#include "rga/im2d.h"


// --- COCO 类别名称 (80类) ---
const std::vector<std::string> COCO_NAMES_NPU = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
};

// --- Rockchip 官方的辅助函数 (int8_t) ---
inline static int clamp(float val, int min, int max) { return val > min ? (val < max ? val : max) : min; }
inline static int32_t __clip(float val, float min, float max)
{
    float f = val <= min ? min : (val >= max ? max : val);
    return f;
}
static int8_t qnt_f32_to_affine_i8(float f32, int32_t zp, float scale)
{
    float dst_val = (f32 / scale) + zp;
    int8_t res = (int8_t)__clip(dst_val, -128, 127);
    return res;
}
static float deqnt_affine_i8_to_f32(int8_t qnt, int32_t zp, float scale) { return ((float)qnt - (float)zp) * scale; }
// --- 结束辅助函数 ---

float NPUProcessor::sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// --- 构造函数 ---
NPUProcessor::NPUProcessor(QObject *parent)
    : QObject(parent),
      m_is_busy(false),
      isInitialized(false),
      model_data(nullptr),
      model_data_size(0),
      ctx(0),
      input_attrs(nullptr),
      output_attrs(nullptr),
      input_width(0), input_height(0), input_channels(0),
      next_track_id(0),
      npu_input_buffer(nullptr),
      npu_input_buffer_size(0),
      m_confThreshold(0.5f),   // <-- 【初始化默认值】
      m_nmsThreshold(0.1f),    // <-- 【初始化默认值】
      m_personAlertDurationMs(5000) // <-- 【初始化默认值】
{
    lastAlertTime.start();
    lastAlertTime.invalidate();
    m_strides = {8, 16, 32};
    m_anchor_grid.push_back({{10, 13}, {16, 30}, {33, 23}});
    m_anchor_grid.push_back({{30, 61}, {62, 45}, {59, 119}});
    m_anchor_grid.push_back({{116, 90}, {156, 198}, {373, 326}});
    // 使用 m_personAlertDurationMs
    qDebug() << "NPUProcessor created. Alert cooldown:" << alertCooldownMs << "ms. Person duration threshold:" << m_personAlertDurationMs << "ms";
}

// --- 析构函数  ---
NPUProcessor::~NPUProcessor()
{
    qDebug() << "NPU Processor destroying...";
    if (ctx > 0) {
        qDebug() << "  Releasing RKNN context...";
        rknn_destroy(ctx);
        ctx = 0;
    }
    if (model_data) {
        qDebug() << "  Freeing model data buffer...";
        free(model_data);
        model_data = nullptr;
    }
    if (input_attrs) {
        qDebug() << "  Freeing input attributes...";
        delete[] input_attrs;
        input_attrs = nullptr;
    }
     if (output_attrs) {
        qDebug() << "  Freeing output attributes...";
        delete[] output_attrs;
        output_attrs = nullptr;
     }
    if (npu_input_buffer) {
        qDebug() << "  Freeing NPU input buffer...";
        free(npu_input_buffer);
        npu_input_buffer = nullptr;
    }
    qDebug() << "NPU Processor destroyed.";
}

// --- load_model ---
static unsigned char *load_model(const char *filename, int *model_size) {
    FILE *fp = fopen(filename, "rb");
    if (fp == nullptr) { qCritical("fopen %s fail!", filename); return nullptr; }
    fseek(fp, 0, SEEK_END);
    int size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    unsigned char *data = (unsigned char *)malloc(size);
    if (data == nullptr) { qCritical("malloc model data buffer fail!"); fclose(fp); return nullptr; }
    int read_ret = fread(data, 1, size, fp);
    if (read_ret != size) { qCritical("fread model fail!"); fclose(fp); free(data); return nullptr; }
    *model_size = size;
    fclose(fp);
    return data;
}


// --- init 函数  ---
void NPUProcessor::init(const QString& modelPath)
{
    qDebug() << "NPU Processor initializing in thread:" << QThread::currentThreadId();
    isInitialized = false;

    model_data = load_model(modelPath.toStdString().c_str(), &model_data_size);
    if (model_data == nullptr) { qCritical() << "Failed to load RKNN model from" << modelPath; return; }
    qDebug() << "RKNN model loaded, size:" << model_data_size << "bytes";

    int ret = rknn_init(&ctx, model_data, model_data_size, 0, NULL);
    if (ret < 0) { qCritical() << "rknn_init error ret=" << ret; free(model_data); model_data = nullptr; return; }
    qDebug() << "RKNN context initialized.";

    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != RKNN_SUCC) { qCritical() << "rknn_query(RKNN_QUERY_IN_OUT_NUM) error ret=" << ret; rknn_destroy(ctx); ctx=0; return; }
    qDebug() << "Model info: inputs=" << io_num.n_input << ", outputs=" << io_num.n_output;

    if (io_num.n_input != 1) { qCritical() << "Model requires 1 input, but got" << io_num.n_input; rknn_destroy(ctx); ctx=0; return; }

    if (io_num.n_output != 3) {
        qCritical() << "Model requires 3 outputs, but got" << io_num.n_output;
        rknn_destroy(ctx);
        ctx = 0; // 【Segmentation fault】
        return;
    }

    qDebug() << "Querying input tensor attributes...";
    input_attrs = new rknn_tensor_attr[io_num.n_input];
    memset(input_attrs, 0, io_num.n_input * sizeof(rknn_tensor_attr));
    input_attrs[0].index = 0;
    ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[0]), sizeof(rknn_tensor_attr));
    if (ret != RKNN_SUCC) {
        qCritical() << "rknn_query(RKNN_QUERY_INPUT_ATTR) error ret=" << ret;
        rknn_destroy(ctx); ctx=0; delete[] input_attrs; input_attrs = nullptr;
        return;
    }

    if (input_attrs[0].fmt == RKNN_TENSOR_NCHW) {
        input_channels = input_attrs[0].dims[1];
        input_height = input_attrs[0].dims[2];
        input_width = input_attrs[0].dims[3];
    } else { // NHWC
        input_height = input_attrs[0].dims[1];
        input_width = input_attrs[0].dims[2];
        input_channels = input_attrs[0].dims[3];
    }
    qDebug() << "    Detected Input Size: H=" << input_height << ", W=" << input_width << ", C=" << input_channels;

    qDebug() << "Querying output tensor attributes...";
    output_attrs = new rknn_tensor_attr[io_num.n_output];
    memset(output_attrs, 0, io_num.n_output * sizeof(rknn_tensor_attr));
    for (uint32_t i = 0; i < io_num.n_output; ++i) {
        output_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC) {
             qCritical() << "rknn_query(RKNN_QUERY_OUTPUT_ATTR) for output" << i << "error ret=" << ret;
             rknn_destroy(ctx); ctx=0; delete[] input_attrs; input_attrs = nullptr; delete[] output_attrs; output_attrs = nullptr;
             return;
        }
        qDebug() << "  Output[" << i << "]: name=" << output_attrs[i].name
                 << ", dims=[" << output_attrs[i].dims[0]
                 << "," << output_attrs[i].dims[1]
                 << "," << output_attrs[i].dims[2]
                 << "," << output_attrs[i].dims[3] << "]"
                 << ", type=" << output_attrs[i].type
                 << ", zp=" << output_attrs[i].zp
                 << ", scale=" << output_attrs[i].scale;
    }

    if (input_width <= 0 || input_height <= 0 || input_channels <= 0) {
        qCritical() << "Invalid model input dimensions.";
        rknn_destroy(ctx); ctx=0; delete[] input_attrs; input_attrs = nullptr; delete[] output_attrs; output_attrs = nullptr; return;
    }
    npu_input_buffer_size = input_width * input_height * input_channels;

    npu_input_buffer = malloc(npu_input_buffer_size);
    if (npu_input_buffer == nullptr) {
        qCritical() << "Failed to allocate NPU input buffer (size=" << npu_input_buffer_size << ")";
        rknn_destroy(ctx); ctx=0; delete[] input_attrs; input_attrs = nullptr; delete[] output_attrs; output_attrs = nullptr; return;
    }
    qDebug() << "NPU input buffer allocated (" << npu_input_buffer_size << "bytes) at" << npu_input_buffer;

    classNames = COCO_NAMES_NPU;

    isInitialized = true;
    qInfo() << "NPU Processor initialized SUCCESSFULLY with model:" << modelPath;
}

// --- 【实现这些槽函数】 ---
void NPUProcessor::setConfThreshold(float conf)
{
    m_confThreshold = conf;
    // 使用 qDebug 确认值已在正确线程中设置
    qDebug() << "[NPU-YOLOv5] CONF_THRESHOLD set to:" << m_confThreshold;
}

void NPUProcessor::setNmsThreshold(float nms)
{
    m_nmsThreshold = nms;
    qDebug() << "[NPU-YOLOv5] NMS_THRESHOLD set to:" << m_nmsThreshold;
}

void NPUProcessor::setAlertDuration(int seconds)
{
    m_personAlertDurationMs = seconds * 1000; // 将秒转换为毫秒
    qDebug() << "[NPU-YOLOv5] Alert Duration set to:" << m_personAlertDurationMs << "ms";
}
// --- 结束新实现 ---


// --- processFrame 函数 ---
void NPUProcessor::processFrame(const QImage& frame)
{
    if (m_is_busy.load(std::memory_order_acquire)) {
        return;
    }
    m_is_busy.store(true, std::memory_order_release);

    QElapsedTimer total_timer;
    total_timer.start();

    if (!isInitialized || frame.isNull() || ctx <= 0 || npu_input_buffer == nullptr) {
        emit processingFinished(QImage(), 0);
        m_is_busy.store(false, std::memory_order_release);
        return;
    }

    // --- RGA 硬件预处理 ---
    QElapsedTimer rga_timer;
    rga_timer.start();
    im_rect src_rect, dst_rect, fill_rect, prect;
    rga_buffer_t src_buf = {};
    rga_buffer_t dst_buf = {};
    rga_buffer_t pat = {};
    im_opt_t opt = {};
    IM_STATUS rga_ret;
    float r = std::min((float)input_width / frame.width(), (float)input_height / frame.height());
    int new_unpad_w = int(round(frame.width() * r));
    int new_unpad_h = int(round(frame.height() * r));
    int dw = (input_width - new_unpad_w) / 2;
    int dh = (input_height - new_unpad_h) / 2;
    src_buf.vir_addr = (void*)frame.bits();
    src_buf.width = frame.width();
    src_buf.height = frame.height();
    src_buf.wstride = frame.width();
    src_buf.hstride = frame.height();
    src_buf.format = RK_FORMAT_RGB_888;
    dst_buf.vir_addr = npu_input_buffer;
    dst_buf.width = input_width;
    dst_buf.height = input_height;
    dst_buf.wstride = input_width;
    dst_buf.hstride = input_height;
    dst_buf.format = RK_FORMAT_BGR_888;
    fill_rect = {0, 0, input_width, input_height};
    int gray_color = 0xFF727272; // 0xAABBGGRR (114,114,114)
    rga_ret = imfill(dst_buf, fill_rect, gray_color, 1, NULL);
    if (rga_ret != IM_STATUS_SUCCESS) {
        qWarning() << "RGA imfill failed with code" << rga_ret;
        emit processingFinished(QImage(), total_timer.elapsed());
        m_is_busy.store(false, std::memory_order_release);
        return;
    }
    src_rect = {0, 0, frame.width(), frame.height()};
    dst_rect = {dw, dh, new_unpad_w, new_unpad_h};
    prect = {};
    opt = {};
    rga_ret = improcess(src_buf, dst_buf, pat, src_rect, dst_rect, prect,
                        -1, NULL, &opt, IM_SYNC);
    if (rga_ret != IM_STATUS_SUCCESS) {
        qWarning() << "RGA improcess failed with code" << rga_ret;
        emit processingFinished(QImage(), total_timer.elapsed());
        m_is_busy.store(false, std::memory_order_release);
        return;
    }
    QElapsedTimer timer;
    timer.start();

    // --- 设置 RKNN 输入  ---
    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].buf = npu_input_buffer;
    inputs[0].size = npu_input_buffer_size;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].pass_through = 0;

    int ret = rknn_inputs_set(ctx, io_num.n_input, inputs);
    if (ret < 0) {
        qCritical() << "rknn_inputs_set error ret=" << ret;
        emit processingFinished(QImage(), timer.elapsed());
        m_is_busy.store(false, std::memory_order_release);
        return;
    }

    // --- 执行 NPU 推理  ---
    ret = rknn_run(ctx, nullptr);
    if (ret < 0) {
        qCritical() << "rknn_run error ret=" << ret;
        emit processingFinished(QImage(), timer.elapsed());
        m_is_busy.store(false, std::memory_order_release);
        return;
    }

    // --- 获取 NPU 输出  ---
    memset(outputs, 0, sizeof(outputs));
    for (uint32_t i = 0; i < io_num.n_output; ++i) {
        outputs[i].want_float = 0;
    }
    ret = rknn_outputs_get(ctx, io_num.n_output, outputs, nullptr);
    if (ret < 0) {
        qCritical() << "rknn_outputs_get error ret=" << ret;
        emit processingFinished(QImage(), timer.elapsed());
        m_is_busy.store(false, std::memory_order_release);
        return;
    }

    cv::Mat mat_for_drawing(input_height, input_width, CV_8UC3, npu_input_buffer);

    // --- 调用后处理  ---
    std::vector<cv::Rect_<float>> final_detections;
    std::vector<int> final_classIds;
    std::vector<float> final_confidences;
    postprocess_yolov5_3_output(outputs,
                                cv::Size(frame.width(), frame.height()),
                                final_detections, final_classIds, final_confidences);

    ret = rknn_outputs_release(ctx, io_num.n_output, outputs);
    if (ret < 0) { qWarning() << "rknn_outputs_release error ret=" << ret; }

    // --- 更新跟踪器 ---
    updateTracker(final_detections, final_classIds, final_confidences);

    // --- 【绘制所有物体，但只对 person 报警】 ---
    bool alert_triggered_this_frame = false;
    int current_person_count = 0; // 这个变量现在只用于左上角的计数

    for (const auto& obj : tracked_objects) {
        if (obj.frames_since_seen == 0) { // 只绘制当前帧看到的

            // 缩放回 640x640 绘制画布
            cv::Rect_<float> draw_rect;
            draw_rect.x = obj.rect.x * r + dw;
            draw_rect.y = obj.rect.y * r + dh;
            draw_rect.width = obj.rect.width * r;
            draw_rect.height = obj.rect.height * r;

            // --- 【根据 classId 设置不同颜色】 ---
            // 'person' (ID 0) 始终为绿色，其他物体用 ID 区分颜色
            cv::Scalar color = (obj.classId == 0) ?
                                cv::Scalar(0, 255, 0) :
                                cv::Scalar((obj.classId * 30) % 255, (obj.classId * 70) % 255, (obj.classId * 110) % 255);

            cv::rectangle(mat_for_drawing, draw_rect, color, 2);

            // 计数器只统计 person
            if (obj.classId == 0) {
                current_person_count++;
            }

            // --- 【显示所有类别的标签】 ---
            std::stringstream ss_label;
            if (obj.classId >= 0 && obj.classId < classNames.size()) {
                 ss_label << classNames[obj.classId] << " ";
            }
            ss_label << "ID:" << obj.id;
            ss_label << " " << std::fixed << std::setprecision(2) << obj.confidence;
            std::string label = ss_label.str();
            cv::putText(mat_for_drawing, label, cv::Point(draw_rect.x, draw_rect.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);


            // --- 【报警逻辑只针对 person】 ---
            // 这部分逻辑只在检测到 'person' 时才执行
            if (obj.classId == 0) {
                // 使用 m_personAlertDurationMs
                if (obj.appearanceTimer.isValid() && obj.appearanceTimer.elapsed() >= m_personAlertDurationMs && !obj.alertSentForThisAppearance) {
                    if (!alert_triggered_this_frame && (!lastAlertTime.isValid() || lastAlertTime.elapsed() > alertCooldownMs)) {
                        qDebug() << "NPU: Person ID" << obj.id << " visible >= 5s. Cooldown passed. Sending alert.";
                        cv::Mat mat_rgb_alert;
                        cv::cvtColor(mat_for_drawing, mat_rgb_alert, cv::COLOR_BGR2RGB);
                        QImage alertImage(mat_rgb_alert.data, mat_rgb_alert.cols, mat_rgb_alert.rows, mat_rgb_alert.step, QImage::Format_RGB888);
                        emit alertNeeded(alertImage.copy());
                        lastAlertTime.restart();
                        alert_triggered_this_frame = true;
                        for (auto& tracked_obj : tracked_objects) { if (tracked_obj.id == obj.id) { tracked_obj.alertSentForThisAppearance = true; break; } }
                    }
                }
            }
        }
    }
    // 左上角计数器只显示 person
    cv::putText(mat_for_drawing, "Persons: " + std::to_string(current_person_count), cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);

    // --- cv::Mat -> QImage & 发送结果  ---
    cv::cvtColor(mat_for_drawing, mat_for_drawing, cv::COLOR_BGR2RGB);
    QImage resultImage(mat_for_drawing.data, mat_for_drawing.cols, mat_for_drawing.rows, mat_for_drawing.step, QImage::Format_RGB888);

    emit processingFinished(resultImage.copy(), total_timer.elapsed());

    m_is_busy.store(false, std::memory_order_release);
}

// --- 【NCHW + INT8 高性能后处理函数】 ---
int NPUProcessor::postprocess_yolov5_3_output(rknn_output *all_outputs,
                                           const cv::Size& original_img_size,
                                           std::vector<cv::Rect_<float>>& detections,
                                           std::vector<int>& classIds,
                                           std::vector<float>& confidences)
{
    detections.clear();
    classIds.clear();
    confidences.clear();

    std::vector<cv::Rect> boxes_nms;
    std::vector<float> confs_nms;
    std::vector<int> classIds_nms;

    const int elements_per_det = NUM_CLASSES_NPU + 5; // 85
    const int num_anchors = 3;

    float r = std::min((float)input_width / original_img_size.width, (float)input_height / original_img_size.height);
    int pad_w = (int)((input_width - original_img_size.width * r) / 2);
    int pad_h = (int)((input_height - original_img_size.height * r) / 2);
    float scale_w = (float)original_img_size.width / (input_width - 2 * pad_w);
    float scale_h = (float)original_img_size.height / (input_height - 2 * pad_h);

    int32_t zp_0 = output_attrs[0].zp;
    float scale_0 = output_attrs[0].scale;
    // 使用 m_confThreshold
    int8_t thres_i8 = qnt_f32_to_affine_i8(m_confThreshold, zp_0, scale_0);

    // --- 循环处理 3 个输出 ---
    for (int output_idx = 0; output_idx < 3; ++output_idx)
    {
        int8_t* data = (int8_t*)all_outputs[output_idx].buf;
        int stride = m_strides[output_idx];
        const std::vector<cv::Size>& anchors = m_anchor_grid[output_idx];

        int32_t zp = output_attrs[output_idx].zp;
        float scale = output_attrs[output_idx].scale;

        if (output_attrs[output_idx].dims[1] != (num_anchors * elements_per_det)) {
             qWarning() << "NPU postprocess: Output" << output_idx << "has unexpected channel count. Expected"
                        << (num_anchors * elements_per_det) << "but got" << output_attrs[output_idx].dims[1];
             continue;
        }

        int grid_h = output_attrs[output_idx].dims[2];
        int grid_w = output_attrs[output_idx].dims[3];
        int grid_plane_size = grid_h * grid_w;

        // 循环网格
        for (int h = 0; h < grid_h; ++h) {
            for (int w = 0; w < grid_w; ++w) {
                for (int a = 0; a < num_anchors; ++a) {

                    const int8_t* proposal_base = data + (a * elements_per_det) * grid_plane_size;
                    const int8_t* proposal = proposal_base + (h * grid_w + w);

                    // 2. 解码 obj_conf (int8)
                    int8_t obj_conf_i8 = proposal[4 * grid_plane_size];

                    // 3.  检查“物体置信度”，如果连物体都不是，就跳过
                    if (obj_conf_i8 < thres_i8) {
                        continue;
                    }

                    // 4. 【遍历所有80个类别，找到分数最高的那个】
                    int8_t* class_scores = (int8_t*)(proposal + 5 * grid_plane_size);
                    int8_t max_class_score_i8 = -128;
                    int best_class_id = -1;

                    for (int c = 0; c < NUM_CLASSES_NPU; ++c) {
                        int8_t current_score_i8 = class_scores[c * grid_plane_size];
                        if (current_score_i8 > max_class_score_i8) {
                            max_class_score_i8 = current_score_i8;
                            best_class_id = c;
                        }
                    }

                    // 5. 【使用找到的最佳类别计算最终置信度】
                    float obj_conf = deqnt_affine_i8_to_f32(obj_conf_i8, zp, scale);
                    float max_class_prob = deqnt_affine_i8_to_f32(max_class_score_i8, zp, scale);
                    float final_conf = obj_conf * max_class_prob;

                    // 6. 使用 m_confThreshold 过滤
                    if (final_conf < m_confThreshold) {
                        continue;
                    }

                    // 7. 解码 [tx, ty, tw, th]
                    float tx = deqnt_affine_i8_to_f32(proposal[0 * grid_plane_size], zp, scale);
                    float ty = deqnt_affine_i8_to_f32(proposal[1 * grid_plane_size], zp, scale);
                    float tw = deqnt_affine_i8_to_f32(proposal[2 * grid_plane_size], zp, scale);
                    float th = deqnt_affine_i8_to_f32(proposal[3 * grid_plane_size], zp, scale);

                    // 8. 将 [tx, ty, tw, th] 转换回 [cx, cy, w, h]
                    float cx = (sigmoid(tx) * 2.0f - 0.5f + w) * stride;
                    float cy = (sigmoid(ty) * 2.0f - 0.5f + h) * stride;
                    float out_w = pow(sigmoid(tw) * 2.0f, 2) * anchors[a].width;
                    float out_h = pow(sigmoid(th) * 2.0f, 2) * anchors[a].height;

                    // 9. 转换为 [x1, y1, w, h]
                    float x1 = cx - out_w * 0.5f;
                    float y1 = cy - out_h * 0.5f;

                    // 10. 缩放回原始图像尺寸
                    float scaled_x1 = (x1 - pad_w) * scale_w;
                    float scaled_y1 = (y1 - pad_h) * scale_h;
                    float scaled_w = out_w * scale_w;
                    float scaled_h = out_h * scale_h;

                    // 11. 存储用于 NMS
                    boxes_nms.push_back(cv::Rect(static_cast<int>(scaled_x1),
                                                 static_cast<int>(scaled_y1),
                                                 static_cast<int>(scaled_w),
                                                 static_cast<int>(scaled_h)));
                    confs_nms.push_back(final_conf);
                    classIds_nms.push_back(best_class_id); // <-- 使用找到的最佳 ID
                }
            }
        }
    } // 结束 3 个输出的循环

    std::vector<int> nms_indices;
    // 使用 m_confThreshold 和 m_nmsThreshold
    cv::dnn::NMSBoxes(boxes_nms, confs_nms, m_confThreshold, m_nmsThreshold, nms_indices);

    for (int index : nms_indices) {
        detections.push_back(cv::Rect_<float>(boxes_nms[index]));
        classIds.push_back(classIds_nms[index]);
        confidences.push_back(confs_nms[index]);
    }

    return detections.size();
}

// --- calculate_iou 和 updateTracker 函數 (保持不变) ---
float NPUProcessor::calculate_iou(const cv::Rect_<float>& rect1, const cv::Rect_<float>& rect2) {
    float x_left = std::max(rect1.x, rect2.x);
    float y_top = std::max(rect1.y, rect2.y);
    float x_right = std::min(rect1.x + rect1.width, rect2.x + rect2.width);
    float y_bottom = std::min(rect1.y + rect1.height, rect2.y + rect2.height);
    if (x_right < x_left || y_bottom < y_top) return 0.0f;
    float intersection_area = (x_right - x_left) * (y_bottom - y_top);
    float rect1_area = rect1.width * rect1.height;
    float rect2_area = rect2.width * rect2.height;
    float union_area = rect1_area + rect2_area - intersection_area;
    return union_area > 0 ? intersection_area / union_area : 0.0f;
}

void NPUProcessor::updateTracker(const std::vector<cv::Rect_<float>>& detections,
                                 const std::vector<int>& classIds,
                                 const std::vector<float>& confidences) {
    const float iou_threshold = 0.3f;
    for (auto& obj : tracked_objects) {
        obj.frames_since_seen++;
        if (obj.frames_since_seen > 1) {
            if (obj.appearanceTimer.isValid()) obj.appearanceTimer.invalidate();
            obj.alertSentForThisAppearance = false;
        }
    }
    std::vector<bool> matched_detections(detections.size(), false);
    for (size_t i = 0; i < tracked_objects.size(); ++i) {
        if (tracked_objects[i].frames_since_seen > TrackedObject::max_frames_to_forget) continue;
        float best_iou = 0.0f; int best_match_idx = -1;
        for (size_t j = 0; j < detections.size(); ++j) {

            // --- 【跟踪所有类别】 ---
            // 只要类别 ID 相同，就尝试匹配
            if (!matched_detections[j] && tracked_objects[i].classId == classIds[j]) {
                float iou = calculate_iou(tracked_objects[i].rect, detections[j]);
                if (iou > iou_threshold && iou > best_iou) { best_iou = iou; best_match_idx = j; }
            }
        }
        if (best_match_idx != -1) {
            tracked_objects[i].rect = detections[best_match_idx];
            tracked_objects[i].frames_since_seen = 0;
            tracked_objects[i].confidence = confidences[best_match_idx];
            matched_detections[best_match_idx] = true;
            if (!tracked_objects[i].appearanceTimer.isValid()) {
                tracked_objects[i].appearanceTimer.start();
                tracked_objects[i].alertSentForThisAppearance = false;
            }
        }
    }
    for (size_t j = 0; j < detections.size(); ++j) {

        // --- 【添加所有未匹配上的新物体】 ---
        if (!matched_detections[j]) {
            TrackedObject new_obj;
            new_obj.id = next_track_id++;
            new_obj.rect = detections[j];
            new_obj.classId = classIds[j];
            new_obj.confidence = confidences[j];
            new_obj.frames_since_seen = 0;
            new_obj.appearanceTimer.start();
            new_obj.alertSentForThisAppearance = false;
            tracked_objects.push_back(new_obj);
        }
    }
    tracked_objects.erase(std::remove_if(tracked_objects.begin(), tracked_objects.end(),
        [](const TrackedObject& obj){ return obj.frames_since_seen > TrackedObject::max_frames_to_forget; }),
        tracked_objects.end());
}
