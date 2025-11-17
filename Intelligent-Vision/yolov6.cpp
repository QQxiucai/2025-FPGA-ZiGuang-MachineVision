// =======================================================
// 文件名: yolov6.cpp
// =======================================================
#include "yolov6.h"
#include <QDebug>
#include <QThread>
#include <QPainter>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <sstream>
#include <iomanip>
#include <set> // 用于 NMS

// --- COCO 类别名称 (80类) - (与 NPUProcessor 相同) ---
const std::vector<std::string> COCO_NAMES_V6 = {
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

// --- 构造函数 ---
YOLOv6Processor::YOLOv6Processor(QObject *parent)
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
      m_confThreshold(0.5f),   // <-- 【【新：初始化默认值】】
      m_nmsThreshold(0.35f),  // <-- 【【新：初始化默认值】】
      m_personAlertDurationMs(5000) // <-- 【【新：初始化默认值】】
{
    lastAlertTime.start();
    lastAlertTime.invalidate();

    // 【新】加载 COCO 标签
    classNames = COCO_NAMES_V6;

    // 【修改】使用 m_personAlertDurationMs
    qDebug() << "[YOLOv6] Processor created. Alert cooldown:" << alertCooldownMs << "ms. Person duration threshold:" << m_personAlertDurationMs << "ms";
}

// --- 析构函数 (与 NPUProcessor 相同) ---
YOLOv6Processor::~YOLOv6Processor()
{
    qDebug() << "[YOLOv6] Processor destroying...";
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
    qDebug() << "[YOLOv6] Processor destroyed.";
}

// --- 辅助函数：加载模型 (与 NPUProcessor 相同) ---
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

// --- init 函数 (适配 YOLOv6) ---
void YOLOv6Processor::init(const QString& modelPath)
{
    qDebug() << "[YOLOv6] Processor initializing in thread:" << QThread::currentThreadId();
    isInitialized = false;

    model_data = load_model(modelPath.toStdString().c_str(), &model_data_size);
    if (model_data == nullptr) { qCritical() << "[YOLOv6] Failed to load RKNN model from" << modelPath; return; }
    qDebug() << "[YOLOv6] RKNN model loaded, size:" << model_data_size << "bytes";

    int ret = rknn_init(&ctx, model_data, model_data_size, 0, NULL);
    if (ret < 0) { qCritical() << "[YOLOv6] rknn_init error ret=" << ret; free(model_data); model_data = nullptr; return; }
    qDebug() << "[YOLOv6] RKNN context initialized.";

    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != RKNN_SUCC) { qCritical() << "[YOLOv6] rknn_query(RKNN_QUERY_IN_OUT_NUM) error ret=" << ret; rknn_destroy(ctx); ctx=0; return; }
    qDebug() << "[YOLOv6] Model info: inputs=" << io_num.n_input << ", outputs=" << io_num.n_output;

    if (io_num.n_input != 1) { qCritical() << "[YOLOv6] Model requires 1 input, but got" << io_num.n_input; rknn_destroy(ctx); ctx=0; return; }

    // 【修改】YOLOv6 有多个输出 (通常是 6 或 9)
    if (io_num.n_output % 3 != 0) {
        qWarning() << "[YOLOv6] WARNING: Model output number is" << io_num.n_output << "which is not a multiple of 3. (Expected 6 or 9)";
    }
    if (io_num.n_output > 9) {
         qCritical() << "[YOLOv6] Model has too many outputs (" << io_num.n_output << "). Max supported is 9."; rknn_destroy(ctx); ctx=0; return;
    }


    qDebug() << "[YOLOv6] Querying input tensor attributes...";
    input_attrs = new rknn_tensor_attr[io_num.n_input];
    memset(input_attrs, 0, io_num.n_input * sizeof(rknn_tensor_attr));
    input_attrs[0].index = 0;
    ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[0]), sizeof(rknn_tensor_attr));
    if (ret != RKNN_SUCC) {
        qCritical() << "[YOLOv6] rknn_query(RKNN_QUERY_INPUT_ATTR) error ret=" << ret;
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

    qDebug() << "[YOLOv6] Querying output tensor attributes...";
    output_attrs = new rknn_tensor_attr[io_num.n_output];
    memset(output_attrs, 0, io_num.n_output * sizeof(rknn_tensor_attr));
    for (uint32_t i = 0; i < io_num.n_output; ++i) {
        output_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC) {
             qCritical() << "[YOLOv6] rknn_query(RKNN_QUERY_OUTPUT_ATTR) for output" << i << "error ret=" << ret;
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
        qCritical() << "[YOLOv6] Invalid model input dimensions.";
        rknn_destroy(ctx); ctx=0; delete[] input_attrs; input_attrs = nullptr; delete[] output_attrs; output_attrs = nullptr; return;
    }
    npu_input_buffer_size = input_width * input_height * input_channels;

    npu_input_buffer = malloc(npu_input_buffer_size);
    if (npu_input_buffer == nullptr) {
        qCritical() << "[YOLOv6] Failed to allocate NPU input buffer (size=" << npu_input_buffer_size << ")";
        rknn_destroy(ctx); ctx=0; delete[] input_attrs; input_attrs = nullptr; delete[] output_attrs; output_attrs = nullptr; return;
    }
    qDebug() << "[YOLOv6] NPU input buffer allocated (" << npu_input_buffer_size << "bytes) at" << npu_input_buffer;

    isInitialized = true;
    qInfo() << "[YOLOv6] Processor initialized SUCCESSFULLY with model:" << modelPath;
}

// --- 【【【【新：实现这些槽函数】】】】 ---
void YOLOv6Processor::setConfThreshold(float conf)
{
    m_confThreshold = conf;
    qDebug() << "[YOLOv6] CONF_THRESHOLD set to:" << m_confThreshold;
}

void YOLOv6Processor::setNmsThreshold(float nms)
{
    m_nmsThreshold = nms;
    qDebug() << "[YOLOv6] NMS_THRESHOLD set to:" << m_nmsThreshold;
}

void YOLOv6Processor::setAlertDuration(int seconds)
{
    m_personAlertDurationMs = seconds * 1000; // 将秒转换为毫秒
    qDebug() << "[YOLOv6] Alert Duration set to:" << m_personAlertDurationMs << "ms";
}
// --- 结束新实现 ---


// --- processFrame 函数 (与 NPUProcessor 新版几乎相同) ---
void YOLOv6Processor::processFrame(const QImage& frame)
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

    // --- RGA 硬件预处理 (不变) ---
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
        qWarning() << "[YOLOv6] RGA imfill failed with code" << rga_ret;
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
        qWarning() << "[YOLOv6] RGA improcess failed with code" << rga_ret;
        emit processingFinished(QImage(), total_timer.elapsed());
        m_is_busy.store(false, std::memory_order_release);
        return;
    }
    QElapsedTimer timer;
    timer.start();

    // --- 设置 RKNN 输入 (不变) ---
    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].buf = npu_input_buffer;
    inputs[0].size = npu_input_buffer_size;
    inputs[0].type = RKNN_TENSOR_UINT8; // 假设模型输入为 UINT8
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].pass_through = 0;
    int ret = rknn_inputs_set(ctx, io_num.n_input, inputs);
    if (ret < 0) {
        qCritical() << "[YOLOv6] rknn_inputs_set error ret=" << ret;
        emit processingFinished(QImage(), timer.elapsed());
        m_is_busy.store(false, std::memory_order_release);
        return;
    }

    // --- 执行 NPU 推理 (不变) ---
    ret = rknn_run(ctx, nullptr);
    if (ret < 0) {
        qCritical() << "[YOLOv6] rknn_run error ret=" << ret;
        emit processingFinished(QImage(), timer.elapsed());
        m_is_busy.store(false, std::memory_order_release);
        return;
    }

    // --- 获取 NPU 输出 (不变) ---
    memset(outputs, 0, sizeof(outputs));
    for (uint32_t i = 0; i < io_num.n_output; ++i) {
        outputs[i].want_float = 0; // 请求原始 int8/uint8
    }
    ret = rknn_outputs_get(ctx, io_num.n_output, outputs, nullptr);
    if (ret < 0) {
        qCritical() << "[YOLOv6] rknn_outputs_get error ret=" << ret;
        emit processingFinished(QImage(), timer.elapsed());
        m_is_busy.store(false, std::memory_order_release);
        return;
    }

    cv::Mat mat_for_drawing(input_height, input_width, CV_8UC3, npu_input_buffer);

    // --- 【【【【关键调用】】】】 ---
    // --- 调用新的 YOLOv6 后处理函数 ---
    std::vector<cv::Rect_<float>> final_detections;
    std::vector<int> final_classIds;
    std::vector<float> final_confidences;

    postprocess_yolov6(outputs,
                       cv::Size(frame.width(), frame.height()),
                       final_detections, final_classIds, final_confidences);

    ret = rknn_outputs_release(ctx, io_num.n_output, outputs);
    if (ret < 0) { qWarning() << "[YOLOv6] rknn_outputs_release error ret=" << ret; }

    // --- 更新跟踪器 (不变) ---
    updateTracker(final_detections, final_classIds, final_confidences);

    // --- 绘制结果 (与 NPUProcessor 新版相同) ---
    bool alert_triggered_this_frame = false;
    int current_person_count = 0;

    for (const auto& obj : tracked_objects) {
        if (obj.frames_since_seen == 0) {

            // 缩放回 640x640 绘制画布
            cv::Rect_<float> draw_rect;
            draw_rect.x = obj.rect.x * r + dw;
            draw_rect.y = obj.rect.y * r + dh;
            draw_rect.width = obj.rect.width * r;
            draw_rect.height = obj.rect.height * r;

            // 'person' (ID 0) 始终为绿色，其他物体用 ID 区分颜色
            cv::Scalar color = (obj.classId == 0) ?
                                cv::Scalar(0, 255, 0) :
                                cv::Scalar((obj.classId * 30) % 255, (obj.classId * 70) % 255, (obj.classId * 110) % 255);

            cv::rectangle(mat_for_drawing, draw_rect, color, 2);

            if (obj.classId == 0) {
                current_person_count++;
            }

            std::stringstream ss_label;
            if (obj.classId >= 0 && obj.classId < classNames.size()) {
                 ss_label << classNames[obj.classId] << " ";
            }
            ss_label << "ID:" << obj.id;
            ss_label << " " << std::fixed << std::setprecision(2) << obj.confidence;
            std::string label = ss_label.str();
            cv::putText(mat_for_drawing, label, cv::Point(draw_rect.x, draw_rect.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);


            // --- 【保留：报警逻辑只针对 person】 ---
            if (obj.classId == 0) {
                // 【修改】使用 m_personAlertDurationMs
                if (obj.appearanceTimer.isValid() && obj.appearanceTimer.elapsed() >= m_personAlertDurationMs && !obj.alertSentForThisAppearance) {
                    if (!alert_triggered_this_frame && (!lastAlertTime.isValid() || lastAlertTime.elapsed() > alertCooldownMs)) {
                        qDebug() << "[YOLOv6] Person ID" << obj.id << " visible >= 5s. Cooldown passed. Sending alert.";
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
    cv::putText(mat_for_drawing, "Persons: " + std::to_string(current_person_count), cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);

    // --- cv::Mat -> QImage & 发送结果 (不变) ---
    cv::cvtColor(mat_for_drawing, mat_for_drawing, cv::COLOR_BGR2RGB);
    QImage resultImage(mat_for_drawing.data, mat_for_drawing.cols, mat_for_drawing.rows, mat_for_drawing.step, QImage::Format_RGB888);

    emit processingFinished(resultImage.copy(), total_timer.elapsed());

    m_is_busy.store(false, std::memory_order_release);
}


// ==============================================================================
// --- 【【【【【【【【 以下为从 postprocess.cc 完整移植的 YOLOv6 逻辑 】】】】】】】】 ---
// ==============================================================================

// --- 移植的辅助函数 (转为类方法) ---
inline int32_t YOLOv6Processor::__clip(float val, float min, float max)
{
    float f = val <= min ? min : (val >= max ? max : val);
    return f;
}

int8_t YOLOv6Processor::qnt_f32_to_affine(float f32, int32_t zp, float scale)
{
    float dst_val = (f32 / scale) + zp;
    int8_t res = (int8_t)__clip(dst_val, -128, 127);
    return res;
}

float YOLOv6Processor::deqnt_affine_to_f32(int8_t qnt, int32_t zp, float scale)
{
    return ((float)qnt - (float)zp) * scale;
}

float YOLOv6Processor::sigmoid(float x)
{
    return 1.0 / (1.0 + expf(-x));
}

// --- 【核心】YOLOv6 DFL 解码函数 ---
void YOLOv6Processor::compute_dfl(float* tensor, int dfl_len, float* box)
{
    for (int b = 0; b < 4; b++) {
        float exp_t[dfl_len];
        float exp_sum = 0;
        float acc_sum = 0;
        for (int i = 0; i < dfl_len; i++) {
            exp_t[i] = exp(tensor[i + b * dfl_len]);
            exp_sum += exp_t[i];
        }

        for (int i = 0; i < dfl_len; i++) {
            acc_sum += exp_t[i] / exp_sum * i;
        }
        box[b] = acc_sum;
    }
}

// --- 【核心】YOLOv6 NCHW(i8) 解码函数 ---
// 适配 RKNPU2 (RK3568/3588) 的 int8 NCHW 布局
int YOLOv6Processor::process_i8(int8_t *box_tensor, int32_t box_zp, float box_scale,
                                int8_t *score_tensor, int32_t score_zp, float score_scale,
                                int8_t *score_sum_tensor, int32_t score_sum_zp, float score_sum_scale,
                                int grid_h, int grid_w, int stride, int dfl_len,
                                std::vector<float> &boxes,
                                std::vector<float> &objProbs,
                                std::vector<int> &classId,
                                float threshold)
{
    int validCount = 0;
    int grid_len = grid_h * grid_w;
    int8_t score_thres_i8 = qnt_f32_to_affine(threshold, score_zp, score_scale);
    int8_t score_sum_thres_i8 = (score_sum_tensor != nullptr) ?
                                qnt_f32_to_affine(threshold, score_sum_zp, score_sum_scale) : 0;

    for (int i = 0; i < grid_h; i++)
    {
        for (int j = 0; j < grid_w; j++)
        {
            int offset = i * grid_w + j;
            int max_class_id = -1;

            // 通过 score sum 起到快速过滤的作用
            if (score_sum_tensor != nullptr) {
                if (score_sum_tensor[offset] < score_sum_thres_i8) {
                    continue;
                }
            }

            int8_t max_score = score_thres_i8; //【修改】使用阈值作为初始值
            int score_offset = offset;
            for (int c = 0; c < OBJ_CLASS_NUM; c++) {
                int8_t score = score_tensor[score_offset];
                if (score > max_score) // 【修改】移除多余的 > score_thres_i8 检查
                {
                    max_score = score;
                    max_class_id = c;
                }
                score_offset += grid_len; // NCHW 布局
            }

            // compute box
            if (max_class_id != -1) { // 【修改】检查 max_class_id
                int box_offset = offset;
                float box[4];
                if (dfl_len > 1) {
                    /// dfl
                    float before_dfl[dfl_len * 4];
                    for (int k = 0; k < dfl_len * 4; k++) {
                        before_dfl[k] = deqnt_affine_to_f32(box_tensor[box_offset], box_zp, box_scale);
                        box_offset += grid_len; // NCHW 布局
                    }
                    compute_dfl(before_dfl, dfl_len, box);
                } else {
                    for (int k = 0; k < 4; k++) {
                        box[k] = deqnt_affine_to_f32(box_tensor[box_offset], box_zp, box_scale);
                        box_offset += grid_len; // NCHW 布局
                    }
                }

                // --- 【YOLOv6 解码】 ---
                float x1, y1, x2, y2, w, h;
                x1 = (-box[0] + j + 0.5) * stride;
                y1 = (-box[1] + i + 0.5) * stride;
                x2 = (box[2] + j + 0.5) * stride;
                y2 = (box[3] + i + 0.5) * stride;
                w = x2 - x1;
                h = y2 - y1;
                // --- 结束解码 ---

                boxes.push_back(x1);
                boxes.push_back(y1);
                boxes.push_back(w);
                boxes.push_back(h);

                objProbs.push_back(deqnt_affine_to_f32(max_score, score_zp, score_scale));
                classId.push_back(max_class_id);
                validCount++;
            }
        }
    }
    //qDebug() << "[YOLOv6] validCount=" << validCount << ", grid=" << grid_h << ", stride=" << stride;
    return validCount;
}

// --- 【核心】YOLOv6 主后处理函数 (移植自 post_process) ---
int YOLOv6Processor::postprocess_yolov6(rknn_output *all_outputs,
                                      const cv::Size& original_img_size,
                                      std::vector<cv::Rect_<float>>& detections,
                                      std::vector<int>& classIds,
                                      std::vector<float>& confidences)
{
    // 清空上一帧的结果
    detections.clear();
    classIds.clear();
    confidences.clear();

    // 存储 NMS 前的原始框体
    std::vector<float> filterBoxes;
    std::vector<float> objProbs;
    std::vector<int> classId_vec; // 避免与类成员 classIds 冲突
    int validCount = 0;
    int stride = 0;
    int grid_h = 0;
    int grid_w = 0;

    // --- 【新】计算缩放参数 (用于最后反算) ---
    float r = std::min((float)input_width / original_img_size.width, (float)input_height / original_img_size.height);
    int pad_w = (int)((input_width - original_img_size.width * r) / 2);
    int pad_h = (int)((input_height - original_img_size.height * r) / 2);
    float scale_w = (float)original_img_size.width / (input_width - 2 * pad_w);
    float scale_h = (float)original_img_size.height / (input_height - 2 * pad_h);
    // --- 结束 ---

    // 假设有 3 个分支 (strides 8, 16, 32)
    int output_per_branch = io_num.n_output / 3;
    if (io_num.n_output % 3 != 0) {
        qWarning() << "[YOLOv6] Postprocess: Output count" << io_num.n_output << "is not multiple of 3.";
        return -1;
    }

    for (int i = 0; i < 3; i++)
    {
        void *score_sum = nullptr;
        int32_t score_sum_zp = 0;
        float score_sum_scale = 1.0;
        if (output_per_branch == 3) {
            score_sum = all_outputs[i * output_per_branch + 2].buf;
            score_sum_zp = output_attrs[i * output_per_branch + 2].zp;
            score_sum_scale = output_attrs[i * output_per_branch + 2].scale;
        }
        int box_idx = i * output_per_branch;
        int score_idx = i * output_per_branch + 1;

        // 检查数据类型 (假设 RKNPU2/RK3568 使用 INT8)
        if (output_attrs[box_idx].type != RKNN_TENSOR_INT8 || output_attrs[score_idx].type != RKNN_TENSOR_INT8) {
             qWarning() << "[YOLOv6] Postprocess: Output type is not INT8. (type=" << output_attrs[box_idx].type << ")";
             // 您可以在这里添加对 process_u8 或 process_fp32 的调用
             return -1;
        }

        // RKNPU2 总是 NCHW
        grid_h = output_attrs[box_idx].dims[2];
        grid_w = output_attrs[box_idx].dims[3];
        stride = input_height / grid_h;

        // DFL 长度
        int dfl_len = output_attrs[box_idx].dims[1] / 4;

        // 【修改】使用 m_confThreshold 替换 CONF_THRESH
        validCount += process_i8((int8_t *)all_outputs[box_idx].buf, output_attrs[box_idx].zp, output_attrs[box_idx].scale,
                                   (int8_t *)all_outputs[score_idx].buf, output_attrs[score_idx].zp, output_attrs[score_idx].scale,
                                   (int8_t *)score_sum, score_sum_zp, score_sum_scale,
                                   grid_h, grid_w, stride, dfl_len,
                                   filterBoxes, objProbs, classId_vec, m_confThreshold);
    }

    // no object detect
    if (validCount <= 0) {
        return 0;
    }

    std::vector<int> indexArray;
    for (int i = 0; i < validCount; ++i) {
        indexArray.push_back(i);
    }
    quick_sort_indice_inverse(objProbs, 0, validCount - 1, indexArray);

    std::set<int> class_set(std::begin(classId_vec), std::end(classId_vec));

    for (auto c : class_set) {
        // 【修改】使用 m_nmsThreshold 替换 NMS_THRESH
        nms(validCount, filterBoxes, classId_vec, indexArray, c, m_nmsThreshold);
    }

    int last_count = 0;

    /* box valid detect target */
    for (int i = 0; i < validCount; ++i)
    {
        if (indexArray[i] == -1 || last_count >= OBJ_NUMB_MAX_SIZE) {
            continue;
        }
        int n = indexArray[i];

        // 框体是 640x640 坐标系下的 [x1, y1, w, h]
        float x1_raw = filterBoxes[n * 4 + 0];
        float y1_raw = filterBoxes[n * 4 + 1];
        float w_raw = filterBoxes[n * 4 + 2];
        float h_raw = filterBoxes[n * 4 + 3];

        // --- 【新】反算回原始图像坐标系 ---
        float scaled_x1 = (x1_raw - pad_w) * scale_w;
        float scaled_y1 = (y1_raw - pad_h) * scale_h;
        float scaled_w = w_raw * scale_w;
        float scaled_h = h_raw * scale_h;

        // --- 存储到最终的输出 vector ---
        detections.push_back(cv::Rect_<float>(
            std::max(0.0f, scaled_x1),
            std::max(0.0f, scaled_y1),
            std::min((float)original_img_size.width - scaled_x1, scaled_w),
            std::min((float)original_img_size.height - scaled_y1, scaled_h)
        ));
        classIds.push_back(classId_vec[n]);
        confidences.push_back(objProbs[i]); // 注意：objProbs 已经排序

        last_count++;
    }
    return last_count;
}


// --- 【核心】NMS 辅助函数 (移植自 postprocess.cc) ---
float YOLOv6Processor::CalculateOverlap(float xmin0, float ymin0, float xmax0, float ymax0, float xmin1, float ymin1, float xmax1,
                                  float ymax1)
{
    float w = fmax(0.f, fmin(xmax0, xmax1) - fmax(xmin0, xmin1) + 1.0);
    float h = fmax(0.f, fmin(ymax0, ymax1) - fmax(ymin0, ymin1) + 1.0);
    float i = w * h;
    float u = (xmax0 - xmin0 + 1.0) * (ymax0 - ymin0 + 1.0) + (xmax1 - xmin1 + 1.0) * (ymax1 - ymin1 + 1.0) - i;
    return u <= 0.f ? 0.f : (i / u);
}

int YOLOv6Processor::nms(int validCount, std::vector<float> &outputLocations, std::vector<int> classIds, std::vector<int> &order,
                   int filterId, float threshold)
{
    for (int i = 0; i < validCount; ++i)
    {
        int n = order[i];
        if (n == -1 || classIds[n] != filterId) {
            continue;
        }
        for (int j = i + 1; j < validCount; ++j)
        {
            int m = order[j];
            if (m == -1 || classIds[m] != filterId) {
                continue;
            }
            // [x1, y1, w, h]
            float xmin0 = outputLocations[n * 4 + 0];
            float ymin0 = outputLocations[n * 4 + 1];
            float xmax0 = outputLocations[n * 4 + 0] + outputLocations[n * 4 + 2];
            float ymax0 = outputLocations[n * 4 + 1] + outputLocations[n * 4 + 3];

            float xmin1 = outputLocations[m * 4 + 0];
            float ymin1 = outputLocations[m * 4 + 1];
            float xmax1 = outputLocations[m * 4 + 0] + outputLocations[m * 4 + 2];
            float ymax1 = outputLocations[m * 4 + 1] + outputLocations[m * 4 + 3];

            float iou = CalculateOverlap(xmin0, ymin0, xmax0, ymax0, xmin1, ymin1, xmax1, ymax1);

            if (iou > threshold) {
                order[j] = -1;
            }
        }
    }
    return 0;
}

int YOLOv6Processor::quick_sort_indice_inverse(std::vector<float> &input, int left, int right, std::vector<int> &indices)
{
    float key;
    int key_index;
    int low = left;
    int high = right;
    if (left < right)
    {
        key_index = indices[left];
        key = input[left];
        while (low < high)
        {
            while (low < high && input[high] <= key)
            {
                high--;
            }
            input[low] = input[high];
            indices[low] = indices[high];
            while (low < high && input[low] >= key)
            {
                low++;
            }
            input[high] = input[low];
            indices[high] = indices[low];
        }
        input[low] = key;
        indices[low] = key_index;
        quick_sort_indice_inverse(input, left, low - 1, indices);
        quick_sort_indice_inverse(input, low + 1, right, indices);
    }
    return low;
}


// --- 【核心】跟踪器函数 (移植自 NPUProcessor) ---
float YOLOv6Processor::calculate_iou(const cv::Rect_<float>& rect1, const cv::Rect_<float>& rect2) {
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

void YOLOv6Processor::updateTracker(const std::vector<cv::Rect_<float>>& detections,
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
