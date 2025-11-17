// =======================================================
// 文件名: cpu_processor.cpp (完整修改版 - 适配 NCHW 三输出)
// =======================================================
#include "cpu_processor.h"
#include <QDebug>
#include <QThread>
#include <QElapsedTimer>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <algorithm>
#include <iomanip>
#include <cmath>
#include <fstream>
#include <sstream>

// --- 构造函数实现 (修改) ---
CPUProcessor::CPUProcessor(QObject *parent)
    : QObject(parent),
      m_is_busy(false),
      isInitialized(false),
      m_confThreshold(0.5f),
      m_nmsThreshold(0.4f),
      m_personAlertDurationMs(5000),
      next_track_id(0)
{
    lastAlertTime.start();
    lastAlertTime.invalidate();
    qDebug() << "CPUProcessor created. Alert cooldown:" << alertCooldownMs << "ms. Person duration threshold:" << m_personAlertDurationMs << "ms";

    // --- 【【新：添加YOLOv5-3输出的参数】】 ---
    m_strides = {8, 16, 32};
    m_anchor_grid.push_back({{10, 13}, {16, 30}, {33, 23}});
    m_anchor_grid.push_back({{30, 61}, {62, 45}, {59, 119}});
    m_anchor_grid.push_back({{116, 90}, {156, 198}, {373, 326}});
    // --- 结束添加 ---
}

// --- 析构函数实现 (不变) ---
CPUProcessor::~CPUProcessor()
{
    qDebug() << "CPU Processor destroyed.";
}

// --- 【【新：添加 sigmoid 函数】】 ---
float CPUProcessor::sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// --- init 函数实现 (不变) ---
void CPUProcessor::init(const QString& modelPath)
{
    qDebug() << "CPU Processor initializing in thread:" << QThread::currentThreadId();
    isInitialized = false;
    try {
        net = cv::dnn::readNetFromONNX(modelPath.toStdString());
        if (net.empty()) {
             qCritical() << "-----------------------------------------";
             qCritical() << "加载 ONNX 模型失败:" << modelPath;
             qCritical() << "-----------------------------------------";
             return;
        }
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

        classNames = {
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
        if (classNames.empty()) {
            qWarning() << "类别名称列表为空！";
        }

        qInfo() << "-----------------------------------------";
        qInfo() << "CPU Processor 初始化成功 (OpenVINO)，模型:" << modelPath;
        qInfo() << "-----------------------------------------";
        isInitialized = true;

    } catch (const cv::Exception& e) {
        qCritical() << "-----------------------------------------";
        qCritical() << "初始化过程中发生 OpenCV 异常:" << e.what();
        qCritical() << "-----------------------------------------";
        isInitialized = false;
    } catch (const std::exception& e) {
        qCritical() << "-----------------------------------------";
        qCritical() << "初始化过程中发生标准异常:" << e.what();
        qCritical() << "-----------------------------------------";
        isInitialized = false;
    } catch (...) {
         qCritical() << "-----------------------------------------";
         qCritical() << "初始化过程中发生未知异常。";
         qCritical() << "-----------------------------------------";
         isInitialized = false;
    }
}


// --- processFrame 函数实现 (不变) ---
void CPUProcessor::processFrame(const QImage& frame)
{
    if (m_is_busy.load(std::memory_order_acquire)) {
        return;
    }
    m_is_busy.store(true, std::memory_order_release);

    QElapsedTimer total_timer;
    total_timer.start();

    if (!isInitialized || frame.isNull()) {
        emit processingFinished(QImage(), 0);
        m_is_busy.store(false, std::memory_order_release);
        return;
    }

    // 1. QImage -> cv::Mat (不变)
    cv::Mat mat;
    if (frame.format() == QImage::Format_RGB888) {
         mat = cv::Mat(frame.height(), frame.width(), CV_8UC3, const_cast<uchar*>(frame.bits()), frame.bytesPerLine()).clone();
         cv::cvtColor(mat, mat, cv::COLOR_RGB2BGR);
    } else if (frame.format() == QImage::Format_RGB32 || frame.format() == QImage::Format_ARGB32 || frame.format() == QImage::Format_ARGB32_Premultiplied) {
        cv::Mat temp(frame.height(), frame.width(), CV_8UC4, const_cast<uchar*>(frame.bits()), frame.bytesPerLine());
        cv::cvtColor(temp, mat, cv::COLOR_BGRA2BGR);
    } else {
         qWarning() << "CPUProcessor: 不支持的 QImage 格式:" << frame.format();
         emit processingFinished(QImage(), total_timer.elapsed());
         m_is_busy.store(false, std::memory_order_release);
         return;
    }
    cv::Mat original_mat_for_drawing = mat.clone();

    // 2. 预处理 (不变)
    cv::Mat blob = format_yolov5(mat);

    // 3. 推理与后处理 (不变, 但 detect() 内部逻辑已改变)
    std::vector<cv::Rect_<float>> final_detections;
    std::vector<int> final_classIds;
    std::vector<float> final_confidences;
    detect(blob, original_mat_for_drawing, final_detections, final_classIds, final_confidences);

    // 4. 更新跟踪器 (不变)
    updateTracker(final_detections, final_classIds, final_confidences);

    // 5. 绘制结果 & 执行报警逻辑 (不变)
    bool alert_triggered_this_frame = false;
    int current_person_count = 0;
    for (const auto& obj : tracked_objects) {
        if (obj.frames_since_seen == 0) {
            cv::Scalar color = (obj.classId == 0) ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
            cv::rectangle(original_mat_for_drawing, obj.rect, color, 2);
            if (obj.classId == 0) { current_person_count++; }
            std::stringstream ss_label;
            if (obj.classId >= 0 && obj.classId < classNames.size()) { ss_label << classNames[obj.classId] << " "; }
            ss_label << "ID:" << obj.id;
            ss_label << " " << std::fixed << std::setprecision(2) << obj.confidence;
            std::string label = ss_label.str();
            cv::putText(original_mat_for_drawing, label, cv::Point(obj.rect.x, obj.rect.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);

            if (obj.classId == 0) {
                if (obj.appearanceTimer.isValid() && obj.appearanceTimer.elapsed() >= m_personAlertDurationMs && !obj.alertSentForThisAppearance) {
                    if (!alert_triggered_this_frame && (!lastAlertTime.isValid() || lastAlertTime.elapsed() > alertCooldownMs)) {
                        qDebug() << "CPU: Person ID" << obj.id << "已持续出现 >=" << (m_personAlertDurationMs/1000) << "秒.";
                        cv::Mat mat_rgb;
                        cv::cvtColor(original_mat_for_drawing, mat_rgb, cv::COLOR_BGR2RGB);
                        QImage alertImage(mat_rgb.data, mat_rgb.cols, mat_rgb.rows, mat_rgb.step, QImage::Format_RGB888);
                        emit alertNeeded(alertImage.copy());
                        lastAlertTime.restart();
                        alert_triggered_this_frame = true;
                        for (auto& tracked_obj_ref : tracked_objects) {
                            if (tracked_obj_ref.id == obj.id) {
                                tracked_obj_ref.alertSentForThisAppearance = true;
                                break;
                            }
                        }
                    }
                }
            }
        }
    }
    cv::putText(original_mat_for_drawing, "Persons: " + std::to_string(current_person_count), cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);

    // 6. cv::Mat -> QImage & 发送结果 (不变)
    cv::cvtColor(original_mat_for_drawing, original_mat_for_drawing, cv::COLOR_BGR2RGB);
    QImage resultImage(original_mat_for_drawing.data, original_mat_for_drawing.cols, original_mat_for_drawing.rows, original_mat_for_drawing.step, QImage::Format_RGB888);

    emit processingFinished(resultImage.copy(), total_timer.elapsed());

    m_is_busy.store(false, std::memory_order_release);
}

// --- format_yolov5 函数 (不变) ---
cv::Mat CPUProcessor::format_yolov5(const cv::Mat &source) {
    int col = source.cols;
    int row = source.rows;
    int _max = std::max(col, row);
    cv::Mat result = cv::Mat::zeros(_max, _max, CV_8UC3);
    source.copyTo(result(cv::Rect(0, 0, col, row)));

    if (result.cols != inputWidth || result.rows != inputHeight) {
         float r = std::min((float)inputWidth / result.cols, (float)inputHeight / result.rows);
         int new_unpad_w = int(round(result.cols * r));
         int new_unpad_h = int(round(result.rows * r));
         cv::Mat resized_img;
         cv::resize(result, resized_img, cv::Size(new_unpad_w, new_unpad_h));
         int pad_w = (inputWidth - new_unpad_w) / 2;
         int pad_h = (inputHeight - new_unpad_h) / 2;
         result = cv::Mat::zeros(inputHeight, inputWidth, CV_8UC3);
         resized_img.copyTo(result(cv::Rect(pad_w, pad_h, new_unpad_w, new_unpad_h)));
    }
    result = cv::dnn::blobFromImage(result, 1./255., cv::Size(inputWidth, inputHeight), cv::Scalar(), true, false);
    return result;
 }

// --- 【【【【核心修改：重写 detect 函数以处理3个输出 (NCHW 布局)】】】】 ---
void CPUProcessor::detect(const cv::Mat &image_blob, cv::Mat &original_image,
                          std::vector<cv::Rect_<float>>& detections,
                          std::vector<int>& classIds,
                          std::vector<float>& confidences)
{
    detections.clear();
    classIds.clear();
    confidences.clear();
    if (net.empty()) return;

    net.setInput(image_blob);
    std::vector<cv::Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());


    if (outputs.size() != 3) {
        qWarning() << "CPUProcessor: 错误！模型应有3个输出，但 `net.forward()` 返回了" << outputs.size() << "个。";
        qWarning() << "请确保加载了正确的三输出ONNX模型。";

        // 尝试回退到单输出逻辑
        if (outputs.size() == 1 && outputs[0].dims == 3) {
             qWarning() << "CPUProcessor: 检测到1个输出，回退到单输出解码逻辑...";
             // (单输出逻辑 - 原版)
             int num_proposals = outputs[0].size[1];
             int proposal_length = outputs[0].size[2];
             if (proposal_length != (5 + classNames.size())) {
                 qWarning() << "CPUProcessor: 单输出提议长度错误。";
                 return;
             }
             float* data = (float*)outputs[0].data;
             float original_width = original_image.cols;
             float original_height = original_image.rows;
             float r = std::min((float)inputWidth / original_width, (float)inputHeight / original_height);
             int pad_w = (int)((inputWidth - original_width * r) / 2);
             int pad_h = (int)((inputHeight - original_height * r) / 2);
             float scale_w = (float)original_width / (inputWidth - 2 * pad_w);
             float scale_h = (float)original_height / (inputHeight - 2 * pad_h);
             std::vector<int> tempClassIds;
             std::vector<float> tempConfidences;
             std::vector<cv::Rect> boxes;
             for (int i = 0; i < num_proposals; ++i) {
                float confidence = data[4];
                if (confidence >= m_confThreshold) {
                    float* classes_scores_ptr = data + 5;
                    cv::Mat scores(1, classNames.size(), CV_32FC1, classes_scores_ptr);
                    cv::Point class_id_point;
                    double max_class_score;
                    minMaxLoc(scores, 0, &max_class_score, 0, &class_id_point);
                    if (max_class_score > 0.25) {
                        float final_confidence = confidence * (float)max_class_score;
                        float cx = data[0]; float cy = data[1]; float w = data[2]; float h = data[3];
                        float x1 = (cx - w / 2); float y1 = (cy - h / 2); float x2 = (cx + w / 2); float y2 = (cy + h / 2);
                        int scaled_x1 = static_cast<int>((x1 - pad_w) * scale_w); int scaled_y1 = static_cast<int>((y1 - pad_h) * scale_h);
                        int scaled_x2 = static_cast<int>((x2 - pad_w) * scale_w); int scaled_y2 = static_cast<int>((y2 - pad_h) * scale_h);
                        scaled_x1 = std::max(0, scaled_x1); scaled_y1 = std::max(0, scaled_y1);
                        scaled_x2 = std::min((int)original_width - 1, scaled_x2); scaled_y2 = std::min((int)original_height - 1, scaled_y2);
                        tempClassIds.push_back(class_id_point.x);
                        tempConfidences.push_back(final_confidence);
                        boxes.push_back(cv::Rect(scaled_x1, scaled_y1, scaled_x2 - scaled_x1, scaled_y2 - scaled_y1));
                    }
                }
                data += proposal_length;
             }
             std::vector<int> nms_result_indices;
             cv::dnn::NMSBoxes(boxes, tempConfidences, m_confThreshold, m_nmsThreshold, nms_result_indices);
             for (int index : nms_result_indices) {
                detections.push_back(cv::Rect_<float>(boxes[index]));
                classIds.push_back(tempClassIds[index]);
                confidences.push_back(tempConfidences[index]);
             }
        }
        return;
    }

    // --- (三输出逻辑) ---

    // --- 存储所有检测框，用于最后NMS ---
    std::vector<cv::Rect> boxes_nms;
    std::vector<float> confs_nms;
    std::vector<int> classIds_nms;

    // --- 计算缩放/填充参数 (与 NPU/CPU 旧版相同) ---
    float original_width = original_image.cols;
    float original_height = original_image.rows;
    float r = std::min((float)inputWidth / original_width, (float)inputHeight / original_height);
    int pad_w = (int)((inputWidth - original_width * r) / 2);
    int pad_h = (int)((inputHeight - original_height * r) / 2);
    float scale_w = (float)original_width / (inputWidth - 2 * pad_w);
    float scale_h = (float)original_height / (inputHeight - 2 * pad_h);

    const int num_anchors = 3;
    const int elements_per_det = NUM_CLASSES_CPU + 5; // 85

    // --- 【【【【核心修改：从NHWC改为NCHW，并动态识别Stride】】】】 ---

    // --- 循环处理 3 个输出 (逻辑改编自 npu_processor.cpp) ---
    for (const cv::Mat& output_mat : outputs) // <-- 【新逻辑】 遍历所有输出
    {
        // --- 适配 OpenCV DNN 的 float/NCHW 输出 ---
        // ONNX模型输出为 [batch, Channels, Height, Width] (NCHW)
        const float* data = (float*)output_mat.data;

        // 检查维度 (假设 batch_size=1)
        if (output_mat.dims != 4) {
             qWarning() << "CPUProcessor: 输出维度不是4 (应为 B,C,H,W)。实际为:" << output_mat.dims;
             continue;
        }

        // --- 【【新：读取NCHW维度】】 ---
        int channels = output_mat.size[1]; // C
        int grid_h = output_mat.size[2]; // H
        int grid_w = output_mat.size[3]; // W

        // --- 【【新：根据 H/W 动态确定 Stride 和 Anchors】】 ---
        int stride_idx = -1;
        if (grid_h == 80) stride_idx = 0; // Stride 8
        else if (grid_h == 40) stride_idx = 1; // Stride 16
        else if (grid_h == 20) stride_idx = 2; // Stride 32
        else {
            qWarning() << "CPUProcessor: 无法识别的输出网格尺寸: H=" << grid_h << ", W=" << grid_w;
            continue;
        }

        int stride = m_strides[stride_idx];
        const std::vector<cv::Size>& anchors = m_anchor_grid[stride_idx];


        if (channels != num_anchors * elements_per_det) {
            // 【【【【这就是您看到的错误】】】】
            qWarning() << "CPUProcessor: 输出 (Stride=" << stride << ") 通道数错误。应为" << (num_anchors * elements_per_det) << "，实际为" << channels;
            continue;
        }

        // --- 【【新：适配 NCHW 布局的指针偏移】】 ---
        int grid_plane_size = grid_h * grid_w; // C平面大小

        // 遍历网格 (H, W)
        for (int h = 0; h < grid_h; ++h) {
            for (int w = 0; w < grid_w; ++w) {

                // 遍历3个锚点
                for (int a = 0; a < num_anchors; ++a) {

                    // --- 适配 NCHW 布局 ---
                    // proposal_base 指向 [a * 85, 0, 0]
                    const float* proposal_base = data + (a * elements_per_det) * grid_plane_size;
                    // proposal 指向 [a * 85, h, w]
                    const float* proposal = proposal_base + (h * grid_w + w);

                    // 解码 obj_conf (float)
                    // (proposal + 4*grid_plane_size) 指向 [a*85 + 4, h, w]
                    float obj_conf = sigmoid(proposal[4 * grid_plane_size]);

                    if (obj_conf < m_confThreshold) {
                        continue;
                    }

                    // 找到分数最高的类别
                    // class_scores_ptr 指向 [a*85 + 5, h, w]
                    const float* class_scores_ptr = proposal + 5 * grid_plane_size;

                    // 【【新：正确的 NCHW 类别分数解码】】
                    cv::Point class_id_point;
                    float max_class_score = 0.0f;
                    class_id_point.x = -1;

                    for (int c = 0; c < NUM_CLASSES_CPU; ++c) {
                        // 关键：在C维度上跳跃
                        float score = class_scores_ptr[c * grid_plane_size];
                        if (score > max_class_score) {
                            max_class_score = score;
                            class_id_point.x = c;
                        }
                    }
                    max_class_score = sigmoid(max_class_score);
                    // 【【结束 NCHW 类别解码】】


                    float final_conf = obj_conf * max_class_score;

                    if (final_conf < m_confThreshold) {
                        continue;
                    }

                    // 解码 [tx, ty, tw, th] (float) (NCHW 指针)
                    float tx = proposal[0 * grid_plane_size];
                    float ty = proposal[1 * grid_plane_size];
                    float tw = proposal[2 * grid_plane_size];
                    float th = proposal[3 * grid_plane_size];

                    // 转换回 [cx, cy, w, h] (与NPU版本逻辑一致)
                    float cx = (sigmoid(tx) * 2.0f - 0.5f + w) * stride;
                    float cy = (sigmoid(ty) * 2.0f - 0.5f + h) * stride;
                    float out_w = pow(sigmoid(tw) * 2.0f, 2) * anchors[a].width;
                    float out_h = pow(sigmoid(th) * 2.0f, 2) * anchors[a].height;

                    // 转换为 [x1, y1, w, h]
                    float x1 = cx - out_w * 0.5f;
                    float y1 = cy - out_h * 0.5f;

                    // 缩放回原始图像尺寸
                    float scaled_x1 = (x1 - pad_w) * scale_w;
                    float scaled_y1 = (y1 - pad_h) * scale_h;
                    float scaled_w = out_w * scale_w;
                    float scaled_h = out_h * scale_h;

                    // 存储用于 NMS (使用 cv::Rect，因为 NMSBoxes 需要)
                    boxes_nms.push_back(cv::Rect(static_cast<int>(scaled_x1),
                                                 static_cast<int>(scaled_y1),
                                                 static_cast<int>(scaled_w),
                                                 static_cast<int>(scaled_h)));
                    confs_nms.push_back(final_conf);
                    classIds_nms.push_back(class_id_point.x);
                }
            }
        }
    } // 结束 3 个输出的循环

    // --- 在所有输出都处理完后，执行一次 NMS ---
    std::vector<int> nms_indices;
    cv::dnn::NMSBoxes(boxes_nms, confs_nms, m_confThreshold, m_nmsThreshold, nms_indices);

    for (int index : nms_indices) {
        detections.push_back(cv::Rect_<float>(boxes_nms[index]));
        classIds.push_back(classIds_nms[index]);
        confidences.push_back(confs_nms[index]);
    }
    // --- 核心修改结束 ---
}

// --- calculate_iou 函数 (不变) ---
float CPUProcessor::calculate_iou(const cv::Rect_<float>& rect1, const cv::Rect_<float>& rect2) {
    float x_left = std::max(rect1.x, rect2.x);
    float y_top = std::max(rect1.y, rect2.y);
    float x_right = std::min(rect1.x + rect1.width, rect2.x + rect2.width);
    float y_bottom = std::min(rect1.y + rect1.height, rect2.x + rect2.height);
    if (x_right < x_left || y_bottom < y_top) return 0.0f;
    float intersection_area = (x_right - x_left) * (y_bottom - y_top);
    float rect1_area = rect1.width * rect1.height;
    float rect2_area = rect2.width * rect2.height;
    float union_area = rect1_area + rect2_area - intersection_area;
    return union_area > 0 ? intersection_area / union_area : 0.0f;
}

// --- updateTracker 函数实现 (不变) ---
void CPUProcessor::updateTracker(const std::vector<cv::Rect_<float>>& detections,
                                 const std::vector<int>& classIds,
                                 const std::vector<float>& confidences)
{
    const float iou_threshold = 0.3f;
    for (auto& obj : tracked_objects) {
        obj.frames_since_seen++;
        if (obj.frames_since_seen > 1) {
             if (obj.appearanceTimer.isValid()) {
                 obj.appearanceTimer.invalidate();
             }
             obj.alertSentForThisAppearance = false;
        }
    }
    std::vector<bool> matched_detections(detections.size(), false);
    for (size_t i = 0; i < tracked_objects.size(); ++i) {
        if (tracked_objects[i].frames_since_seen > TrackedObject::max_frames_to_forget) continue;
        float best_iou = 0.0f;
        int best_match_idx = -1;
        for (size_t j = 0; j < detections.size(); ++j) {
            if (!matched_detections[j] && tracked_objects[i].classId == classIds[j]) {
                float iou = calculate_iou(tracked_objects[i].rect, detections[j]);
                if (iou > iou_threshold && iou > best_iou) {
                    best_iou = iou;
                    best_match_idx = j;
                }
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
    tracked_objects.erase(
        std::remove_if(tracked_objects.begin(), tracked_objects.end(),
                       [](const TrackedObject& obj){ return obj.frames_since_seen > TrackedObject::max_frames_to_forget; }),
        tracked_objects.end());
}


// --- 槽函数实现 (不变) ---
void CPUProcessor::setConfThreshold(float conf)
{
    m_confThreshold = conf;
    qDebug() << "[CPU] CONF_THRESHOLD set to:" << m_confThreshold;
}

void CPUProcessor::setNmsThreshold(float nms)
{
    m_nmsThreshold = nms;
    qDebug() << "[CPU] NMS_THRESHOLD set to:" << m_nmsThreshold;
}

void CPUProcessor::setAlertDuration(int seconds)
{
    m_personAlertDurationMs = seconds * 1000;
    qDebug() << "[CPU] Alert Duration set to:" << m_personAlertDurationMs << "ms";
}
