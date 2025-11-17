// =======================================================
// 文件名: mobilenet.cpp (完整修改版)
// =======================================================
#include "mobilenet.h"
#include <QDebug>
#include <QThread>
#include <QFile>
#include <QTextStream>
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

// --- 构造函数 (修改) ---
MobilenetProcessor::MobilenetProcessor(QObject *parent)
    : QObject(parent),
      m_is_busy(false),
      isInitialized(false),
      // --- 【【新：初始化 m_topK】】 ---
      m_topK(3), // 默认显示 Top 3
      // --- 结束修改 ---
      model_data(nullptr),
      model_data_size(0),
      ctx(0),
      input_attrs(nullptr),
      output_attrs(nullptr),
      input_width(0), input_height(0), input_channels(0),
      npu_input_buffer(nullptr),
      npu_input_buffer_size(0),
      display_width(640),
      display_height(640),
      display_buffer_640(nullptr),
      display_buffer_size(0)
{
    qDebug() << "[Mobilenet] Processor created.";
}

// --- 析构函数 (不变) ---
MobilenetProcessor::~MobilenetProcessor()
{
    qDebug() << "[Mobilenet] Processor destroying...";
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
    if (display_buffer_640) {
        qDebug() << "  Freeing Display buffer...";
        free(display_buffer_640);
        display_buffer_640 = nullptr;
    }
    qDebug() << "[Mobilenet] Processor destroyed.";
}

// --- load_model 辅助函数 (不变) ---
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

// --- loadLabels (不变) ---
bool MobilenetProcessor::loadLabels(const QString& labelPath)
{
    QFile file(labelPath);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        qWarning() << "[Mobilenet] Cannot open label file:" << labelPath;
        return false;
    }
    m_labels.clear();
    QTextStream in(&file);
    while (!in.atEnd()) {
        QString line = in.readLine();
        m_labels.push_back(line.toStdString());
    }
    file.close();
    qDebug() << "[Mobilenet] Loaded" << m_labels.size() << "labels from" << labelPath;
    return m_labels.size() > 0;
}


// --- init 函数 (不变) ---
void MobilenetProcessor::init(const QString& modelPath)
{
    qDebug() << "[Mobilenet] Processor initializing in thread:" << QThread::currentThreadId();
    isInitialized = false;

    // 1. 加载标签
    QString labelPath = "synset.txt";
    if (!loadLabels(labelPath)) {
        qCritical() << "[Mobilenet] Failed to load labels from" << labelPath;
    }

    // 2. 加载模型
    model_data = load_model(modelPath.toStdString().c_str(), &model_data_size);
    if (model_data == nullptr) { qCritical() << "[Mobilenet] Failed to load RKNN model from" << modelPath; return; }
    qDebug() << "[Mobilenet] RKNN model loaded, size:" << model_data_size << "bytes";
    int ret = rknn_init(&ctx, model_data, model_data_size, 0, NULL);
    if (ret < 0) { qCritical() << "[Mobilenet] rknn_init error ret=" << ret; free(model_data); model_data = nullptr; return; }
    qDebug() << "[Mobilenet] RKNN context initialized.";

    // 3. 查询 IO
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != RKNN_SUCC || io_num.n_input != 1 || io_num.n_output != 1) {
        qCritical() << "[Mobilenet] Model I/O mismatch! (Inputs:" << io_num.n_input << ", Outputs:" << io_num.n_output << ")";
        rknn_destroy(ctx); ctx=0; return;
    }
    qDebug() << "[Mobilenet] Model info: inputs=" << io_num.n_input << ", outputs=" << io_num.n_output;

    // 4. 查询输入属性
    input_attrs = new rknn_tensor_attr[io_num.n_input];
    memset(input_attrs, 0, io_num.n_input * sizeof(rknn_tensor_attr));
    input_attrs[0].index = 0;
    ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[0]), sizeof(rknn_tensor_attr));
    if (ret != RKNN_SUCC) {
        qCritical() << "[Mobilenet] rknn_query(RKNN_QUERY_INPUT_ATTR) error ret=" << ret;
        rknn_destroy(ctx); ctx=0; delete[] input_attrs; input_attrs = nullptr; return;
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
    qDebug() << "    Detected Input Size (NPU): H=" << input_height << ", W=" << input_width;

    // 5. 查询输出属性
    output_attrs = new rknn_tensor_attr[io_num.n_output];
    memset(output_attrs, 0, io_num.n_output * sizeof(rknn_tensor_attr));
    output_attrs[0].index = 0;
    ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[0]), sizeof(rknn_tensor_attr));
    if (ret != RKNN_SUCC) {
         qCritical() << "[Mobilenet] rknn_query(RKNN_QUERY_OUTPUT_ATTR) error ret=" << ret;
         rknn_destroy(ctx); ctx=0; delete[] input_attrs; input_attrs = nullptr; delete[] output_attrs; output_attrs = nullptr; return;
    }
    qDebug() << "    Detected Output: " << output_attrs[0].n_elems << " classes.";

    // 6. 分配缓冲区 (不变)
    if (input_width <= 0 || input_height <= 0 || input_channels <= 0) {
        qCritical() << "[Mobilenet] Invalid model input dimensions.";
        rknn_destroy(ctx); ctx=0; delete[] input_attrs; input_attrs = nullptr; delete[] output_attrs; output_attrs = nullptr; return;
    }

    // 缓冲 1: NPU 输入 (224x224x3)
    npu_input_buffer_size = input_width * input_height * input_channels;
    npu_input_buffer = malloc(npu_input_buffer_size);
    if (npu_input_buffer == nullptr) {
        qCritical() << "[Mobilenet] Failed to allocate NPU input buffer (size=" << npu_input_buffer_size << ")";
        rknn_destroy(ctx); ctx=0; delete[] input_attrs; input_attrs = nullptr; delete[] output_attrs; output_attrs = nullptr; return;
    }

    // 缓冲 2: 显示 (640x640x3)
    display_buffer_size = display_width * display_height * input_channels;
    display_buffer_640 = malloc(display_buffer_size);
    if (display_buffer_640 == nullptr) {
        qCritical() << "[Mobilenet] Failed to allocate Display buffer (size=" << display_buffer_size << ")";
        free(npu_input_buffer); npu_input_buffer = nullptr;
        rknn_destroy(ctx); ctx=0; delete[] input_attrs; input_attrs = nullptr; delete[] output_attrs; output_attrs = nullptr; return;
    }

    isInitialized = true;
    qInfo() << "[Mobilenet] Processor initialized SUCCESSFULLY with model:" << modelPath;
}

// --- processFrame 函数 (修改) ---
void MobilenetProcessor::processFrame(const QImage& frame)
{
    if (m_is_busy.load(std::memory_order_acquire)) {
        return;
    }
    m_is_busy.store(true, std::memory_order_release);

    QElapsedTimer total_timer;
    total_timer.start();

    if (!isInitialized || frame.isNull() || ctx <= 0 || npu_input_buffer == nullptr || display_buffer_640 == nullptr) {
        emit processingFinished(QImage(), 0);
        m_is_busy.store(false, std::memory_order_release);
        return;
    }

    // --- RGA 预处理 - 步骤 1 (显示) (不变) ---
    im_rect src_rect, dst_rect_640, fill_rect_640;
    rga_buffer_t src_buf = {};
    rga_buffer_t dst_buf_640 = {};
    IM_STATUS rga_ret;

    float r_640 = std::min((float)display_width / frame.width(), (float)display_height / frame.height());
    int new_unpad_w_640 = int(round(frame.width() * r_640));
    int new_unpad_h_640 = int(round(frame.height() * r_640));
    int dw_640 = (display_width - new_unpad_w_640) / 2;
    int dh_640 = (display_height - new_unpad_h_640) / 2;

    src_buf.vir_addr = (void*)frame.bits();
    src_buf.width = frame.width();
    src_buf.height = frame.height();
    src_buf.wstride = frame.width();
    src_buf.hstride = frame.height();
    src_buf.format = RK_FORMAT_RGB_888;

    dst_buf_640.vir_addr = display_buffer_640;
    dst_buf_640.width = display_width;
    dst_buf_640.height = display_height;
    dst_buf_640.wstride = display_width;
    dst_buf_640.hstride = display_height;
    dst_buf_640.format = RK_FORMAT_RGB_888;

    fill_rect_640 = {0, 0, display_width, display_height};
    int gray_color = 0xFF727272;
    rga_ret = imfill(dst_buf_640, fill_rect_640, gray_color, 1, NULL);
    if (rga_ret != IM_STATUS_SUCCESS) {
        qWarning() << "[Mobilenet] RGA imfill (640) failed with code" << rga_ret;
        m_is_busy.store(false, std::memory_order_release); return;
    }

    src_rect = {0, 0, frame.width(), frame.height()};
    dst_rect_640 = {dw_640, dh_640, new_unpad_w_640, new_unpad_h_640};
    rga_ret = improcess(src_buf, dst_buf_640, {}, src_rect, dst_rect_640, {},
                        -1, NULL, NULL, IM_SYNC);
    if (rga_ret != IM_STATUS_SUCCESS) {
        qWarning() << "[Mobilenet] RGA improcess (640) failed with code" << rga_ret;
        m_is_busy.store(false, std::memory_order_release); return;
    }


    // --- RGA 预处理 - 步骤 2 (NPU) (不变) ---
    rga_buffer_t src_buf_640 = dst_buf_640;
    rga_buffer_t dst_buf_224 = {};
    im_rect dst_rect_224, fill_rect_224;

    float r_224 = std::min((float)input_width / src_buf_640.width, (float)input_height / src_buf_640.height);
    int new_unpad_w_224 = int(round(src_buf_640.width * r_224));
    int new_unpad_h_224 = int(round(src_buf_640.height * r_224));
    int dw_224 = (input_width - new_unpad_w_224) / 2;
    int dh_224 = (input_height - new_unpad_h_224) / 2;

    dst_buf_224.vir_addr = npu_input_buffer;
    dst_buf_224.width = input_width;
    dst_buf_224.height = input_height;
    dst_buf_224.wstride = input_width;
    dst_buf_224.hstride = input_height;
    dst_buf_224.format = RK_FORMAT_RGB_888;

    fill_rect_224 = {0, 0, input_width, input_height};
    rga_ret = imfill(dst_buf_224, fill_rect_224, gray_color, 1, NULL);
    if (rga_ret != IM_STATUS_SUCCESS) {
        qWarning() << "[Mobilenet] RGA imfill (224) failed with code" << rga_ret;
        m_is_busy.store(false, std::memory_order_release); return;
    }

    src_rect = {0, 0, src_buf_640.width, src_buf_640.height};
    dst_rect_224 = {dw_224, dh_224, new_unpad_w_224, new_unpad_h_224};
    rga_ret = improcess(src_buf_640, dst_buf_224, {}, src_rect, dst_rect_224, {},
                        -1, NULL, NULL, IM_SYNC);
    if (rga_ret != IM_STATUS_SUCCESS) {
        qWarning() << "[Mobilenet] RGA improcess (224) failed with code" << rga_ret;
        m_is_busy.store(false, std::memory_order_release); return;
    }


    // --- 设置 RKNN 输入 (不变) ---
    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].buf = npu_input_buffer;
    inputs[0].size = npu_input_buffer_size;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].pass_through = 0;
    int ret = rknn_inputs_set(ctx, io_num.n_input, inputs);
    if (ret < 0) {
        qCritical() << "[Mobilenet] rknn_inputs_set error ret=" << ret;
        m_is_busy.store(false, std::memory_order_release); return;
    }

    // --- NPU 推理 (不变) ---
    ret = rknn_run(ctx, nullptr);
    if (ret < 0) {
        qCritical() << "[Mobilenet] rknn_run fail! ret=" << ret;
        m_is_busy.store(false, std::memory_order_release); return;
    }

    // --- 获取 NPU 输出 (不变) ---
    memset(outputs, 0, sizeof(outputs));
    outputs[0].want_float = 1;
    ret = rknn_outputs_get(ctx, io_num.n_output, outputs, nullptr);
    if (ret < 0) {
        qCritical() << "[Mobilenet] rknn_outputs_get error ret=" << ret;
        m_is_busy.store(false, std::memory_order_release); return;
    }

    // --- 核心后处理 (不变) ---
    int topk = 5; // <-- 【保持不变】我们仍然计算 Top 5
    mobilenet_result results[topk];
    float* output_data = (float*)outputs[0].buf;
    int num_classes = output_attrs[0].n_elems;
    softmax(output_data, num_classes);
    get_topk_with_indices(output_data, num_classes, topk, results);
    rknn_outputs_release(ctx, io_num.n_output, outputs);


    // --- 在 640x640 图像上绘制 (修改) ---

    // 1. 准备 640x640 画布 (不变)
    cv::Mat mat_for_drawing(display_height, display_width, CV_8UC3, display_buffer_640);

    // 2. (字体等不变)
    float font_scale = 0.8;
    int line_height = 30;
    int text_thickness = 2;
    cv::Scalar text_color = cv::Scalar(0, 255, 0);

    if (m_labels.empty()) {
        cv::putText(mat_for_drawing,
                    "No labels loaded",
                    cv::Point(10, 30),
                    cv::FONT_HERSHEY_SIMPLEX,
                    font_scale, cv::Scalar(0, 0, 255), 2);
    } else {
        // --- 【【【【修改：使用 m_topK 控制绘制数量】】】】 ---
        // int num_to_draw = std::min(3, topk); // <-- 【旧代码】
        int num_to_draw = m_topK; // <-- 【新代码】
        // --- 结束修改 ---

        for (int i = 0; i < num_to_draw; ++i)
        {
            if (results[i].cls < m_labels.size()) {

                // (标签清理逻辑不变)
                std::string full_label = m_labels[results[i].cls];
                std::string label_to_draw = full_label;
                size_t first_space = full_label.find(' ');
                if (first_space != std::string::npos) {
                    label_to_draw = full_label.substr(first_space + 1);
                }

                float score = results[i].score;
                std::stringstream ss;
                ss << label_to_draw << ": " << std::fixed << std::setprecision(2) << (score * 100.0) << "%";
                std::string text_to_draw = ss.str();
                cv::Point text_origin(10, (i + 1) * line_height);

                cv::putText(mat_for_drawing,
                            text_to_draw,
                            text_origin,
                            cv::FONT_HERSHEY_SIMPLEX,
                            font_scale,
                            text_color,
                            text_thickness);
            }
        }
    }

    // --- cv::Mat -> QImage & 发送结果 (不变) ---
    QImage resultImage(mat_for_drawing.data, mat_for_drawing.cols, mat_for_drawing.rows, mat_for_drawing.step, QImage::Format_RGB888);
    emit processingFinished(resultImage.copy(), total_timer.elapsed());

    m_is_busy.store(false, std::memory_order_release);
}


// --- 【【【【新：添加槽函数实现】】】】 ---
void MobilenetProcessor::setTopK(int k)
{
    // VideoWindow 已经做了 1-5 的限制，但我们在此再次确认
    if (k > 0 && k <= 5) {
        m_topK = k;
    }
    qDebug() << "[Mobilenet] Top-K (display) set to:" << m_topK;
}


// ==============================================================================
// --- 【【【【【【【【 以下辅助函数保持不变 】】】】】】】】 ---
// ==============================================================================
//

void MobilenetProcessor::swap(element_t* a, element_t* b) {
    element_t temp = *a;
    *a = *b;
    *b = temp;
}

int MobilenetProcessor::partition(element_t arr[], int low, int high) {
    float pivot = arr[high].value;
    int i = low - 1;

    for (int j = low; j <= high - 1; j++) {
        if (arr[j].value >= pivot) { // >= for descending order
            i++;
            swap(&arr[i], &arr[j]);
        }
    }

    swap(&arr[i + 1], &arr[high]);
    return (i + 1);
}

void MobilenetProcessor::quick_sort(element_t arr[], int low, int high) {
    if (low < high) {
        int pi = partition(arr, low, high);
        quick_sort(arr, low, pi - 1);
        quick_sort(arr, pi + 1, high);
    }
}

void MobilenetProcessor::softmax(float* array, int size) {
    float max_val = array[0];
    for (int i = 1; i < size; i++) {
        if (array[i] > max_val) {
            max_val = array[i];
        }
    }

    for (int i = 0; i < size; i++) {
        array[i] -= max_val;
    }

    float sum = 0.0;
    for (int i = 0; i < size; i++) {
        array[i] = expf(array[i]);
        sum += array[i];
    }

    for (int i = 0; i < size; i++) {
        array[i] /= sum;
    }
}

void MobilenetProcessor::get_topk_with_indices(float arr[], int size, int k, mobilenet_result* result) {

    element_t* elements = (element_t*)malloc(size * sizeof(element_t));
    for (int i = 0; i < size; i++) {
        elements[i].value = arr[i];
        elements[i].index = i;
    }

    quick_sort(elements, 0, size - 1);

    for (int i = 0; i < k; i++) {
        result[i].score = elements[i].value;
        result[i].cls = elements[i].index;
    }

    free(elements);
}
