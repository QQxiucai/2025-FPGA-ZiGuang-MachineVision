// =======================================================
// 文件名: videowindow.cpp
// =======================================================
#include "videowindow.h"
#include "imageprovider.h"
#include "capturedevice.h"
#include "npu_processor.h"
#include "cpu_processor.h"
#include "sdkuploader.h"
#include "yolov6.h"
#include "mobilenet.h"
#include <QThread>
#include <QDebug>
#include <QDateTime>
#include <QtConcurrent/QtConcurrent>
#include <QtNetwork/QNetworkAccessManager>
#include <QtNetwork/QNetworkRequest>
#include <QtNetwork/QNetworkReply>
#include <QUrl>
#include <QJsonDocument>
#include <QJsonObject>
#include <QByteArray>
#include <QCryptographicHash>
#include <QBuffer>
#include <QFile>
#include <QFileInfo>
#include <QTemporaryFile>
#include <QHttpMultiPart>
#include <QMimeDatabase>
#include <QDateTime>
#include <QMessageAuthenticationCode>
#include <algorithm>

extern "C" {
    #include <qiniu/base.h>
    #include <qiniu/conf.h>
}

// --- 辅助函数用于限制范围 ---
template<typename T>
T clamp(T value, T min, T max) {
    return std::min(std::max(value, min), max);
}


VideoWindow::VideoWindow(QObject *parent)
    : QObject(parent),
      m_captureThread(nullptr),
      m_captureDevice(nullptr),
      m_npuThread(nullptr),
      m_npuProcessor(nullptr),
      m_cpuThread(nullptr),
      m_cpuProcessor(nullptr),
      m_rawImageProvider(nullptr),
      m_processedImageProvider(nullptr),
      m_uploadThread(nullptr),
      m_sdkUploader(nullptr),
      m_yolov6Thread(nullptr),
      m_yolov6Processor(nullptr),
      m_mobilenetThread(nullptr),
      m_mobilenetProcessor(nullptr),
      m_processingMode(ProcessingMode::None),
      m_processor_is_ready(true),
      m_inference_last_ms(0),
      m_capture_last_fps(0.0),
      m_inferenceFrameCount(0),
      m_inferenceTotalTimeMs(0),
      networkManager(nullptr),
      isUploading(false),
      // --- 初始化新成员 ---
      m_confThreshold(0.5),       // 默认值
      m_nmsThreshold(0.4),        // 默认值
      m_alertDurationSeconds(5),  // 默认值
      m_mobilenetTopK(3)          // 默认值
{
    Qiniu_Servend_Init(-1);

    m_rawImageProvider = new ImageProvider();
    m_processedImageProvider = new ImageProvider();

    // 2. 创建工作器
    m_captureDevice = new CaptureDevice();
    m_npuProcessor = new NPUProcessor(); // YOLOv5
    m_cpuProcessor = new CPUProcessor();
    m_sdkUploader = new SdkUploader();
    m_yolov6Processor = new YOLOv6Processor();
    m_mobilenetProcessor = new MobilenetProcessor();

    // 3. 创建线程
    m_captureThread = new QThread();
    m_npuThread = new QThread();
    m_cpuThread = new QThread();
    m_uploadThread = new QThread();
    m_uploadThread->setObjectName("UploaderThread");
    m_yolov6Thread = new QThread();
    m_yolov6Thread->setObjectName("YOLOv6Thread");
    m_mobilenetThread = new QThread();
    m_mobilenetThread->setObjectName("MobilenetThread");

    // 4. 将工作器移动到线程
    m_captureDevice->moveToThread(m_captureThread);
    m_npuProcessor->moveToThread(m_npuThread);
    m_cpuProcessor->moveToThread(m_cpuThread);
    m_sdkUploader->moveToThread(m_uploadThread);
    m_yolov6Processor->moveToThread(m_yolov6Thread);
    m_mobilenetProcessor->moveToThread(m_mobilenetThread);

    // 5. 连接信号和槽
    connect(m_npuThread, &QThread::finished, m_npuProcessor, &QObject::deleteLater);
    connect(m_cpuThread, &QThread::finished, m_cpuProcessor, &QObject::deleteLater);
    connect(m_captureThread, &QThread::finished, m_captureDevice, &QObject::deleteLater);
    connect(m_uploadThread, &QThread::finished, m_sdkUploader, &QObject::deleteLater);
    connect(m_yolov6Thread, &QThread::finished, m_yolov6Processor, &QObject::deleteLater);
    connect(m_mobilenetThread, &QThread::finished, m_mobilenetProcessor, &QObject::deleteLater);

    // (处理完成信号)
    connect(m_npuProcessor, &NPUProcessor::processingFinished, this, &VideoWindow::onProcessingFinished, Qt::QueuedConnection);
    connect(m_cpuProcessor, &CPUProcessor::processingFinished, this, &VideoWindow::onProcessingFinished, Qt::QueuedConnection);
    connect(m_yolov6Processor, &YOLOv6Processor::processingFinished, this, &VideoWindow::onProcessingFinished, Qt::QueuedConnection);
    connect(m_mobilenetProcessor, &MobilenetProcessor::processingFinished, this, &VideoWindow::onProcessingFinished, Qt::QueuedConnection);

    // (采集器连接)
    connect(m_captureDevice, &CaptureDevice::frameReady, this, &VideoWindow::onFrameFromCaptureDevice, Qt::QueuedConnection);
    connect(m_captureDevice, &CaptureDevice::fpsMetricsUpdated, this, &VideoWindow::onCaptureFpsUpdated, Qt::QueuedConnection);
    connect(m_captureDevice, &CaptureDevice::errorOccurred, this, &VideoWindow::onDeviceError, Qt::QueuedConnection);

    // (分发信号连接)
    connect(this, &VideoWindow::sig_startNPU, m_npuProcessor, &NPUProcessor::processFrame);
    connect(this, &VideoWindow::sig_startCPU, m_cpuProcessor, &CPUProcessor::processFrame);
    connect(this, &VideoWindow::sig_startYOLOv6, m_yolov6Processor, &YOLOv6Processor::processFrame);
    connect(this, &VideoWindow::sig_startMobilenet, m_mobilenetProcessor, &MobilenetProcessor::processFrame);

    // (报警信号连接)
    connect(m_npuProcessor, &NPUProcessor::alertNeeded, this, &VideoWindow::onAlertNeeded, Qt::QueuedConnection);
    connect(m_cpuProcessor, &CPUProcessor::alertNeeded, this, &VideoWindow::onAlertNeeded, Qt::QueuedConnection);
    connect(m_yolov6Processor, &YOLOv6Processor::alertNeeded, this, &VideoWindow::onAlertNeeded, Qt::QueuedConnection);
    // (Mobilenet 没有 alertNeeded 信号)

    // (上传线程连接)
    connect(this, &VideoWindow::sig_initUploader, m_sdkUploader, &SdkUploader::init);
    connect(this, &VideoWindow::sig_startUpload, m_sdkUploader, &SdkUploader::doUpload);
    connect(m_sdkUploader, &SdkUploader::uploadFinished, this, &VideoWindow::onUploadFinished, Qt::QueuedConnection);

    // (网络管理器初始化)
    networkManager = new QNetworkAccessManager(this);
    alertCooldownTimer.start();
    alertCooldownTimer.invalidate();


    // --- 连接广播信号到所有处理器的槽 ---

    // 1. 置信度 (连接到 3 个 YOLO 处理器)
    connect(this, &VideoWindow::sig_updateConfThreshold, m_npuProcessor, &NPUProcessor::setConfThreshold);
    connect(this, &VideoWindow::sig_updateConfThreshold, m_cpuProcessor, &CPUProcessor::setConfThreshold);
    connect(this, &VideoWindow::sig_updateConfThreshold, m_yolov6Processor, &YOLOv6Processor::setConfThreshold);

    // 2. NMS 阈值 (连接到 3 个 YOLO 处理器)
    connect(this, &VideoWindow::sig_updateNmsThreshold, m_npuProcessor, &NPUProcessor::setNmsThreshold);
    connect(this, &VideoWindow::sig_updateNmsThreshold, m_cpuProcessor, &CPUProcessor::setNmsThreshold);
    connect(this, &VideoWindow::sig_updateNmsThreshold, m_yolov6Processor, &YOLOv6Processor::setNmsThreshold);

    // 3. 报警时间 (连接到 3 个 YOLO 处理器)
    connect(this, &VideoWindow::sig_updateAlertDuration, m_npuProcessor, &NPUProcessor::setAlertDuration);
    connect(this, &VideoWindow::sig_updateAlertDuration, m_cpuProcessor, &CPUProcessor::setAlertDuration);
    connect(this, &VideoWindow::sig_updateAlertDuration, m_yolov6Processor, &YOLOv6Processor::setAlertDuration);

    // 4. Mobilenet Top-K (仅连接到 Mobilenet)
    connect(this, &VideoWindow::sig_updateMobilenetTopK, m_mobilenetProcessor,
            QOverload<int>::of(&MobilenetProcessor::setTopK));
}

VideoWindow::~VideoWindow()
{
    if (m_captureThread && m_captureThread->isRunning()) {
        m_captureDevice->stop();
        m_captureThread->quit();
        m_captureThread->wait(1000);
    }
    // ... (所有其他线程的清理) ...
    if (m_npuThread && m_npuThread->isRunning()) {
        m_npuThread->quit();
        m_npuThread->wait(1000);
    }
    if (m_cpuThread && m_cpuThread->isRunning()) {
        m_cpuThread->quit();
        m_cpuThread->wait(1000);
    }
    if (m_uploadThread && m_uploadThread->isRunning()) {
        m_uploadThread->quit();
        m_uploadThread->wait(1000);
    }
    if (m_yolov6Thread && m_yolov6Thread->isRunning()) {
        m_yolov6Thread->quit();
        m_yolov6Thread->wait(1000);
    }
    if (m_mobilenetThread && m_mobilenetThread->isRunning()) {
        m_mobilenetThread->quit();
        m_mobilenetThread->wait(1000);
    }

    Qiniu_Servend_Cleanup();
}

void VideoWindow::startPipeline(const QString& npuModelPath, const QString& cpuModelPath, const QString& yolov6ModelPath, const QString& mobilenetModelPath)
{
    // (启动所有线程...)
    QMetaObject::invokeMethod(m_npuProcessor, "init", Qt::QueuedConnection, Q_ARG(QString, npuModelPath));
    m_npuThread->start();
    QMetaObject::invokeMethod(m_cpuProcessor, "init", Qt::QueuedConnection, Q_ARG(QString, cpuModelPath));
    m_cpuThread->start();
    QMetaObject::invokeMethod(m_yolov6Processor, "init", Qt::QueuedConnection, Q_ARG(QString, yolov6ModelPath));
    m_yolov6Thread->start();
    QMetaObject::invokeMethod(m_mobilenetProcessor, "init", Qt::QueuedConnection, Q_ARG(QString, mobilenetModelPath));
    m_mobilenetThread->start();
    m_uploadThread->start();
    emit sig_initUploader(qiniuAccessKey, qiniuSecretKey);
    QMetaObject::invokeMethod(m_captureDevice, "openDevice", Qt::QueuedConnection);
    QMetaObject::invokeMethod(m_captureDevice, "start", Qt::QueuedConnection);
    m_captureThread->start();

    qDebug() << "All pipelines started.";
}

// --- Getters ---
ImageProvider* VideoWindow::getRawImageProvider() const { return m_rawImageProvider; }
ImageProvider* VideoWindow::getProcessedImageProvider() const { return m_processedImageProvider; }
QString VideoWindow::qml_raw_image_source() const { return m_qml_raw_image_source; }
QString VideoWindow::qml_processed_image_source() const { return m_qml_processed_image_source; }
double VideoWindow::qml_capture_fps() const { return m_capture_last_fps; }
double VideoWindow::qml_inference_fps() const
{
    if (m_inferenceTotalTimeMs > 0 && m_inferenceFrameCount > 0) {
        return (double)m_inferenceFrameCount * 1000.0 / m_inferenceTotalTimeMs; //
    }
    return 0.0;
}
int VideoWindow::qml_inference_ms() const { return m_inference_last_ms; }

// --- Getters for Q_PROPERTY ---
double VideoWindow::qml_confThreshold() const { return m_confThreshold; }
double VideoWindow::qml_nmsThreshold() const { return m_nmsThreshold; }
int VideoWindow::qml_alertDuration() const { return m_alertDurationSeconds; }
int VideoWindow::qml_mobilenetTopK() const { return m_mobilenetTopK; }


// --- QML 槽函数 (添加帧率重置) ---

void VideoWindow::startNPU()
{
    qDebug() << "QML Button: Switched to NPU Mode (YOLOv5)";
    m_processingMode = ProcessingMode::NPU;

    // --- 【重置计数器】 ---
    m_inferenceFrameCount = 0;
    m_inferenceTotalTimeMs = 0;
    m_inferenceFpsReportTimer.restart();
    emit qml_inference_fps_changed(); // 立即更新UI

    m_processor_is_ready.store(true);
}

void VideoWindow::startCPU()
{
    qDebug() << "QML Button: Switched to CPU Mode";
    m_processingMode = ProcessingMode::CPU;

    // --- 【重置计数器】 ---
    m_inferenceFrameCount = 0;
    m_inferenceTotalTimeMs = 0;
    m_inferenceFpsReportTimer.restart();
    emit qml_inference_fps_changed(); // 立即更新UI

    m_processor_is_ready.store(true);
}

void VideoWindow::stopProcessing()
{
    qDebug() << "QML Button: Switched to None Mode (Stop)";
    m_processingMode = ProcessingMode::None;

    // --- 【重置计数器】 ---
    m_inferenceFrameCount = 0;
    m_inferenceTotalTimeMs = 0;
    m_inferenceFpsReportTimer.restart();
    emit qml_inference_fps_changed(); // 立即更新UI

    m_processor_is_ready.store(true);
}

void VideoWindow::startYOLOv6()
{
    qDebug() << "QML Button: Switched to YOLOv6 Mode";
    m_processingMode = ProcessingMode::YOLOv6;

    // --- 【重置计数器】 ---
    m_inferenceFrameCount = 0;
    m_inferenceTotalTimeMs = 0;
    m_inferenceFpsReportTimer.restart();
    emit qml_inference_fps_changed(); // 立即更新UI

    m_processor_is_ready.store(true);
}

void VideoWindow::startMobilenet()
{
    qDebug() << "QML Button: Switched to Mobilenet Mode";
    m_processingMode = ProcessingMode::Mobilenet;

    // --- 【重置计数器】 ---
    m_inferenceFrameCount = 0;
    m_inferenceTotalTimeMs = 0;
    m_inferenceFpsReportTimer.restart();
    emit qml_inference_fps_changed(); // 立即更新UI

    m_processor_is_ready.store(true);
}


// --- 内部槽函数 onFrameFromCaptureDevice  ---
void VideoWindow::onFrameFromCaptureDevice(const QImage& frame)
{
    if (frame.isNull()) return;

    // 1. 原始图像
    m_rawImageProvider->updateImage(frame);
    m_qml_raw_image_source = QString("image://livefeed_raw/frame?t=%1").arg(QDateTime::currentMSecsSinceEpoch());
    emit qml_raw_image_source_changed();

    // 2. 检查处理器是否就绪
    if (!m_processor_is_ready.load(std::memory_order_acquire)) {
        return;
    }

    // 3. 根据模式分发
    bool task_dispatched = false;
    if (m_processingMode == ProcessingMode::NPU) { // YOLOv5
        emit sig_startNPU(frame.copy());
        task_dispatched = true;
    } else if (m_processingMode == ProcessingMode::CPU) {
        emit sig_startCPU(frame.copy());
        task_dispatched = true;
    } else if (m_processingMode == ProcessingMode::YOLOv6) {
        emit sig_startYOLOv6(frame.copy());
        task_dispatched = true;
    } else if (m_processingMode == ProcessingMode::Mobilenet) {
        emit sig_startMobilenet(frame.copy());
        task_dispatched = true;
    }

    if (task_dispatched) {
        m_processor_is_ready.store(false, std::memory_order_release);
    }
}

// --- 内部槽函数 onProcessingFinished  ---
void VideoWindow::onProcessingFinished(const QImage& resultImage, qint64 frame_time_ms)
{
    if (!resultImage.isNull()) {
        m_processedImageProvider->updateImage(resultImage);
        m_qml_processed_image_source = QString("image://livefeed_processed/frame?t=%1").arg(QDateTime::currentMSecsSinceEpoch());
        emit qml_processed_image_source_changed();
    }

    m_inference_last_ms = frame_time_ms;
    m_inferenceTotalTimeMs += frame_time_ms;
    m_inferenceFrameCount++;

    if (m_inferenceFpsReportTimer.elapsed() >= 1000) {
        emit qml_inference_fps_changed();
        emit qml_inference_ms_changed();
        m_inferenceFpsReportTimer.restart();
        m_inferenceFrameCount = 0;
        m_inferenceTotalTimeMs = 0;
    } else {
        emit qml_inference_ms_changed();
    }

    m_processor_is_ready.store(true, std::memory_order_release);
}

// --- 内部槽函数 onCaptureFpsUpdated, onDeviceError ---
void VideoWindow::onCaptureFpsUpdated(double avgFps)
{
    m_capture_last_fps = avgFps;
    emit qml_capture_fps_changed();
}

void VideoWindow::onDeviceError(const QString& errorMessage)
{
    qCritical() << "Capture device error:" << errorMessage;
}


// --- QML 调用的槽函数 (新实现) ---

void VideoWindow::changeConfThreshold(double delta)
{
    //步长 0.05，范围 [0.1, 0.9]
    m_confThreshold = clamp(m_confThreshold + delta, 0.1, 0.9);

    // 1. 通知 QML 更新 UI
    emit qml_confThresholdChanged();
    // 2. 广播给所有处理器
    emit sig_updateConfThreshold(static_cast<float>(m_confThreshold));

    qDebug() << "VideoWindow: Conf Threshold set to" << m_confThreshold;
}

void VideoWindow::changeNmsThreshold(double delta)
{
    //步长 0.05，范围 [0.1, 0.9]
    m_nmsThreshold = clamp(m_nmsThreshold + delta, 0.1, 0.9);

    emit qml_nmsThresholdChanged();
    emit sig_updateNmsThreshold(static_cast<float>(m_nmsThreshold));

    qDebug() << "VideoWindow: NMS Threshold set to" << m_nmsThreshold;
}

void VideoWindow::changeAlertDuration(int delta)
{
    //步长 1s，范围 [1, 10]
    m_alertDurationSeconds = clamp(m_alertDurationSeconds + delta, 1, 10);

    emit qml_alertDurationChanged();
    emit sig_updateAlertDuration(m_alertDurationSeconds);

    qDebug() << "VideoWindow: Alert Duration set to" << m_alertDurationSeconds << "s";
}

void VideoWindow::changeMobilenetTopK(int delta)
{
    //步长 1，范围 [1, 5]
    m_mobilenetTopK = clamp(m_mobilenetTopK + delta, 1, 5);

    emit qml_mobilenetTopKChanged();
    emit sig_updateMobilenetTopK(m_mobilenetTopK);

    qDebug() << "VideoWindow: Mobilenet Top-K set to" << m_mobilenetTopK;
}


// =======================================================
// --- 【网络功能】 ---
// =======================================================

void VideoWindow::onAlertNeeded(const QImage& alertImage)
{
    qDebug() << "【NETWORK】 onAlertNeeded: 信号已接收。";
    if (isUploading) {
        qDebug() << "【NETWORK】 onAlertNeeded: 正在上传中，跳过本次报警。";
        return;
    }
    if (alertCooldownTimer.isValid() && alertCooldownTimer.elapsed() < alertGlobalCooldownMs) {
        qDebug() << "【NETWORK】 onAlertNeeded: 处于全局冷却中 (" << (alertGlobalCooldownMs - alertCooldownTimer.elapsed()) << "ms 剩余)，跳过本次报警。";
        return;
    }

    qDebug() << "【NETWORK】 onAlertNeeded: 检查通过，开始 C-SDK 上传流程...";
    isUploading = true;
    alertCooldownTimer.restart();

    QString qiniuKey = QString("alert_%1.jpg").arg(QDateTime::currentDateTime().toString("yyyyMMdd_hhmmsszzz"));

    emit sig_startUpload(alertImage.copy(), qiniuBucketName, qiniuKey, qiniuBucketUrl);
}

void VideoWindow::onUploadFinished(bool success, const QString& imageUrl, const QString& errorMsg)
{
    qDebug() << "【NETWORK】 onUploadFinished: 收到 C-SDK 上传完成信号。";
    isUploading = false;

    if (success) {
        qDebug() << "【NETWORK】 onUploadFinished: C-SDK 上传成功! URL:" << imageUrl;
        qDebug() << "【NETWORK】 onUploadFinished: 准备发送喵提醒...";
        sendMiaoTiXingAlert(imageUrl);
    } else {
        qWarning() << "【NETWORK】 onUploadFinished: C-SDK 上传失败! 错误:" << errorMsg;
    }
}

void VideoWindow::sendMiaoTiXingAlert(const QString& imageUrl)
{
    qDebug() << "【NETWORK】 sendMiaoTiXingAlert: G正在发送喵提醒通知到 ID:" << miaoTiXingId;

    QString text_prefix = "检测到移动目标，告警图片：";
    QString encoded_prefix = QString::fromUtf8(QUrl::toPercentEncoding(text_prefix.toUtf8()));
    QString encoded_text = encoded_prefix + imageUrl;

    qint64 ts_num = QDateTime::currentSecsSinceEpoch();
    QString ts_str = QString::number(ts_num);

    QString requestUrl = QString("http://miaotixing.com/trigger?id=%1&text=%2&ts=%3&type=json")
                         .arg(miaoTiXingId)
                         .arg(encoded_text)
                         .arg(ts_str);

    qDebug() << "【NETWORK】 sendMiaoTiXingAlert: Request URL:" << requestUrl;

    QNetworkRequest request(requestUrl);
    request.setHeader(QNetworkRequest::ContentTypeHeader, "application/x-www-form-urlencoded");
    request.setHeader(QNetworkRequest::UserAgentHeader, "Mozilla/5.0 (QtNetwork)");

    QNetworkReply *reply = networkManager->post(request, QByteArray());
    connect(reply, &QNetworkReply::finished, this, &VideoWindow::handleMiaoTiXingFinished);
}

void VideoWindow::handleMiaoTiXingFinished()
{
    qDebug() << "【NETWORK】 handleMiaoTiXingFinished: 收到喵提醒发送完成信号。";
    QNetworkReply *reply = qobject_cast<QNetworkReply*>(sender());
    if (!reply) {
         qWarning() << "【NETWORK】 handleMiaoTiXingFinished: Sender 为空!";
         return;
    }

    if (reply->error() == QNetworkReply::NoError) {
        QByteArray responseData = reply->readAll();
        qDebug() << "【NETWORK】 handleMiaoTiXingFinished: 喵提醒通知发送成功! 响应:" << responseData;
    } else {
        qWarning() << "【NETWORK】 handleMiaoTiXingFinished: 发送喵提醒通知失败! 错误:" << reply->errorString();
        qWarning() << "【NETWORK】 handleMiaoTiXingFinished: 服务器响应:" << reply->readAll();
    }
    reply->deleteLater();
}
