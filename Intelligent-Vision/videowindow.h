// =======================================================
// 文件名: videowindow.h (完整修改版)
// =======================================================
#ifndef VIDEOWINDOW_H
#define VIDEOWINDOW_H

#include <QObject>
#include <QElapsedTimer>
#include <atomic>
#include "imageprovider.h"
#include "capturedevice.h"
#include "npu_processor.h"
#include "cpu_processor.h"
#include "sdkuploader.h"
#include "yolov6.h"
#include "mobilenet.h"
#include <QtNetwork/QNetworkAccessManager>
#include <QThread>

class QNetworkReply;

// (枚举保持不变)
enum class ProcessingMode {
    None,
    CPU,
    NPU,    // 这是 YOLOv5
    YOLOv6,
    Mobilenet
};
enum class CaptureSource {
    None,
    Device
};

class VideoWindow : public QObject
{
    Q_OBJECT
    // (现有 Q_PROPERTY)
    Q_PROPERTY(QString qml_raw_image_source READ qml_raw_image_source NOTIFY qml_raw_image_source_changed)
    Q_PROPERTY(QString qml_processed_image_source READ qml_processed_image_source NOTIFY qml_processed_image_source_changed)
    Q_PROPERTY(double qml_capture_fps READ qml_capture_fps NOTIFY qml_capture_fps_changed)
    Q_PROPERTY(double qml_inference_fps READ qml_inference_fps NOTIFY qml_inference_fps_changed)
    Q_PROPERTY(int qml_inference_ms READ qml_inference_ms NOTIFY qml_inference_ms_changed)

    // --- 【【新：添加 Q_PROPERTY】】 ---
    Q_PROPERTY(double qml_confThreshold READ qml_confThreshold NOTIFY qml_confThresholdChanged)
    Q_PROPERTY(double qml_nmsThreshold READ qml_nmsThreshold NOTIFY qml_nmsThresholdChanged)
    Q_PROPERTY(int qml_alertDuration READ qml_alertDuration NOTIFY qml_alertDurationChanged)
    Q_PROPERTY(int qml_mobilenetTopK READ qml_mobilenetTopK NOTIFY qml_mobilenetTopKChanged)


public:
    explicit VideoWindow(QObject *parent = nullptr);
    ~VideoWindow();

    void startPipeline(const QString& npuModelPath, const QString& cpuModelPath, const QString& yolov6ModelPath, const QString& mobilenetModelPath);

    // (现有 Getters)
    QString qml_raw_image_source() const;
    QString qml_processed_image_source() const;
    double qml_capture_fps() const;
    double qml_inference_fps() const;
    int qml_inference_ms() const;
    ImageProvider* getRawImageProvider() const;
    ImageProvider* getProcessedImageProvider() const;

    // --- 【【新：Getters for Q_PROPERTY】】 ---
    double qml_confThreshold() const;
    double qml_nmsThreshold() const;
    int qml_alertDuration() const;
    int qml_mobilenetTopK() const;

public slots:
    // (QML 控制槽函数)
    void startNPU(); // YOLOv5
    void startCPU();
    void stopProcessing();
    void startYOLOv6();
    void startMobilenet();

    // --- 【【新：QML 调用的槽函数】】 ---
    Q_INVOKABLE void changeConfThreshold(double delta);
    Q_INVOKABLE void changeNmsThreshold(double delta);
    Q_INVOKABLE void changeAlertDuration(int delta);
    Q_INVOKABLE void changeMobilenetTopK(int delta);

private slots:
    // (内部槽函数保持不变)
    void onFrameFromCaptureDevice(const QImage& frame);
    void onProcessingFinished(const QImage& resultImage, qint64 frame_time_ms);
    void onCaptureFpsUpdated(double avgFps);
    void onDeviceError(const QString& errorMessage);
    void onAlertNeeded(const QImage& alertImage);
    void onUploadFinished(bool success, const QString& imageUrl, const QString& errorMsg);
    void handleMiaoTiXingFinished();

signals:
    // (QML 信号保持不变)
    void qml_raw_image_source_changed();
    void qml_processed_image_source_changed();
    void qml_capture_fps_changed();
    void qml_inference_fps_changed();
    void qml_inference_ms_changed();

    // --- 【【新：Q_PROPERTY 的 NOTIFY 信号】】 ---
    void qml_confThresholdChanged();
    void qml_nmsThresholdChanged();
    void qml_alertDurationChanged();
    void qml_mobilenetTopKChanged();

    // (内部信号)
    void sig_startNPU(const QImage& frame); // YOLOv5
    void sig_startCPU(const QImage& frame);
    void sig_startYOLOv6(const QImage& frame);
    void sig_startMobilenet(const QImage& frame);
    void sig_initUploader(const QString& accessKey, const QString& secretKey);
    void sig_startUpload(const QImage& image, const QString& bucket, const QString& qiniuKey, const QString& bucketUrl);

    // --- 【【新：广播到处理器的信号】】 ---
    void sig_updateConfThreshold(float threshold);
    void sig_updateNmsThreshold(float threshold);
    void sig_updateAlertDuration(int seconds);
    void sig_updateMobilenetTopK(int k);


private:
    void sendMiaoTiXingAlert(const QString& imageUrl);

    // (线程和处理器成员)
    QThread* m_captureThread;
    CaptureDevice* m_captureDevice;
    QThread* m_npuThread;
    NPUProcessor* m_npuProcessor; // YOLOv5
    QThread* m_cpuThread;
    CPUProcessor* m_cpuProcessor;
    ImageProvider* m_rawImageProvider;
    ImageProvider* m_processedImageProvider;
    QThread* m_uploadThread;
    SdkUploader* m_sdkUploader;
    QThread* m_yolov6Thread;
    YOLOv6Processor* m_yolov6Processor;
    QThread* m_mobilenetThread;
    MobilenetProcessor* m_mobilenetProcessor;

    // (内部状态成员)
    ProcessingMode m_processingMode;
    std::atomic<bool> m_processor_is_ready;
    qint64 m_inference_last_ms;
    double m_capture_last_fps;
    QString m_qml_raw_image_source;
    QString m_qml_processed_image_source;
    QElapsedTimer m_inferenceFpsReportTimer;
    int m_inferenceFrameCount;
    qint64 m_inferenceTotalTimeMs;

    // (网络成员保持不变)
    QNetworkAccessManager *networkManager;
    QElapsedTimer alertCooldownTimer;
    const qint64 alertGlobalCooldownMs = 30000;
    bool isUploading;

    // --- 【【新：存储当前值的私有成员】】 ---
    double m_confThreshold;
    double m_nmsThreshold;
    int m_alertDurationSeconds;
    int m_mobilenetTopK;

    // (配置保持不变)
    const QString qiniuAccessKey = "31ZuDeiuQY8K_Jim2obU4dOzYLhawTr5qHmLhGpa";
    const QString qiniuSecretKey = "mg-qmj_HrBRx2kLP_00CgfgHu5rwTL1mi38uBiLX";
    const QString qiniuBucketName = "intelligent8vision";
    const QString qiniuBucketUrl = "t4y23vd9r.hd-bkt.clouddn.com";
    const QString miaoTiXingId = "t4aD8qL";
};

#endif // VIDEOWINDOW_H
