// =======================================================
// 文件名: sdkuploader.cpp
// =======================================================
#include "sdkuploader.h"
#include <QDebug>
#include <QFile>
#include <QTemporaryFile>
#include <QMutexLocker>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <QElapsedTimer>
#include <QThread>
#include <curl/curl.h>

// 包含 C-SDK 头文件
extern "C" {
    #include <qiniu/base.h>
    #include <qiniu/conf.h>
    #include <qiniu/io.h>
    #include <qiniu/rs.h>
    #include <qiniu/http.h>
}

// 定义华东区域 (Z0) 的 HTTPS 上传域名
const char* QINIU_UP_HOST = "up-z0.qiniup.com";

SdkUploader::SdkUploader(QObject *parent)
    : QObject(parent), m_client(nullptr), m_mac(nullptr), m_isInitialized(false)
{
    qDebug() << "【SDK】 SdkUploader created in thread" << QThread::currentThreadId();
}

SdkUploader::~SdkUploader()
{
    QMutexLocker locker(&m_sdkMutex);
    qDebug() << "【SDK】 Destroying SdkUploader...";
    if (m_client) {
        Qiniu_Client_Cleanup(m_client);
        free(m_client);
    }
    if (m_mac) {
        free(m_mac);
    }
    qDebug() << "【SDK】 SdkUploader destroyed.";
}

void SdkUploader::init(const QString& accessKey, const QString& secretKey)
{
    QMutexLocker locker(&m_sdkMutex);
    if (m_isInitialized) return;

    qDebug() << "【SDK】 Initializing Qiniu C-SDK in thread" << QThread::currentThreadId();

    // --- 【使用 QByteArray 持久化密钥】 ---

    // 1. 将 QString 转换为 QByteArray 并存储在类成员中
    m_ak_bytes = accessKey.toUtf8();
    m_sk_bytes = secretKey.toUtf8();

    // 2. 将全局指针指向我们持久化的内存
    QINIU_ACCESS_KEY = m_ak_bytes.constData();
    QINIU_SECRET_KEY = m_sk_bytes.constData();


    // 2. 分配 Qiniu_Mac
    m_mac = (Qiniu_Mac*)malloc(sizeof(Qiniu_Mac));
    if (!m_mac) {
        qWarning() << "【SDK】 Failed to allocate Qiniu_Mac.";
        return;
    }
    m_mac->accessKey = QINIU_ACCESS_KEY;
    m_mac->secretKey = QINIU_SECRET_KEY;

    // 3. 分配和初始化客户端
    m_client = (Qiniu_Client*)malloc(sizeof(Qiniu_Client));
    if (!m_client) {
        qWarning() << "【SDK】 Failed to allocate Qiniu_Client.";
        return;
    }
    Qiniu_Client_InitMacAuth(m_client, 1024, m_mac);

    // --- 【设置默认上传域名】---
    // Qiniu_Conf_SetHost 和 QINIU_HOST_UP 现在已定义 (来自 conf.h)
    //Qiniu_Conf_SetHost(QINIU_HOST_UP, QINIU_UP_HOST);

    m_isInitialized = true;
    qDebug() << "【SDK】 Qiniu C-SDK initialized successfully (Hardcoded Z0 UP Host).";
}

void SdkUploader::doUpload(const QImage& image, const QString& bucket, const QString& qiniuKey, const QString& bucketUrl)
{
    QMutexLocker locker(&m_sdkMutex);
    if (!m_isInitialized) {
        qWarning() << "【SDK】 doUpload called, but SDK is not initialized.";
        emit uploadFinished(false, "", "SDK not initialized");
        return;
    }

    qDebug() << "【SDK】 doUpload: Received upload job for key:" << qiniuKey;
    QElapsedTimer uploadTimer;
    uploadTimer.start();

    // --- 1. 将 QImage 保存到临时文件 ---
    QString tempFilePath = "/tmp/alert_image_sdk.jpg";
    std::string temp_std_path = tempFilePath.toStdString();
    cv::Mat alert_mat;

    if (image.format() == QImage::Format_RGB888) {
        alert_mat = cv::Mat(image.height(), image.width(), CV_8UC3, (void*)image.bits(), image.bytesPerLine()).clone();
        cv::cvtColor(alert_mat, alert_mat, cv::COLOR_RGB2BGR);
    } else {
        qWarning() << "【SDK】 doUpload: Image format is not RGB888.";
        emit uploadFinished(false, "", "Invalid image format");
        return;
    }

    if (!cv::imwrite(temp_std_path, alert_mat)) {
        qWarning() << "【SDK】 doUpload: cv::imwrite failed to save file.";
        emit uploadFinished(false, "", "Failed to save temp file");
        return;
    }
    qDebug() << "【SDK】 doUpload: Image saved to" << tempFilePath;

    // --- 2. C-SDK: 生成上传 Token ---
    Qiniu_RS_PutPolicy putPolicy;
    Qiniu_Zero(putPolicy);

    std::string scope_str = (bucket + ":" + qiniuKey).toStdString();
    putPolicy.scope = scope_str.c_str();
    putPolicy.expires = 3600; // 1小时有效期

    char* uptoken = Qiniu_RS_PutPolicy_Token(&putPolicy, m_mac);
    // ...
    qDebug() << "【SDK】 doUpload: Upload token generated. Token:" << uptoken;
    // ...
    // --- 3. C-SDK: 执行上传 ---
    Qiniu_Io_PutRet putRet;
    Qiniu_Io_PutExtra putExtra;
    Qiniu_Zero(putExtra);

    // ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
    // 添加这行代码 (这是旧版 SDK 设置上传区域的方法)
    putExtra.upHost = "http://up-z0.qiniup.com";

    std::string key_str = qiniuKey.toStdString();

    Qiniu_Error err = Qiniu_Io_PutFile(
        m_client,
        &putRet,
        uptoken,
        key_str.c_str(),
        temp_std_path.c_str(),
        &putExtra
    );

    // --- 4. 处理结果 ---
    if (err.code != 200) {
        qWarning() << "【SDK】 C-SDK Qiniu_Io_PutFile failed!";
        qWarning() << "【SDK】 Error Code:" << err.code;
        qWarning() << "【SDK】 Error Msg:" << err.message;

        qWarning() << "【SDK】 Response Header:" << Qiniu_Buffer_CStr(&m_client->respHeader);
        qWarning() << "【SDK】 Response Body:" << Qiniu_Buffer_CStr(&m_client->b);

        emit uploadFinished(false, "", err.message);
    } else {
        // 成功!
        qDebug() << "【SDK】 C-SDK Upload Success! (Took" << uploadTimer.elapsed() << "ms)";
        qDebug() << "【SDK】 Hash:" << putRet.hash;
        qDebug() << "【SDK】 Key:" << putRet.key;

        QString imageUrl = "http://" + bucketUrl + "/" + qiniuKey;
        emit uploadFinished(true, imageUrl, "");
    }

    // --- 5. 清理 ---
    Qiniu_Free(uptoken);
    QFile::remove(tempFilePath);
}
