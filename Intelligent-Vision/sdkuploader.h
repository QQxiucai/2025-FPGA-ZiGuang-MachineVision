// =======================================================
// 文件名: sdkuploader.h (已修复)
// =======================================================
#ifndef SDKUPLOADER_H
#define SDKUPLOADER_H

#include <QObject>
#include <QImage>
#include <QMutex>
#include <QByteArray> // <-- 【新】添加这个头文件

// --- 【【【【修复 1: 包含 C-SDK 完整头文件】】】】 ---
// 这将为 C++ 编译器提供 Qiniu_Client 和 Qiniu_Mac 的
// 完整定义，解决所有 "incomplete type" 和 "sizeof" 错误。
extern "C" {
    #include <qiniu/base.h>
    #include <qiniu/conf.h>
    #include <qiniu/io.h>
    #include <qiniu/rs.h>
}
// --- 结束修复 ---

// (前向声明已不再需要)
// struct Qiniu_Client;
// struct Qiniu_Mac;

class SdkUploader : public QObject
{
    Q_OBJECT
public:
    explicit SdkUploader(QObject *parent = nullptr);
    ~SdkUploader();

public slots:
    // 由 VideoWindow 调用的槽
    void init(const QString& accessKey, const QString& secretKey);
    void doUpload(const QImage& image, const QString& bucket, const QString& qiniuKey, const QString& bucketUrl);

signals:
    // 发送回 VideoWindow 的信号
    void uploadFinished(bool success, const QString& imageUrl, const QString& errorMsg);

private:
    // C-SDK 需要的变量
    Qiniu_Client* m_client;
    Qiniu_Mac* m_mac;
    bool m_isInitialized;

    // 七牛云 C-SDK 依赖 OpenSSL，它不是线程安全的
    // 我们必须使用互斥锁保护所有 SDK 调用
    QMutex m_sdkMutex;

    // --- 【【【【新：添加这两行】】】】 ---
    QByteArray m_ak_bytes; // 用于持久存储 Access Key
    QByteArray m_sk_bytes; // 用于持久存储 Secret Key
};

#endif // SDKUPLOADER_H
