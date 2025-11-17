#ifndef IMAGEPROVIDER_H
#define IMAGEPROVIDER_H

#include <QQuickImageProvider>
#include <QImage>
#include <QMutex>

/*
 * ImageProvider 类
 * 它是 QML 和 C++ 之间的图像桥梁。
 * QML (在 Image 元素中) 会调用 requestImage() 来 "拉取" 图像。
 * 我们的 C++ 线程会 "推送" 图像到 m_latestImage。
 */
class ImageProvider : public QQuickImageProvider
{
public:
    ImageProvider() : QQuickImageProvider(QQuickImageProvider::Image) {}

    // QML 调用的核心函数
    QImage requestImage(const QString &id, QSize *size, const QSize &requestedSize) override {
        Q_UNUSED(id);
        Q_UNUSED(requestedSize);

        QImage image;
        {
            // 锁住，安全地复制图像
            QMutexLocker locker(&m_mutex);
            image = m_latestImage;
        }

        if (size) {
            *size = image.size();
        }
        return image;
    }

    // 供 C++ 调用的函数 (在 NPUProcessor 完成后调用)
    void updateImage(const QImage &image) {
        // 锁住，安全地更新图像
        QMutexLocker locker(&m_mutex);
        m_latestImage = image.copy(); // 必须是 .copy()
    }

private:
    QImage m_latestImage;
    QMutex m_mutex; // 必须使用互斥锁，因为 QML 渲染线程和 NPU 线程会同时访问
};

#endif // IMAGEPROVIDER_H


