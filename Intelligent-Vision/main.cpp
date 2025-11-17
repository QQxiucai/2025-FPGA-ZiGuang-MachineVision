// =======================================================
// 文件名: main.cpp (已修改以支持所有模型)
// =======================================================
#include <QGuiApplication>
#include <QQuickView>
#include <QQmlContext>
#include <QThread>
#include <QDebug>
#include "videowindow.h"
#include "imageprovider.h"

int main(int argc, char *argv[])
{
    QCoreApplication::setAttribute(Qt::AA_EnableHighDpiScaling);
    QGuiApplication app(argc, argv);

    // 1. 创建 C++ 控制器
    VideoWindow controller;

    // 2. 创建 QQuickView
    QQuickView view;

    // 3. 注册 C++ 对象到 QML
    view.engine()->addImageProvider(QLatin1String("livefeed_raw"), controller.getRawImageProvider());
    view.engine()->addImageProvider(QLatin1String("livefeed_processed"), controller.getProcessedImageProvider());
    view.engine()->rootContext()->setContextProperty(QLatin1String("videoController"), &controller);

    // 4. 设置 QML 源文件
    view.setSource(QUrl(QStringLiteral("qrc:/mainview.qml")));

    // 5. 检查 QML 是否加载成功
    if (view.status() == QQuickView::Error) {
        qWarning() << "Failed to load QML:" << view.errors();
        return -1;
    }

    // 6. 显示 QML 窗口
    view.setResizeMode(QQuickView::SizeRootObjectToView);
    view.show();

    // 7. 【【【【修改：启动 C++ 后台线程】】】】
    // 假设您的模型都存放在 /home/linaro/ 目录下
    // (Mobilenet 模型名称来自您提供的 PDF)
    controller.startPipeline("/home/linaro/yolov5n.rknn",        // npuModelPath (YOLOv5)
                             "/home/linaro/yolov5n.onnx",        // cpuModelPath
                             "/home/linaro/yolov6n.rknn",         // <-- 【新】yolov6ModelPath
                             "/home/linaro/mobilenet_v2.rknn"); // <-- 【新】mobilenetModelPath

    return app.exec();
}
