# 1. Qt 模块 (添加 concurrent 用于后台线程)
QT += core gui network qml quick concurrent

CONFIG += c++11
DEFINES += QT_DEPRECATED_WARNINGS

# ===================================================================
# 2. 基础路径 (在您的 Ubuntu PC 上)
# ===================================================================
RKNPU_ARM_PATH = /home/rich/rk_project/rknpu_sdk
OPENCV_NEW_PATH = /home/rich/rk_project/install_opencv_4.9.0
# 【【【【新】】】】
# 指向您刚刚 "make install" 编译出的 C-SDK 目录
QINIU_SDK_INSTALL_PATH = /home/rich/rk_project/c-sdk/install

# ===================================================================
# 3. 交叉编译库和头文件路径
# ===================================================================
# 根据您的 toolchain-aarch64.cmake 和 apt 安装路径:
CROSS_COMPILE_LIB_PATH = /usr/lib/aarch64-linux-gnu
CROSS_COMPILE_INCLUDE_PATH = /usr/include/aarch64-linux-gnu

# ===================================================================
# 4. 包含路径 (INCLUDEPATH)
# ===================================================================
INCLUDEPATH += \
    $$OPENCV_NEW_PATH/include/opencv4 \
    $$RKNPU_ARM_PATH/include \
    $$QINIU_SDK_INSTALL_PATH/include \  # <-- 【新】使用 C-SDK 的 include 目录
    $$CROSS_COMPILE_INCLUDE_PATH \
    /usr/include

# ===================================================================
# 5. 链接库 (LIBS)
# ===================================================================
message(">>>>> 正在为 RK3568 交叉编译 (使用 OpenCV 4.9.0)... <<<<<")

# OpenCV 库
LIBS += -L$$OPENCV_NEW_PATH/lib \
        -lopencv_core \
        -lopencv_dnn \
        -lopencv_imgproc \
        -lopencv_highgui \
        -lopencv_imgcodecs \
        -lopencv_videoio

# RKNN 和 RGA 库
LIBS += -L$$RKNPU_ARM_PATH/lib \
        -lrknnrt \
        -lrga

# 【【【【新】】】】
# C-SDK 库 (来自您的 "install" 目录)
LIBS += -L$$QINIU_SDK_INSTALL_PATH/lib \
        -lqiniu # <-- 【新】链接 libqiniu.so

# C-SDK 依赖库 (来自您的 sysroot)
LIBS += -L$$CROSS_COMPILE_LIB_PATH
LIBS += -lcurl -lssl -lcrypto -lm

# ===================================================================
# 6. 源文件 (SOURCES)
# ===================================================================
SOURCES += \
    main.cpp \
    mobilenet.cpp \
    videowindow.cpp \
    capturedevice.cpp \
    npu_processor.cpp \
    cpu_processor.cpp \
    sdkuploader.cpp \ # <-- 【新】我们的 C-SDK 封装类
    yolov6.cpp

# (【【【【已移除】】】】: 不再需要 SOURCES += ... c-sdk/qiniu/*.c)

# ===================================================================
# 7. 头文件 (HEADERS)
# ===================================================================
HEADERS += \
    mobilenet.h \
    videowindow.h \
    imageprovider.h \
    capturedevice.h \
    npu_processor.h \
    cpu_processor.h \
    tracker_types.h \
    sdkuploader.h \ # <-- 【新】
    yolov6.h

# ===================================================================
# 8. 资源文件 (RESOURCES)
# ===================================================================
RESOURCES += \
    resources.qrc
