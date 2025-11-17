// =======================================================
// 文件名: capturedevice.cpp
// =======================================================

#include "capturedevice.h"
#include <QDebug>
#include <QThread>
#include <sys/ioctl.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>
#include <cerrno>
#include <QElapsedTimer>

// --- 【 RGA 头文件】 ---
#include "rga/RgaApi.h"
#include "rga/im2d.h"
// --- 结束 ---

CaptureDevice::CaptureDevice(QObject *parent)
    : QObject(parent),
      captureThread(nullptr),
      running(false),
      dma_fd(-1),
      frameCount(0),
      lastReportTimeMs(0)
{
    setFrameProperties(1280, 720, 2);
}

CaptureDevice::~CaptureDevice()
{
    if (running) {
        stop();
    }
    if (captureThread && captureThread->isRunning()) {
        captureThread->quit();
        captureThread->wait(1000);
    }
}

void CaptureDevice::setFrameProperties(int width, int height, int bpp)
{
    frameWidth = width;
    frameHeight = height;
    frameBpp = bpp;
    qDebug() << "CaptureDevice: Set frame properties to" << frameWidth << "x" << frameHeight << "@" << bpp * 8 << "bpp";
}

bool CaptureDevice::openDevice(const QString &devPath)
{
    dma_fd = ::open(devPath.toStdString().c_str(), O_RDWR);
    if (dma_fd < 0) {
        emit errorOccurred(QString("Failed to open DMA device %1. Error: %2").arg(devPath).arg(strerror(errno)));
        return false;
    }
    return true;
}

void CaptureDevice::start()
{
    if (dma_fd < 0 || running) {
        return;
    }
    running = true;
    captureThread = QThread::create([this]() { captureLoop(); });
    connect(captureThread, &QThread::finished, captureThread, &QObject::deleteLater);
    captureThread->start();
}

void CaptureDevice::stop()
{
    qDebug() << "CaptureDevice::stop() called from thread" << QThread::currentThreadId();
    running = false;
}


// --- 【旧的软件转换（不再调用，但保留函数以备后用）】 ---
void rgb565_to_rgb888_qt(unsigned char *rgb565_buf, unsigned char *rgb888_buf, int pixel_num) {
    for (int i = 0; i < pixel_num; i++) {
        unsigned short p = ((unsigned short *)rgb565_buf)[i];
        rgb888_buf[i * 3 + 0] = (p & 0xF800) >> 8;
        rgb888_buf[i * 3 + 1] = (p & 0x07E0) >> 3;
        rgb888_buf[i * 3 + 2] = (p & 0x001F) << 3;
    }
}


// --- 【 RGA 转换】 ---
void CaptureDevice::captureLoop()
{
    const int width = frameWidth;
    const int height = frameHeight;
    const int BYTES_PER_PIXEL = frameBpp;
    const int SINGLE_ROW_BYTES = width * BYTES_PER_PIXEL;
    const int TOTAL_PIXELS = width * height;
    const int FULL_FRAME_BYTES = TOTAL_PIXELS * BYTES_PER_PIXEL;

    if (width <= 0 || height <= 0 || BYTES_PER_PIXEL <= 0 || SINGLE_ROW_BYTES <= 0 || TOTAL_PIXELS <= 0) {
        emit errorOccurred("Invalid frame properties in captureLoop.");
        running = false;
        return;
    }
    if (SINGLE_ROW_BYTES % 4 != 0) {
        emit errorOccurred(QString("SINGLE_ROW_BYTES (%1) is not divisible by 4.").arg(SINGLE_ROW_BYTES));
        running = false;
        return;
    }

    // frame_buf_565 (RGB565, 2 bytes)
    unsigned char *frame_buf_565 = (unsigned char *)malloc(FULL_FRAME_BYTES);
    // frame_buf_888 (RGB888, 3 bytes)
    unsigned char *frame_buf_888 = (unsigned char *)malloc(TOTAL_PIXELS * 3);

    if (!frame_buf_565 || !frame_buf_888) {
        emit errorOccurred("Failed to malloc frame buffers in captureLoop.");
        running = false;
        if (frame_buf_565) free(frame_buf_565);
        if (frame_buf_888) free(frame_buf_888);
        return;
    }
    qDebug() << "Allocated buffers: frame_buf_565 (" << FULL_FRAME_BYTES << "bytes), frame_buf_888 (" << TOTAL_PIXELS * 3 << "bytes)";

    DMA_OPERATION dma_op;
    memset(&dma_op, 0, sizeof(dma_op));
    dma_op.current_len = SINGLE_ROW_BYTES / 4;
    qDebug() << "DMA operation length (dwords per row for triggering):" << dma_op.current_len;

    qDebug() << "Mapping DMA address...";
    if (ioctl(dma_fd, PCI_MAP_ADDR_CMD, &dma_op) < 0) {
        emit errorOccurred(QString("ioctl(PCI_MAP_ADDR_CMD) failed: %1").arg(strerror(errno)));
        running = false;
        free(frame_buf_565);
        free(frame_buf_888);
        return;
    }
    qDebug() << "DMA address mapped.";

    qDebug() << "Flushing DMA buffers...";
    for (int frame = 0; frame < 2; ++frame) {
        if (!running) break;
        for (int i = 0; i < height; i++) {
            if (!running) break;
            if (ioctl(dma_fd, PCI_DMA_WRITE_CMD, &dma_op) < 0) { qWarning("Flush: DMA_WRITE failed"); break; }
             for (int k = 0; k < 2000; ++k);
        }
    }
     if (!running) {
        qDebug() << "Stop requested during buffer flush. Unmapping DMA address...";
        ioctl(dma_fd, PCI_UMAP_ADDR_CMD, &dma_op);
        free(frame_buf_565);
        free(frame_buf_888);
        ::close(dma_fd);
        dma_fd = -1;
        qDebug() << "CaptureDevice buffers freed.";
        return;
     }
    qDebug() << "Buffer flush complete. Starting main capture loop.";


    fpsReportTimer.start();
    lastReportTimeMs = fpsReportTimer.elapsed();
    frameCount = 0;

    QElapsedTimer step_timer;
    QElapsedTimer total_frame_timer;

    while (running)
    {
        total_frame_timer.start();
        step_timer.start();

        bool frame_capture_ok = true;

        for (int i = 0; i < height; i++)
        {
            if (!running) { frame_capture_ok = false; break; }

            // 1. 触发
            if (ioctl(dma_fd, PCI_DMA_WRITE_CMD, &dma_op) < 0) {
                emit errorOccurred(QString("ioctl(PCI_DMA_WRITE_CMD) failed at row %1: %2").arg(i).arg(strerror(errno)));
                running = false; frame_capture_ok = false; break;
            }

            // 2. 忙等待行同步 (第一个瓶颈)
            for (int k = 0; k < 2000; ++k);


            // 3. 读取
            if (SINGLE_ROW_BYTES > DMA_MAX_PACKET_SIZE) {
                emit errorOccurred(QString("DMA_MAX_PACKET_SIZE (%1) is smaller than row size (%2)!").arg(DMA_MAX_PACKET_SIZE).arg(SINGLE_ROW_BYTES));
                running = false; frame_capture_ok = false; break;
            }
            if (ioctl(dma_fd, PCI_READ_FROM_KERNEL_CMD, &dma_op) < 0) {
                emit errorOccurred(QString("ioctl(PCI_READ_FROM_KERNEL_CMD) failed at row %1: %2").arg(i).arg(strerror(errno)));
                running = false; frame_capture_ok = false; break;
            }

            // 4. 拼接
            memcpy(frame_buf_565 + (size_t)i * SINGLE_ROW_BYTES, dma_op.data.read_buf, SINGLE_ROW_BYTES);

        } // end for (each row)

        if (!running || !frame_capture_ok) break;

        // --- 计时点 1 ---
        qint64 row_capture_ms = step_timer.elapsed();
        step_timer.restart();


        // --- 【使用 RGA 进行 RGB565 -> RGB888 硬件转换】 ---

        // 1. 定义 RGA 源 (RGB565)
        rga_buffer_t src_buf = {};
        src_buf.vir_addr = (void*)frame_buf_565;
        src_buf.width = width;
        src_buf.height = height;
        src_buf.wstride = width; // wstride 是像素宽度
        src_buf.hstride = height;
        src_buf.format = RK_FORMAT_RGB_565; // 输入格式

        // 2. 定义 RGA 目标 (RGB888)
        rga_buffer_t dst_buf = {};
        dst_buf.vir_addr = (void*)frame_buf_888;
        dst_buf.width = width;
        dst_buf.height = height;
        dst_buf.wstride = width; // wstride 是像素宽度
        dst_buf.hstride = height;
        dst_buf.format = RK_FORMAT_RGB_888; // 输出格式

        // 3. 定义 1:1 复制区域
        im_rect src_rect = {0, 0, width, height};
        im_rect dst_rect = {0, 0, width, height};

        // 4. 执行 RGA 转换
        // (IM_SYNC 表示同步执行)
        IM_STATUS rga_ret = improcess(src_buf, dst_buf, {}, src_rect, dst_rect, {}, -1, NULL, NULL, IM_SYNC);

        if (rga_ret != IM_STATUS_SUCCESS) {
            qWarning() << "[Capture] RGA conversion (RGB565->RGB888) failed with code" << rga_ret << ". Skipping frame.";

            // 记录失败的计时
            qint64 convert_ms = step_timer.elapsed();
            step_timer.restart();
            qint64 emit_ms = step_timer.elapsed();

            //qWarning() << "[Capture] --- FRAME TIMING (RGA FAILED) ---";
            //qWarning() << "[Capture] Step 1 (Row Loop):" << row_capture_ms << "ms";
            //qWarning() << "[Capture] Step 2 (RGB Convert):" << convert_ms << "ms (FAILED)";
            //qWarning() << "[Capture] Step 3 (Emit QImage):" << emit_ms << "ms";
            //qWarning() << "[Capture] --- TOTAL Frame Prep:" << total_frame_timer.elapsed() << "ms ---";

            continue; // 跳过这一帧
        }
        // --- 结束 RGA 优化 ---


        // --- 计时点 2 ---
        qint64 convert_ms = step_timer.elapsed();
        step_timer.restart();

        QImage frameImage(frame_buf_888, width, height, width * 3, QImage::Format_RGB888);
        emit frameReady(frameImage.copy());

        // --- 计时点 3 和日志打印 ---
        qint64 emit_ms = step_timer.elapsed();

        //qDebug() << "[Capture] --- FRAME TIMING ---";
        //qDebug() << "[Capture] Step 1 (Row Loop):" << row_capture_ms << "ms";
        //qDebug() << "[Capture] Step 2 (RGB Convert):" << convert_ms << "ms"; // <--- 观察这个值！
        //qDebug() << "[Capture] Step 3 (Emit QImage):" << emit_ms << "ms";
        //qDebug() << "[Capture] --- TOTAL Frame Prep:" << total_frame_timer.elapsed() << "ms ---";


        // --- 更新帧率计算 ---
        frameCount++;
        qint64 currentTimeMs = fpsReportTimer.elapsed();
        qint64 timeSinceLastReportMs = currentTimeMs - lastReportTimeMs;
        if (timeSinceLastReportMs >= 1000) {
             double avgFps = (timeSinceLastReportMs > 0) ? ((double)frameCount * 1000.0 / timeSinceLastReportMs) : 0.0;
             emit fpsMetricsUpdated(avgFps);
             frameCount = 0;
             lastReportTimeMs = currentTimeMs;
        }
    }

    // --- 清理工作 ---
    qDebug() << "Capture loop finished. Cleaning up in thread" << QThread::currentThreadId();
    qDebug() << "Calling UMAP in captureLoop()";
    ioctl(dma_fd, PCI_UMAP_ADDR_CMD, &dma_op);
    qDebug() << "Freeing buffers...";
    free(frame_buf_565);
    free(frame_buf_888);
    qDebug() << "Closing device file descriptor...";
    ::close(dma_fd);
    dma_fd = -1;
    qDebug() << "CaptureDevice cleanup complete.";
}
