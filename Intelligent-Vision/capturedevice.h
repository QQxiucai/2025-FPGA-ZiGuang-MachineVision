#ifndef CAPTUREDEVICE_H
#define CAPTUREDEVICE_H

#include <QObject>
#include <QImage>
#include <QThread>
#include <QElapsedTimer>
#include <sys/ioctl.h>

// --- All the #define macros ---
#define PCIE_DRIVER_FILE_PATH "/dev/pango_pci_driver"

// --- 【【关键修正】】 ---
// #define TYPE 'S' // <-- [错误] "TYPE" 是一个太通用的宏，与 Qt QML 头文件冲突
#define PCIE_DRIVER_IOCTL_TYPE 'S' // <-- [修正] 改为一个唯一的宏名称
// --- 结束修正 ---

#define PCI_READ_DATA_CMD        _IOWR(PCIE_DRIVER_IOCTL_TYPE, 0, int) // <-- [修正] 使用新宏
#define PCI_WRITE_DATA_CMD       _IOWR(PCIE_DRIVER_IOCTL_TYPE, 1, int) // <-- [修正] 使用新宏
#define PCI_MAP_ADDR_CMD         _IOWR(PCIE_DRIVER_IOCTL_TYPE, 2, int) // <-- [修正] 使用新宏
#define PCI_WRITE_TO_KERNEL_CMD  _IOWR(PCIE_DRIVER_IOCTL_TYPE, 3, int) // <-- [修正] 使用新宏
#define PCI_DMA_READ_CMD         _IOWR(PCIE_DRIVER_IOCTL_TYPE, 4, int) // <-- [修正] 使用新宏
#define PCI_DMA_WRITE_CMD        _IOWR(PCIE_DRIVER_IOCTL_TYPE, 5, int) // <-- [修正] 使用新宏
#define PCI_READ_FROM_KERNEL_CMD _IOWR(PCIE_DRIVER_IOCTL_TYPE, 6, int) // <-- [修正] 使用新宏
#define PCI_UMAP_ADDR_CMD        _IOWR(PCIE_DRIVER_IOCTL_TYPE, 7, int) // <-- [修正] 使用新宏
#define DMA_MAX_PACKET_SIZE      4096

// --- The struct definitions ---
typedef struct _DMA_DATA_ {
    unsigned char read_buf[DMA_MAX_PACKET_SIZE];
    unsigned char write_buf[DMA_MAX_PACKET_SIZE];
} DMA_DATA;

typedef struct _DMA_OPERATION_ {
    unsigned int current_len;
    unsigned int offset_addr;
    unsigned int cmd;
    DMA_DATA data;
} DMA_OPERATION;


class CaptureDevice : public QObject
{
    Q_OBJECT

public:
    explicit CaptureDevice(QObject *parent = nullptr);
    ~CaptureDevice();

    void setFrameProperties(int width, int height, int bpp);
public slots: 
    void start();
    void stop();
    bool openDevice(const QString &devPath = PCIE_DRIVER_FILE_PATH);


signals:
    void frameReady(const QImage &frame);
    void errorOccurred(const QString &errorMessage);
    void fpsMetricsUpdated(double avgFps);

private:
    void captureLoop();

    QThread *captureThread;
    volatile bool running;
    int dma_fd;

    int frameWidth;
    int frameHeight;
    int frameBpp;

    QElapsedTimer fpsReportTimer;
    int frameCount;
    qint64 lastReportTimeMs;
};

#endif // CAPTUREDEVICE_H
