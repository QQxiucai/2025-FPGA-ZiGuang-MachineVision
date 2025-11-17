#ifndef TRACKER_TYPES_H
#define TRACKER_TYPES_H

#include <QElapsedTimer>
#include <opencv2/core/types.hpp> // 包含 cv::Rect_

// 定义跟踪对象结构体
// 这个结构体现在被 CPU 和 NPU 处理器共用
struct TrackedObject {
    int id = -1;                          // 跟踪ID
    cv::Rect_<float> rect;              // 边界框
    int frames_since_seen = 0;          // 距离上次看到的帧数
    // --- 核心修改：移除了 status 成员 ---
    // enum Status { OUTSIDE, INSIDE } status = OUTSIDE;
    int classId = -1;                     // 类别 ID
    float confidence = 0.0f;              // 置信度
    QElapsedTimer appearanceTimer;      // 连续出现计时器
    bool alertSentForThisAppearance = false; // 本次出现是否已报警
    static const int max_frames_to_forget = 10; // 跟踪丢失的帧数阈值
};

#endif // TRACKER_TYPES_H

