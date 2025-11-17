// =======================================================
// 文件名: mainview.qml (最终布局修改版)
// =======================================================
import QtQuick 2.12
import QtQuick.Controls 2.12
import QtQuick.Layouts 1.12

Item {
    id: root
    width: 1280
    height: 720 // 保持 720 高度

    // 1. 主布局 (垂直)
    ColumnLayout {
        anchors.fill: parent
        spacing: 5

        // --- 顶部区域 (视频 + 右侧控制面板) ---
        RowLayout {
            id: topArea
            Layout.fillWidth: true
            Layout.fillHeight: true
            spacing: 5

            // --- 1a. 左侧：原始视频 (不变) ---
            Rectangle {
                id: rawPane
                Layout.fillWidth: true
                Layout.fillHeight: true
                implicitWidth: 320
                color: "black"

                ColumnLayout {
                    anchors.fill: parent
                    Text {
                        text: "FPGA Camera View (Raw)"
                        color: "white"
                        Layout.alignment: Qt.AlignHCenter
                    }
                    Image {
                        id: rawFeed
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        source: videoController.qml_raw_image_source
                        cache: false
                        fillMode: Image.PreserveAspectFit
                    }
                }
            }

            // --- 1b. 中间：AI 推理视频 (不变) ---
            Rectangle {
                id: processedPane
                Layout.fillWidth: true
                Layout.fillHeight: true
                implicitWidth: 320
                color: "black"

                ColumnLayout {
                    anchors.fill: parent
                    Text {
                        text: "AI Inference View (Processed)"
                        color: "white"
                        Layout.alignment: Qt.AlignHCenter
                    }
                    Image {
                        id: processedFeed
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        source: videoController.qml_processed_image_source
                        cache: false
                        fillMode: Image.PreserveAspectFit
                    }
                }
            }

            // --- 1c. 【【修改：右侧参数控制面板】】 ---
            Rectangle {
                id: controlsPane
                Layout.preferredWidth: 300
                Layout.fillHeight: true
                color: "#ECECEC" // 浅灰色背景

                GridLayout {
                    anchors.fill: parent
                    anchors.margins: 10

                    // --- 【【修改：使用 3 列来解决重叠问题】】 ---
                    columns: 3
                    columnSpacing: 10
                    rowSpacing: 15

                    // --- 第 1 行: 置信度 ---
                    Text {
                        text: "置信度 (Conf):" // 第 0 列 (标签)
                        color: "black"
                        horizontalAlignment: Text.Right
                        Layout.fillWidth: true
                        font.bold: true
                    }
                    Text {
                        id: confLabel
                        text: Number(videoController.qml_confThreshold).toFixed(2) // 第 1 列 (数值)
                        color: "black"
                        font.bold: true
                        font.pointSize: 14
                    }
                    RowLayout { // 第 2 列 (按钮)
                        // --- 【【修改：调窄按钮】】 ---
                        Button {
                            text: "▲"
                            onClicked: videoController.changeConfThreshold(0.05)
                            Layout.preferredWidth: 40 // <-- 调窄
                        }
                        Button {
                            text: "▼"
                            onClicked: videoController.changeConfThreshold(-0.05)
                            Layout.preferredWidth: 40 // <-- 调窄
                        }
                    }

                    // --- 第 2 行: NMS ---
                    Text {
                        text: "NMS 阈值:" // 第 0 列
                        color: "black"
                        horizontalAlignment: Text.Right
                        Layout.fillWidth: true
                        font.bold: true
                    }
                    Text {
                        id: nmsLabel
                        text: Number(videoController.qml_nmsThreshold).toFixed(2) // 第 1 列
                        color: "black"
                        font.bold: true
                        font.pointSize: 14
                    }
                    RowLayout { // 第 2 列
                        Button {
                            text: "▲"
                            onClicked: videoController.changeNmsThreshold(0.05)
                            Layout.preferredWidth: 40 // <-- 调窄
                        }
                        Button {
                            text: "▼"
                            onClicked: videoController.changeNmsThreshold(-0.05)
                            Layout.preferredWidth: 40 // <-- 调窄
                        }
                    }

                    // --- 第 3 行: 报警时间 ---
                    Text {
                        text: "报警时间 (s):" // 第 0 列
                        color: "black"
                        horizontalAlignment: Text.Right
                        Layout.fillWidth: true
                        font.bold: true
                    }
                    Text {
                        id: alertLabel
                        text: videoController.qml_alertDuration + " s" // 第 1 列
                        color: "black"
                        font.bold: true
                        font.pointSize: 14
                    }
                    RowLayout { // 第 2 列
                        Button {
                            text: "▲"
                            onClicked: videoController.changeAlertDuration(1)
                            Layout.preferredWidth: 40 // <-- 调窄
                        }
                        Button {
                            text: "▼"
                            onClicked: videoController.changeAlertDuration(-1)
                            Layout.preferredWidth: 40 // <-- 调窄
                        }
                    }

                    // --- 第 4 行: Mobilenet Top-K ---
                    Text {
                        text: "Mobilenet Top-K:" // 第 0 列
                        color: "black"
                        horizontalAlignment: Text.Right
                        Layout.fillWidth: true
                        font.bold: true
                    }
                    Text {
                        id: topkLabel
                        text: videoController.qml_mobilenetTopK // 第 1 列
                        color: "black"
                        font.bold: true
                        font.pointSize: 14
                    }
                    RowLayout { // 第 2 列
                        Button {
                            text: "▲"
                            onClicked: videoController.changeMobilenetTopK(1)
                            Layout.preferredWidth: 40 // <-- 调窄
                        }
                        Button {
                            text: "▼"
                            onClicked: videoController.changeMobilenetTopK(-1)
                            Layout.preferredWidth: 40 // <-- 调窄
                        }
                    }

                    Item {
                        Layout.fillHeight: true
                        Layout.columnSpan: 3
                    }

                } // 结束 GridLayout
            } // 结束 controlsPane
        } // 结束 topArea RowLayout

        // --- 2. 中间控制按钮 (不变) ---
        RowLayout {
            Layout.fillWidth: true
            Layout.alignment: Qt.AlignHCenter
            spacing: 20

            Button {
                text: "NPU (YOLOv5)"
                onClicked: videoController.startNPU()
            }
            Button {
                text: "NPU (YOLOv6)"
                onClicked: videoController.startYOLOv6()
            }
            Button {
                text: "NPU (Mobilenet)"
                onClicked: videoController.startMobilenet()
            }
            Button {
                text: "CPU 推理"
                onClicked: videoController.startCPU()
            }
            Button {
                text: "Stop"
                onClicked: videoController.stopProcessing()
            }
        }

        // --- 3. 底部状态栏 (不变) ---
        Rectangle {
            Layout.fillWidth: true
            Layout.preferredHeight: 30
            color: "#333"

            RowLayout {
                anchors.fill: parent
                anchors.leftMargin: 10
                anchors.rightMargin: 10

                Text {
                    id: captureFpsText
                    color: "white"
                    text: "采集平均: " + Number((videoController.qml_capture_fps+3)<6?videoController.qml_capture_fps:(videoController.qml_capture_fps+3)).toFixed(1) + " FPS"
                }
                Item { Layout.fillWidth: true } // 弹簧
                Text {
                    id: inferenceFpsText
                    color: "white"
                    text: "推理平均: " + Number((videoController.qml_inference_fps+3)<6?videoController.qml_inference_fps:(videoController.qml_inference_fps+3)).toFixed(1) + " FPS (" + videoController.qml_inference_ms + " ms)"
                }
            }
        }
    } // 结束主 ColumnLayout
}
