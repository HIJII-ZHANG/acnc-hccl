/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COLL_ALG_COMM_H
#define COLL_ALG_COMM_H

#include <string>
#include <set>
#include <unordered_set>

#include "hccl_common.h"
#include "hccl_types.h"
#include "transport_pub.h"
#include "stream_pub.h"
#include "local_notify.h"
#include "hccl_trace_info.h"
#include "common.h"
#include "threadManage.h"
#include "transport_common.h"

namespace hccl {
using RankId = u32;

enum OpMode {
    OPBASE = 0,
    OFFLOAD = 1
};

enum DeviceMode {
    HOST = 0,
    AICPU = 1
};

using LevelNSubCommTransport = std::vector<SingleSubCommTransport>;
using OpCommTransport = std::vector<LevelNSubCommTransport>;

struct AlgResourceRequest {
    u64 scratchMemSize = 0;
    u32 streamNum = 0;
    u32 notifyNum = 0;
    bool needAivBuffer = false;
    DeviceMode mode = DeviceMode::HOST;     // 用于区分是host模式，还是aicpu模式
    OpCommTransport opTransport;
    void Describe()
    {
        HCCL_DEBUG("[AlgResourceRequest], scratchMemSize[%u], streamNum[%u], notifyNum[%u], needAivBuffer[%u], "
            "DeviceMode[%d].", scratchMemSize, streamNum, notifyNum, needAivBuffer, mode);
    };
};

struct AlgResourceResponse {
    DeviceMem cclInputMem;
    DeviceMem cclOutputMem;
    DeviceMem paramInputMem;
    DeviceMem paramOutputMem;
    DeviceMem scratchMem;
    DeviceMem aivInputMem;
    DeviceMem aivOutputMem;
    std::vector<Stream> slaveStreams;
    std::vector<Stream> slaveDevStreams;
    std::vector<std::shared_ptr<LocalNotify> > notifiesMain; // Main Signals, 与Aux成对使用，大小等同于slaveStreams
    std::vector<std::shared_ptr<LocalNotify> > notifiesAux; // Auxiliary Signals, 与Main成对使用, 大小等同于slaveStreams
    std::vector<std::shared_ptr<LocalNotify> > notifiesDevMain; // 大小等同于slaveStreams
    std::vector<std::shared_ptr<LocalNotify> > notifiesDevAux; // 大小等同于slaveStreams
    OpCommTransport opTransportResponse; // 默认的Transport资源
    OpCommTransport opTransportResponseBackUp;  // Transport备资源 (借轨场景使用)
    std::vector<std::shared_ptr<ThreadManage>> threadManage;
};

enum class BatchSendRecvCurMode {
    SEND = 0,
    RECV = 1,
    SEND_RECV = 2,
    SEND_RECV_RESERVED
};

// InplaceSupportRetry算法枚举
enum class InplaceSupportRetryStatus {
    AG_BD_CASE = 0,
    RETRY_1_ALLOW_NO_DMA_REDUCE_CASE1 = 1, // executor需要成非DMA削减模式
    RETRY_0_NOT_ALLOW_NO_DMA_REDUCE_CASE1 = 2,
    ALWAYS_NO_DMA_REDUCE = 3,
    RETRY_1_ALLOW_NO_DMA_REDUCE_CASE2 = 4, // executor需要成非DMA削减模式
    RETRY_0_NOT_ALLOW_NO_DMA_REDUCE_CASE2 = 5,
    UNKONWN_EXECUTOR = 6,
    USER_LARGER_THAN_CCL = 7,
    NOT_BASIC_OP_CASE = 8,
    INPLACE_STATUS_END
};

struct OpParam {
    std::string tag = "";
    Stream stream;
    void* inputPtr = nullptr;
    u64 inputSize = 0;
    void* outputPtr = nullptr;
    u64 outputSize = 0;
    HcclReduceOp reduceType = HcclReduceOp::HCCL_REDUCE_RESERVED;
    SyncMode syncMode = SyncMode::DEFAULT_TIMEWAITSYNCMODE;
    RankId root = INVALID_VALUE_RANKID;
    RankId dstRank = 0;
    RankId srcRank = 0;
    bool aicpuUnfoldMode = false;
    HcclTraceInfo* opBaseAtraceInfo = nullptr;
    union {
        struct {
            u64 count;
            HcclDataType dataType;
            u64 strideCount;
        } DataDes = {0, HCCL_DATA_TYPE_RESERVED, 0};
        struct {
            void* counts;
            void* displs;
            HcclDataType dataType;
        } VDataDes;
        struct {
            HcclDataType sendType;
            HcclDataType recvType;
            u64 sendCount;
            void* sendCounts;
            void* recvCounts;
            void* sdispls;
            void* rdispls;
            void* sendCountMatrix;
        } All2AllDataDes;
        struct {
            HcclSendRecvItem* sendRecvItemsPtr;
            u32 itemNum;
            u32 curIterNum;
            BatchSendRecvCurMode curMode;
        } BatchSendRecvDataDes;
    };
    HcclCMDType opType = HcclCMDType::HCCL_CMD_INVALID;
    bool isZeroCopy = false;
    u32 index = 0;
};

struct OpRetryHandler {
    bool inplaceSupportRetry = false;
    bool retryEnable = false;
    InplaceSupportRetryStatus inPlaceSupportRetryStatus = InplaceSupportRetryStatus::INPLACE_STATUS_END;
    bool isInplacePreSync = false;
    bool isPostSync = false;
};

struct Mc2Handler {
    u64 version = 0;        // Mc2Handler 版本标记
    u64 commitAddr = 0;     // mc2 条件算子的监听地址
    u64 finishAddr = 0;     // mc2 写任务的地址
    u64 valueAddr = 0;
    u32 rankSize = 0;       // mc2 作用的卡数
    u32 repeatCnt = 0;      // 一次通信消息可下发多轮通信，标记为通信的轮数
    u8 stepSize = 0;        // 细粒度通信下的通信步长
    u8 skipLocalRankCopy = 0;    // 跳过本卡拷贝
    u8 skipBufferWindowCopy = 0; // 跳过user in到 cclbuffer 的拷贝
};

struct AlgOpContext {
    OpRetryHandler opRetryHandler;
    Mc2Handler mc2Handler;
};
}   // namespace hccl
#endif