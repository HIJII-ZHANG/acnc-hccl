/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ALLTOALL_V_FOR_310P_PUB_H
#define ALLTOALL_V_FOR_310P_PUB_H

#include "allltoall_pipeline_base_pub.h"

namespace hccl {
const uint32_t COMPUTE_CONST = 2; // 计算Rank类型用到的常量
const uint32_t STEP_NUM = 5;
const uint32_t THIRD_STEP = 3;
const uint32_t MAX_RANK_GAP = 3;
const uint32_t DUO_RANK_NUM = 4;
const uint32_t ALIGN_CONST = 128;

struct SendMemBlock {
    u32 dstRank;
    u64 sendLen;
    u64 userInOffset;
    u64 cclDstOffset;
};

struct RecvMemBlock {
    u32 srcRank;
    u64 recvLen;
    u64 userOutOffset;
    u64 cclSrcOffset;
};

struct StepMemInfo {
    u32 srcBuffId;
    u32 dstBuffId;
    UserMemType srcMemType;
    std::pair<u32, u32> readMain; // 读Main的数据的srcRank和dstRank
    std::pair<u32, u32> readMinor; // 读Minor的数据的srcRank和dstRank
};

class AlltoAllVFor310P : public ExecutorBase {
public:
    explicit AlltoAllVFor310P(const HcclDispatcher dispatcher, Stream &mainStream, std::vector<Stream> &subStreams,
        const std::vector<LINK> &links, u32 userRank, u32 userRankSize,
        const std::vector<SendRecvInfo> &allMeshAggregationSendRecvInfo);
    ~AlltoAllVFor310P() override;
    HcclResult Prepare(DeviceMem &userInput, DeviceMem &userOutput, DeviceMem &cclInMem, DeviceMem &cclOutMem,
        const std::vector<std::shared_ptr<LocalNotify>> &signalMainToSub,
        const std::vector<std::shared_ptr<LocalNotify>> &signalSubToMain);
    HcclResult RunAsync();

protected:
private:
    std::string GetStreamIndexString();
    HcclResult NotifySubStreamStart();
    HcclResult WaitSubStreamFinish();
    HcclResult CalcSendInfo(const u32 srcDataRank, const u32 dstDataRank, const u32 times, const u64 subStepLen, SendMemBlock& sendInfo);
    HcclResult CalcRecvInfo(const u32 srcDataRank, const u32 dstDataRank, const u32 times, const u64 subStepLen, RecvMemBlock& recvInfo);
    HcclResult MainFirstLocalCopy(const u32 times, const u32 roundIdx, const u64 subStepLen);
    HcclResult MinorFirstLocalCopy(const u32 times, const u32 roundIdx, const u64 subStepLen);
    HcclResult RunAlltoAllVFor310P();
    HcclResult UpdateSendRecvRankInfo(const u32 roundIdx, const u32 stepIdx);
    HcclResult RunMainStep4(const u32 times, const u64 subStepLen);
    void UpdateMainStepMemInfo(const u32 roundIdx, const u32 stepIdx);
    HcclResult RunMainCommonSteps(const u32 times, const u32 roundIdx, const u32 stepIdx, const u64 subStepLen);
    u64 CalcMaxSendLength();
    HcclResult RunMinorStep4(const u32 times, const u64 subStepLen);
    void UpdateMinorStepMemInfo(const u32 roundIdx, const u32 stepIdx);
    HcclResult RunMinorCommonSteps(const u32 times, const u32 roundIdx, const u32 stepIdx, const u64 subStepLen);
    HcclResult RunMinorSendRecvBuffer(const u32 times, const u32 roundIdx, const u32 stepIdx, const u64 subStepLen);
    HcclResult RunMainSendRecvBuffer(const u32 times, const u32 roundIdx, const u32 stepIdx, const u64 subStepLen);
    HcclResult RunSendRecvBuffer(const u32 times, const u32 roundIdx, const u32 stepIdx, const u64 subStepLen);

    void SetNeighborRanks(const u32 roundIdx);

    Stream mainStream_;
    std::vector<Stream> subStream_;
    const std::vector<LINK> links_;
    u32 userRank_;
    u32 userRankSize_;
    const std::vector<SendRecvInfo> &allMeshAggregationSendRecvInfo_;

    DeviceMem userInput_;
    DeviceMem userOutput_;
    DeviceMem cclInMem_;
    DeviceMem cclOutMem_;
    std::vector<DeviceMem> memList_;

    bool mainRank_ = false;
    bool minorRank_ = false;

    std::vector<std::shared_ptr<LocalNotify>> signalMainToSub_;
    std::vector<std::shared_ptr<LocalNotify>> signalSubToMain_;

    u64 cclBlockSize_;
    u64 maxSizePerLoop_;

    u32 myMinor_;
    u32 myMain_;
    u32 rightMain_;
    u32 rightMinor_;
    u32 leftMain_;
    u32 leftMinor_;
    std::vector<std::pair<u32, u32>> sendRecvRankInfo_;
    StepMemInfo mainStepInfo_;
    StepMemInfo minorStepInfo_;
};
} // namespace hccl
#endif /* ALLTOALL_V_FOR_310P_PUB_H */