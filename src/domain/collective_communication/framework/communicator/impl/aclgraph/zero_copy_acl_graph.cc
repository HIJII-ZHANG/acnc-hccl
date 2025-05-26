/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "zero_copy_acl_graph.h"
#include "stream_utils.h"
namespace hccl {
ZeroCopyAclGraph::ZeroCopyAclGraph() : tagResourceIndex_(0)
{}
ZeroCopyAclGraph::~ZeroCopyAclGraph()
{}
bool ZeroCopyAclGraph::SetAclGraphZeroCopyMode(
    DevType deviceType, HcclCMDType opType, OpParam &opParam, u64 bufferSize, u32 userRankSize)
{
    bool isInGraphCaputureZeroCopy = false;       
    rtModel_t rtModel = nullptr;
    bool isCapture = false;

    if (deviceType != DevType::DEV_TYPE_910_93) {
        HCCL_INFO("[ZeroCopyAclGraph][SetAclGraphZeroCopyMode] Hccl doen't support graph zero copy mode. current "
                    "device is %d not DEV_TYPE_910_93",
                    deviceType);
        return false;
    }
    if (opParam.isZeroCopy) {
        HCCL_INFO("[ZeroCopyAclGraph][SetAclGraphZeroCopyMode] Hccl cann't support graph zero copy mode and operator "
                    "zero copy at the same time.");
        return false;
    }

    GetStreamCaptureInfo(opParam.stream.ptr(), rtModel, isCapture);
    if (isCapture) {
        isInGraphCaputureZeroCopy = SetGraphMode(opType, opParam, bufferSize, userRankSize);
    }      
    return isInGraphCaputureZeroCopy;
}

bool ZeroCopyAclGraph::SetGraphMode(HcclCMDType opType, OpParam &opParam, u64 bufferSize, u32 userRankSize)
{
    if (!opParam.aicpuUnfoldMode || GetExternalInputHcclAivMode()) {
        HCCL_INFO("[ZeroCopyAclGraph][SetAclGraphZeroCopyMode] Hccl cann't support graph zero copy "
                    "mode. Only support on aicpu mode aicpuUnfoldMode %d aiv %d",
                    opParam.aicpuUnfoldMode, GetExternalInputHcclAivMode());
        return false;
    }
    if (IsAlgoSupportAclGraphZeroCopyMode(opType, opParam, bufferSize, userRankSize)) {
        std::stringstream ss;
        ss << std::hex << std::uppercase << (tagResourceIndex_++);
        std::string strTagFix = ss.str();
        opParam.tag = opParam.tag + strTagFix;
        SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB);
        HCCL_INFO("[ZeroCopyAclGraph][SetAclGraphZeroCopyMode] Hccl set op %d workflow mode to SetWorkflowMode "
                    "graph zero copy mode.",
                    opType);
        return true;
    }
    return false;
}

bool ZeroCopyAclGraph::IsAlgoSupportAclGraphZeroCopyMode(
    HcclCMDType opType, OpParam &opParam, u64 bufferSize, u32 userRankSize)
{
    if (opType == HcclCMDType::HCCL_CMD_BROADCAST || opType == HcclCMDType::HCCL_CMD_ALLREDUCE ||
        opType == HcclCMDType::HCCL_CMD_REDUCE || opType == HcclCMDType::HCCL_CMD_SEND ||
        opType == HcclCMDType::HCCL_CMD_RECEIVE || opType == HcclCMDType::HCCL_CMD_ALLGATHER ||
        opType == HcclCMDType::HCCL_CMD_ALLTOALL || opType == HcclCMDType::HCCL_CMD_ALLTOALLV ||
        opType == HcclCMDType::HCCL_CMD_SCATTER) {
        HCCL_INFO("[ZeroCopyAclGraph] OP %d support acl graph zero copy.", opType);
        return true;
    }
    if (IsReduceScatterSupportAclGraphZeroCopyMode(opParam, bufferSize, userRankSize)) {
        return true;
    }

    return false;
}

bool ZeroCopyAclGraph::IsReduceScatterSupportAclGraphZeroCopyMode(
    const OpParam &opParam, u64 bufferSize, u32 userRankSize)
{      
    if (opParam.opType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER) {
        u64 totalSize = userRankSize * opParam.DataDes.count * SIZE_TABLE[opParam.DataDes.dataType];
        if (totalSize <= bufferSize) {
            HCCL_INFO(
                "[ZeroCopyAclGraph] REDUCE_SCATTER support acl graph zero copy. scratchMemSize=%lu cclbuffer size =%lu",
                totalSize, bufferSize);
            return true;
        }
        else
        {
            HCCL_INFO("[ZeroCopyAclGraph] REDUCE_SCATTER doesn't support acl graph zero copy. scratchMemSize=%lu "
                        "cclbuffer size =%lu",
                        totalSize, bufferSize);
        }
    }    
    return false;
}

bool ZeroCopyAclGraph::SetAclGraphZeroCopyModeAllocAlgResource(DevType deviceType, const OpParam &opParam,
    AlgResourceResponse &algResResponse, DeviceMem &outputBuffer, std::vector<DeviceMem> &deviceResOrigMem, u32 userRankSize)
{
    bool isSetScratch = false;
   
    if (deviceType != DevType::DEV_TYPE_910_93) {
        HCCL_INFO("[ZeroCopyAclGraph][SetAclGraphZeroCopyModeAllocAlgResource] Hccl doen't support graph zero copy "
                    "mode. current "
                    "device is %d not DEV_TYPE_910_93",
                    deviceType);
        return false;
    }
    if (opParam.isZeroCopy) {
        HCCL_INFO("[ZeroCopyAclGraph][SetAclGraphZeroCopyModeAllocAlgResource] Hccl cann't support graph zero copy "
                    "mode and operator "
                    "zero copy at the same time.");
        return false;
    }

    if (!opParam.aicpuUnfoldMode || GetExternalInputHcclAivMode()) {
        HCCL_INFO("[ZeroCopyAclGraph][SetAclGraphZeroCopyModeAllocAlgResource] Hccl cann't support graph zero copy "
                    "mode. Only support on aicpu mode aicpuUnfoldMode %d aiv %d",
                    opParam.aicpuUnfoldMode, GetExternalInputHcclAivMode());
        return false;
    }

    rtModel_t rtModel = nullptr;
    rtStreamCaptureStatus captureStatus = rtStreamCaptureStatus::RT_STREAM_CAPTURE_STATUS_NONE;
    rtError_t ret = rtStreamGetCaptureInfo(opParam.stream.ptr(), &captureStatus, &rtModel);
    if (ret != RT_ERROR_NONE && captureStatus != rtStreamCaptureStatus::RT_STREAM_CAPTURE_STATUS_ACTIVE) {
        return false;
    }

    isSetScratch = ReduceScatterAllocAlgResource(opParam, algResResponse, outputBuffer, deviceResOrigMem, userRankSize);
    return isSetScratch;
}

bool ZeroCopyAclGraph::ReduceScatterAllocAlgResource(const OpParam &opParam, AlgResourceResponse &algResResponse,
    DeviceMem &outputBuffer, std::vector<DeviceMem> &deviceResOrigMem, u32 userRankSize)
{
    if (IsReduceScatterSupportAclGraphZeroCopyMode(opParam, outputBuffer.size(), userRankSize)) {
        // cce reduce地址32字节对齐，截取32字节对齐后的内存地址
        DeviceMem tmpBuffer =
            DeviceMem::create(outputBuffer.ptr(), outputBuffer.size());
        u64 totalSize = userRankSize * opParam.DataDes.count * SIZE_TABLE[opParam.DataDes.dataType];
        algResResponse.scratchMem = tmpBuffer.range(0, totalSize);
        deviceResOrigMem.emplace_back(std::move(tmpBuffer));
        HCCL_INFO("[ZeroCopyAclGraph][ReduceScatterAllocAlgResource] Alloc resource success.");
        return true;
    }
    return false;
}
}