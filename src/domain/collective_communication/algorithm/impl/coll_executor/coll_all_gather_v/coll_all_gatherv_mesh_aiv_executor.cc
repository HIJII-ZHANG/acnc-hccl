/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_all_gatherv_mesh_aiv_executor.h"

namespace hccl {

AllGatherVMeshAivExecutor::AllGatherVMeshAivExecutor(const HcclDispatcher dispatcher,
                                                           std::unique_ptr<TopoMatcher> &topoMatcher)
    : CollAllGatherVExecutor(dispatcher, topoMatcher)
{
    isBigData_ = false;
}

void AllGatherVMeshAivExecutor::ParseParam(const OpParam& param){
    tag_ = param.tag;
    root_ = param.root;
    aicpuUnfoldMode_ = param.aicpuUnfoldMode;
    opType_ = param.opType;
    // judge data size
    const auto *countsPtr = static_cast<const u64*>(param.VDataDes.counts);
    auto countsPerRank = std::vector<u64>(countsPtr, countsPtr + topoAttr_.userRankSize);
    u64 maxCount = *std::max_element(countsPerRank.begin(), countsPerRank.end());
    u32 unitSize = SIZE_TABLE[param.VDataDes.dataType];
    u64 dataSize = maxCount * unitSize;
    if (dataSize > AIV_ALL_GATHER_SMALL_SIZE) {
        isBigData_ = true;
    } else {
        isBigData_ = false;
    }
    HCCL_INFO("[AllGatherVMeshAivExecutor][ParaseParm] isBigData_ is [%d].", isBigData_);
}

HcclResult AllGatherVMeshAivExecutor::GetIfNeedAivBuffer(bool &needAivBuffer)
{
    // AIV通信需要AIV buffer
    needAivBuffer = true;
    HCCL_INFO("[AllGatherVMeshAivExecutor][GetIfNeedAivBuffer]tag[%s] needAivBuffer is [%u].",
        tag_.c_str(), needAivBuffer);
    return HCCL_SUCCESS;
}

HcclResult AllGatherVMeshAivExecutor::CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport)
{
    TransportMemType inputType = TransportMemType::RESERVED;
    TransportMemType outputType = TransportMemType::RESERVED;
    CHK_RET(CalcTransportMemType(inputType, outputType));
    CHK_RET(CalcLevel0CommInfo(inputType, outputType, opTransport));
    return HCCL_SUCCESS;
}

HcclResult AllGatherVMeshAivExecutor::CalcTransportMemType(TransportMemType &inputType, TransportMemType &outputType)
{
    if(isBigData_){
        inputType = TransportMemType::CCL_INPUT;
    }else{
        inputType = TransportMemType::AIV_INPUT;
    }
    outputType = TransportMemType::AIV_OUTPUT;
    HCCL_INFO("[AllGatherVMeshAivExecutor][CalcTransportMemType] tag[%s] inputType[%d], outputType[%d].",
        tag_.c_str(), inputType, outputType);
    return HCCL_SUCCESS;
}

HcclResult AllGatherVMeshAivExecutor::CalcLevel0CommInfo(TransportMemType inputType,
    TransportMemType outputType,
    std::vector<LevelNSubCommTransport>& opTransport)
{
    CommParaInfo commParaLevel0(COMM_LEVEL0, CommType::COMM_TAG_MESH);
    CHK_RET(CalcCommPlaneInfo(tag_, commParaLevel0, opTransport[COMM_LEVEL0], inputType, outputType));
    return HCCL_SUCCESS;
}

HcclResult AllGatherVMeshAivExecutor::Orchestrate(OpParam& param, AlgResourceResponse& algRes)
{
    HcclUs startut = TIME_NOW();
    tag_ = param.tag;
    algResResp_ = &algRes;

    HcclResult ret = HCCL_SUCCESS;
    ExecMem execMem;

    execMem.inputPtr = param.inputPtr;
    execMem.outputPtr = param.outputPtr;
    execMem.count = (static_cast<const u64 *>(param.VDataDes.counts))[topoAttr_.userRank];

    if(isBigData_){
        execMem.inputMem = algRes.cclInputMem;
    }else{
        execMem.inputMem = algRes.aivInputMem;
    }
    execMem.outputMem = algRes.aivOutputMem;
    ret = KernelRun(param, execMem);

    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[AllGatherVMeshAivExecutor][Orchestrate]errNo[0x%016llx] tag[%s] excutor kernel run failed",
            HCCL_ERROR_CODE(ret), param.tag.c_str()), ret);

    HCCL_INFO("tag[%s], AllGatherVMeshAivExecutor orchestrate success, take time [%lld]us",
        param.tag.c_str(), DURATION_US(TIME_NOW() - startut));
    return HCCL_SUCCESS;
}

HcclResult AllGatherVMeshAivExecutor::KernelRun(const OpParam &param, ExecMem &execMem)
{
    HCCL_INFO("[AllGatherVMeshAivExecutor][KernelRun]allgatherv aiv enter.");

    CHK_RET(CheckCommSize(COMM_LEVEL0, COMM_INDEX_0 + 1));
    SubCommInfo outerCommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);

    void *buffersIn[MAX_RANK_SIZE];
    void *buffersOut[MAX_RANK_SIZE];

    u32 localRank = outerCommInfo.localRank;
    u32 localRankSize = outerCommInfo.localRankSize;
    HCCL_DEBUG("[AllGatherVMeshAivExecutor][KernelRun] userRank [%u] localRank [%u]", topoAttr_.userRank, localRank);

    ExtraArgs extraArgs;
    for (u32 i = 0; i < localRankSize; i++) {
        extraArgs.recvCounts[i] = *(static_cast<const u64 *>(param.VDataDes.counts) + i);
        extraArgs.recvDispls[i] = *(static_cast<const u64 *>(param.VDataDes.displs) + i);
        if (i != localRank) {
            CHK_RET(outerCommInfo.links[i]->GetRemoteMem(UserMemType::INPUT_MEM, &(buffersIn[i])));
            CHK_RET(outerCommInfo.links[i]->GetRemoteMem(UserMemType::OUTPUT_MEM, &(buffersOut[i])));
        } else {
            buffersIn[i] = execMem.inputMem.ptr();
            buffersOut[i] = execMem.outputMem.ptr();
        }
        if (extraArgs.recvCounts[i] > extraArgs.maxCount) {
            extraArgs.maxCount = extraArgs.recvCounts[i];
        }
    }
    bool isOpbase = (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);

    HcclResult ret = ExecuteKernelLaunch(HcclCMDType::HCCL_CMD_ALLGATHER_V, execMem.inputPtr, execMem.outputPtr,
        execMem.count, param.VDataDes.dataType, param.reduceType, localRank, localRankSize, param.root,
        buffersIn, buffersOut, param.tag, param.stream.ptr(), isOpbase, execMem.inputMem.size(), -1, false, &extraArgs);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[AllGatherVMeshAivExecutor][KernelRun]allgatherv aiv failed, return[%d]", ret), ret);

    HCCL_INFO("[AllGatherVMeshAivExecutor][KernelRun]allgatherv aiv run success.");
    return HCCL_SUCCESS;
}

REGISTER_EXEC("AllGatherVMeshAivExecutor", AllGatherVAiv, AllGatherVMeshAivExecutor);

} // namespace hccl