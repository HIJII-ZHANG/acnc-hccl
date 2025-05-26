/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "coll_all_gather_ring_for_910_93_executor.h"

namespace hccl {
CollAllGatherRingFor91093Executor::CollAllGatherRingFor91093Executor(const HcclDispatcher dispatcher,
                                                                   std::unique_ptr<TopoMatcher> &topoMatcher)
    : CollAllGatherExecutor(dispatcher, topoMatcher)
{
    DMAReduceFlag_ = workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE;
}

HcclResult CollAllGatherRingFor91093Executor::CalcStreamNum(u32& streamNum)
{
    u32 totalStreamNum = (topoType_ == TopoType::TOPO_TYPE_NP_DOUBLE_RING ? LEVEL0_PLANE_NUM_IN_NPRING_DOUBLE :
        LEVEL0_PLANE_NUM_IN_NPRING_SINGLE);
    if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        totalStreamNum *= STREAM_NUM_FOR_DMAREDUCE_ONE_RING;
    }
    if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB &&
        GetExternalInputEnableRdmaSdmaConcurrent()) {
        totalStreamNum += (topoType_ == TopoType::TOPO_TYPE_NP_DOUBLE_RING) ? LEVEL0_PLANE_NUM_IN_NPRING_DOUBLE :
        LEVEL0_PLANE_NUM_IN_NPRING_SINGLE;
    }
    streamNum = totalStreamNum - 1;
    HCCL_INFO("[CollAllGatherRingFor91093Executor][CalcStreamNum] tag[%s] streamNum_[%u]",
        tag_.c_str(), streamNum);
    return HCCL_SUCCESS;
}

void CollAllGatherRingFor91093Executor::ParseParam(const OpParam& param)
{
    tag_ = param.tag;
    root_ = param.root;
    aicpuUnfoldMode_ = param.aicpuUnfoldMode;
    opType_ = param.opType;
    isZeroCopy_ = param.isZeroCopy;
}

HcclResult CollAllGatherRingFor91093Executor::CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport)
{
    TransportMemType inputType = TransportMemType::RESERVED;
    TransportMemType outputType = TransportMemType::RESERVED;
    CHK_RET(CalcTransportMemType(inputType, outputType));
    CHK_RET(CalcLevel0CommInfo(inputType, outputType, opTransport));
    CHK_RET(CalcLevel1CommInfo(inputType, outputType, opTransport));
    CHK_RET(CalcLevel2CommInfo(inputType, outputType, opTransport));
    if (isZeroCopy_) {
        CHK_RET(CalcExchangeCommInfo(opTransport));
    }
    return HCCL_SUCCESS;
}

HcclResult CollAllGatherRingFor91093Executor::CalcTransportMemType(TransportMemType &inputType,
    TransportMemType &outputType)
{
    if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        inputType = TransportMemType::CCL_INPUT;
        outputType = TransportMemType::CCL_OUTPUT;
    } else {
        inputType = TransportMemType::PARAM_INPUT;
        outputType = TransportMemType::PARAM_OUTPUT;
    }
    HCCL_INFO("[CollAllGatherRingFor91093Executor][CalcTransportMemType] tag[%s] inputType[%d], outputType[%d]",
        tag_.c_str(), inputType, outputType);
    return HCCL_SUCCESS;
}

HcclResult CollAllGatherRingFor91093Executor::CalcLevel0CommInfo(TransportMemType inputType,
    TransportMemType outputType,
    std::vector<LevelNSubCommTransport>& opTransport)
{
    CommParaInfo commParaLevel0(COMM_LEVEL0, CommType::COMM_TAG_RING_INNER);
    CHK_RET(CalcCommPlaneInfo(tag_, commParaLevel0, opTransport[COMM_LEVEL0], inputType, outputType));
    return HCCL_SUCCESS;
}

HcclResult CollAllGatherRingFor91093Executor::CalcLevel2CommInfo(TransportMemType inputType, TransportMemType outputType,
    std::vector<LevelNSubCommTransport>& opTransport)
{
    if( algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_AHC ||
        algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_AHC_BROKE) {
        HCCL_INFO("[CollAllGatherRingFor91093Executor][CalcLevel2CommInfo] select AHC bypass level2 comm calulate");        
        return HCCL_SUCCESS;
    }
    
    CommParaInfo commParaLevel2(COMM_LEVEL2, CommType::COMM_TAG_MAX);
    if (algType_.algoLevel2 == AlgTypeLevel2::ALG_LEVEL2_NHR) {
        commParaLevel2.commType = CommType::COMM_TAG_NONUNIFORM_HIERARCHICAL_RING;
        HCCL_INFO("[%s]Calc NHRCommInfo", __func__);
    } else if (algType_.algoLevel2 == AlgTypeLevel2::ALG_LEVEL2_NB) {
        commParaLevel2.commType = CommType::COMM_TAG_NONUNIFORM_BRUCK;
        HCCL_INFO("[%s]Calc NBCommInfo", __func__);
    } else {
        commParaLevel2.commType = CommType::COMM_TAG_RING_INNER;
        HCCL_INFO("[%s]Calc RingCommInfo", __func__);
    }
    CHK_RET(CalcCommPlaneInfo(tag_, commParaLevel2, opTransport[COMM_LEVEL2], inputType, outputType));
    return HCCL_SUCCESS;
}

u64 CollAllGatherRingFor91093Executor::CalcLoopMaxCount(const u64 cclBuffSize, const u32 unitSize)
{
    u64 maxCountPerLoop = cclBuffSize / topoAttr_.userRankSize / HCCL_MIN_SLICE_ALIGN
        * HCCL_MIN_SLICE_ALIGN / unitSize;
    return maxCountPerLoop;
}

bool CollAllGatherRingFor91093Executor::IsDataSplitForRdmaSdmaConcurrent(const u64 curSize)
{
    bool isLargeSize = (curSize >= HCCL_SPLIT_SIZE_INTER_SERVER);
    return GetExternalInputEnableRdmaSdmaConcurrent() && (topoAttr_.serverNum > 1) && isLargeSize;
}

HcclResult CollAllGatherRingFor91093Executor::RunIntraSeverAllGather(
    const std::string &tag, DeviceMem &inputMem, DeviceMem &outputMem,
    const u64 count, const HcclDataType &dataType, const std::vector<std::vector<Slice>> &multRingsSliceZero,
    const Stream &stream, s32 profStage, const u64 baseOffset, const HcomCollOpInfo *opInfo,
    const std::vector<std::vector<Slice>> &multRingsUserMemSlice)
{
    CHK_RET(MultiRingAllGather(tag, inputMem, outputMem, count, dataType,
        multRingsSliceZero, stream, profStage, baseOffset, opInfo, multRingsUserMemSlice));
    return HCCL_SUCCESS;
}

HcclResult CollAllGatherRingFor91093Executor::CalcExchangeCommInfo(std::vector<LevelNSubCommTransport>& opTransport)
{
    std::set<u32> commTargetUserRankSet;
    u32 remoteRankSend = 0;
    u32 remoteRankRecv = 0;

    CHK_RET(CalExchangeRemoteRank(remoteRankSend, remoteRankRecv));
    commTargetUserRankSet.insert(remoteRankSend);
    commTargetUserRankSet.insert(remoteRankRecv);
    CommParaInfo commParaInfo(COMM_COMBINE_ORDER, CommType::COMM_TAG_PARTIAL_MESH_COMBINED, INVALID_VALUE_RANKID,
        INVALID_VALUE_RANKID, false, false, commTargetUserRankSet);

    TransportMemType inputType = TransportMemType::CCL_INPUT;
    TransportMemType outputType = TransportMemType::CCL_OUTPUT;

    CHK_RET(CalcCommPlaneInfo(tag_, commParaInfo, opTransport[COMM_COMBINE_ORDER], inputType, outputType));
    LevelNSubCommTransport &commTransport = opTransport[COMM_COMBINE_ORDER];
    for (u32 subCommIndex = 0; subCommIndex < commTransport.size(); subCommIndex++) {
        for (auto &transportRequest : commTransport[subCommIndex].transportRequests) {
            transportRequest.isUsedRdma = topoAttr_.isUsedRdmaMap.at(transportRequest.remoteUserRank);
        }
    }

    return HCCL_SUCCESS;
}

HcclResult CollAllGatherRingFor91093Executor::CalExchangeRemoteRank(u32 &remoteRankSend, u32 &remoteRankRecv)
{
    u32 userRank = topoAttr_.userRank;
    u32 userRankSize = topoAttr_.userRankSize;
    u32 l2Size = topoAttr_.superPodNum;
    CHK_PRT_RET(l2Size == 0,
            HCCL_ERROR("[CollAllGatherRingFor91093Executor][CalExchangeRemoteRank] invalid rank size, level2RankSize is 0"),
            HCCL_E_PARA);
    u32 l1Size = topoAttr_.serverNum / l2Size;
    CHK_PRT_RET(l1Size == 0,
            HCCL_ERROR("[CollAllGatherRingFor91093Executor][CalExchangeRemoteRank] invalid rank size, level1RankSize is 0"),
            HCCL_E_PARA);
    u32 l0Size = userRankSize / l1Size / l2Size;
    u32 l0Index = userRank % l0Size;
    u32 l1ServerIndex = userRank % (l0Size * l1Size) / l0Size;
    u32 l2ServerIndex = userRank / l0Size / l1Size;

    // 计算本端将要发送数据的目标rank
    remoteRankRecv = l0Index * l2Size * l1Size + l1ServerIndex * l2Size + l2ServerIndex;

    // 计算本端将要接收数据的目标rank
    u32 r0 = userRank / (l1Size * l2Size);
    u32 r1 = userRank % (l1Size * l2Size) / l2Size;
    u32 r2 = userRank % (l1Size * l2Size) % l2Size;
    remoteRankSend = r2 * l1Size * l0Size + r1 * l0Size + r0;

    return HCCL_SUCCESS;
}

HcclResult CollAllGatherRingFor91093Executor::GetTransport(u32 commIndex, u32 remoteUserRank, LINK &targetLink)
{
    CHK_PRT_RET(commIndex >= algResResp_->opTransportResponse[COMM_COMBINE_ORDER].size(),
        HCCL_ERROR("[CollAllGatherRingFor91093Executor][GetTransport] commIndex[%u] is larger than "\
        "opTransportResponse size[%zu]",
        remoteUserRank, algResResp_->opTransportResponse[COMM_COMBINE_ORDER].size()), HCCL_E_PARA);
    SingleSubCommTransport &commCombined =
        const_cast<SingleSubCommTransport&>(algResResp_->opTransportResponse[COMM_COMBINE_ORDER][commIndex]);
 
    CHK_PRT_RET(remoteUserRank >= commCombined.userRank2subCommRank.size(),
        HCCL_ERROR("[CollAllGatherRingFor91093Executor][GetTransport] remoteUserRank[%u] is larger than "\
        "userRank2subCommRank map size[%zu]",
        remoteUserRank, commCombined.userRank2subCommRank.size()), HCCL_E_PARA);
 
    u32 remoteRank = commCombined.userRank2subCommRank[remoteUserRank];
    CHK_PRT_RET(remoteRank >= commCombined.links.size(),
        HCCL_ERROR("[CollAllGatherRingFor91093Executor][GetTransport] remoteUserRank[%u], get remoteRank[%u]," \
        "the size of combinedComm links is [%zu]", remoteUserRank, remoteRank, commCombined.links.size()),
        HCCL_E_PARA);
    targetLink = commCombined.links[remoteRank];
    return HCCL_SUCCESS;
}

HcclResult CollAllGatherRingFor91093Executor::ExecuteBarrier(const std::shared_ptr<Transport> &preLink,
    const std::shared_ptr<Transport> &aftLink, Stream &stream)
{
    // 同步与preLink保证数据收发已结束
    CHK_RET(preLink->TxAck(stream));
 
    CHK_RET(aftLink->RxAck(stream));
 
    // 同步与aftLink保证数据收发已结束
    CHK_RET(aftLink->TxDataSignal(stream));
 
    CHK_RET(preLink->RxDataSignal(stream));
 
    return HCCL_SUCCESS;
}

bool CollAllGatherRingFor91093Executor::IsLevel0Neighbor(u32 remoteRank, u32 userRank)
{
    u32 Level0RankSize = topoAttr_.deviceNumPerAggregation;
    CHK_PRT_RET(Level0RankSize == 0,
            HCCL_ERROR("[CollAllGatherRingFor91093Executor][IsLevel0Neighbor] invalid rank size, Level0RankSize is 0"),
            HCCL_E_PARA);
    return (remoteRank / Level0RankSize == userRank / Level0RankSize)
            && (((userRank + 1) % Level0RankSize == remoteRank % Level0RankSize)
                || ((userRank + Level0RankSize - 1) % Level0RankSize == remoteRank % Level0RankSize));
}

HcclResult CollAllGatherRingFor91093Executor::ExchangeData(Stream &stream, const ExecMem &execMem, void *userInBase)
{
    u32 remoteRankSend = 0;
    u32 remoteRankRecv = 0;
    CHK_RET(CalExchangeRemoteRank(remoteRankSend, remoteRankRecv));
    u32 userRank = topoAttr_.userRank;

    u64 inputMemSize = execMem.inputMem.size();
    u32 userRankSize = topoAttr_.userRankSize;
    u32 level2RankSize = topoAttr_.superPodNum;
    CHK_PRT_RET(level2RankSize == 0,
            HCCL_ERROR("[CollAllGatherRingFor91093Executor][ExchangeData] invalid rank size, level2RankSize is 0"),
            HCCL_E_PARA);
    u32 level1RankSize = topoAttr_.serverNum / level2RankSize;
    CHK_PRT_RET(level1RankSize == 0,
            HCCL_ERROR("[CollAllGatherRingFor91093Executor][ExchangeData] invalid rank size, level1RankSize is 0"),
            HCCL_E_PARA);
    u32 level0RankSize = userRankSize / level1RankSize / level2RankSize;
    u32 localLevel1Index = userRank % (level0RankSize * level1RankSize) / level0RankSize;
    u32 localLevel2Index = userRank / level0RankSize / level1RankSize;

    if (remoteRankSend != userRank && remoteRankRecv != userRank) {
        if (!IsLevel0Neighbor(remoteRankSend, topoAttr_.userRank)) {
            // user in mem -> ccl in mem
            DeviceMem srcMem = DeviceMem::create(static_cast<u8 *>(execMem.inputPtr), inputMemSize);
            DeviceMem dstMem = execMem.inputMem.range(0, inputMemSize);
            HcclResult ret = HCCL_SUCCESS;
            ret = HcclD2DMemcpyAsync(dispatcher_, dstMem, srcMem, const_cast<Stream&>(stream));
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                        HCCL_ERROR("[CollAllGatherRingFor91093Executor][ExchangeData]user memcpy Failed, Offset[%llu], Size[%llu] ", 0, inputMemSize), ret);
        }

        HCCL_DEBUG("[CollAllGatherRingFor91093Executor][ExchangeData] localRank %d, remoteRankSend %d, remoteRankRecv %d, TxAsync: dst_offset %d, src %p, size %d; RxAsync src_offset 0, dst %p",
                    userRank, remoteRankSend, remoteRankRecv, userRank * inputMemSize, execMem.inputMem.ptr(), inputMemSize, static_cast<u8 *>(execMem.outputMem.ptr()) + remoteRankRecv * inputMemSize);
        LINK sendLink;
        LINK recvLink;
        CHK_RET(GetTransport(COMM_INDEX_0, remoteRankSend, sendLink));
        CHK_RET(GetTransport(COMM_INDEX_0, remoteRankRecv, recvLink));
        recvLink->TxAck(stream);
        sendLink->RxAck(stream);
        u32 remoteLevel1Index = remoteRankSend % (level0RankSize * level1RankSize) / level0RankSize;
        u32 remoteLevel2Index = remoteRankSend / level0RankSize / level1RankSize;   
        u64 txDstOffset = (remoteLevel1Index * level2RankSize + remoteLevel2Index) * inputMemSize;
        if (IsLevel0Neighbor(remoteRankSend, userRank)) {
            sendLink->TxAsync(UserMemType::OUTPUT_MEM, txDstOffset, execMem.inputPtr, inputMemSize, stream);
        } else {
            sendLink->TxAsync(UserMemType::OUTPUT_MEM, txDstOffset, execMem.inputMem.ptr(), inputMemSize, stream);
        }
        u64 rxDstOffset = (localLevel1Index * level2RankSize + localLevel2Index) * inputMemSize;
        u64 rxSrcOffset = IsLevel0Neighbor(remoteRankRecv, userRank) ? static_cast<u8 *>(execMem.inputPtr) - static_cast<u8 *>(userInBase) : 0;
        recvLink->RxAsync(UserMemType::INPUT_MEM, rxSrcOffset, static_cast<u8 *>(execMem.outputMem.ptr()) + rxDstOffset, inputMemSize, stream);
        CHK_RET(ExecuteBarrier(recvLink, sendLink, stream)); 
    } else {
        HcclResult ret = HCCL_SUCCESS;
        u64 dstMemOffset = (localLevel1Index * level2RankSize + localLevel2Index) * inputMemSize;
        DeviceMem dstMem = execMem.outputMem.range(dstMemOffset, inputMemSize);
        CHK_SMART_PTR_NULL(dstMem);
        if (!DMAReduceFlag_) {
            ret = HcclD2DMemcpyAsync(dispatcher_, dstMem, execMem.inputMem, const_cast<Stream&>(stream));
        } else {
            DeviceMem srcMem = DeviceMem::create(static_cast<u8 *>(execMem.inputPtr), inputMemSize);
            ret = HcclD2DMemcpyAsync(dispatcher_, dstMem, srcMem, const_cast<Stream&>(stream));
        }
        CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR("[CollAllGatherRingFor91093Executor][ExchangeData]all gather double "
                "ring user memcpy Failed, Offset[%llu], Size[%llu]", dstMemOffset, inputMemSize), ret);
    }
    return HCCL_SUCCESS;
}

HcclResult CollAllGatherRingFor91093Executor::KernelRun(const OpParam &param, ExecMem &execMem)
{
    HCCL_INFO("[CollAllGatherRingFor91093Executor][KernelRun] The AllGatherDoubleRingExecutor starts.");
    CHK_RET(ActiveSlaveStreams(param.stream));
    u32 perDataSize = 0;
    CHK_RET(SalGetDataTypeSize(param.DataDes.dataType, perDataSize));
    CHK_PRT_RET(perDataSize == 0,
        HCCL_ERROR("[CollAllGatherRingFor91093Executor][KernelRun]errNo[0x%016llx] datatype[%s] is invalid",
            HCCL_ERROR_CODE(HCCL_E_PARA), GetDataTypeEnumStr(param.DataDes.dataType).c_str()), HCCL_E_PARA);

    CHK_RET(CheckCommSize(COMM_LEVEL0, COMM_INDEX_0 + 1));
    SubCommInfo level0CommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);
    u32 level0ServerIndex = level0CommInfo.localRank;
    u32 level0RankSize = level0CommInfo.localRankSize;
    
    bool isSelectAHC = (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_AHC ||
        algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_AHC_BROKE);
    CommPlane commPlaneLevel1 = isSelectAHC ? COMM_LEVEL1_AHC : COMM_LEVEL1;
    CHK_RET(CheckCommSize(commPlaneLevel1, level0ServerIndex + 1));
    SubCommInfo level1CommInfo = GetSubCommInfo(commPlaneLevel1, level0ServerIndex);
    u32 level1ServerIndex = level1CommInfo.localRank;
    u32 level1RankSize = level1CommInfo.localRankSize;

    u32 level2RankSize ;//AHC bypass level2
    SubCommInfo level2CommInfo;
    if (isSelectAHC) {
        level2CommInfo = level1CommInfo;
        level2RankSize = 1;        
    } else {
        CHK_RET(CheckCommSize(COMM_LEVEL2, COMM_INDEX_0 + 1));
        level2CommInfo = GetSubCommInfo(COMM_LEVEL2, COMM_INDEX_0);
        level2RankSize = level2CommInfo.localRankSize;
    }

    //  第一步，将数据从input内存拷贝到output内存的对应位置
    u64 inputMemSize = execMem.inputMem.size();
    u64 dstMemOffset = topoAttr_.userRank * inputMemSize;
    DeviceMem dstMem = execMem.outputMem.range(dstMemOffset, inputMemSize);
    CHK_SMART_PTR_NULL(dstMem);

    HcomCollOpInfo opInfo = {
        "", execMem.inputPtr, execMem.outputPtr, param.DataDes.count, param.DataDes.dataType, 0, HCCL_REDUCE_RESERVED,
        param.DataDes.strideCount
    };
    HcomCollOpInfo graphModeOpInfo = {
        "", execMem.inputMem.ptr(), execMem.outputMem.ptr(), param.DataDes.count, param.DataDes.dataType, 0,
        HCCL_REDUCE_RESERVED, param.DataDes.strideCount
    };
    HcomCollOpInfo *opInfoPtr = nullptr;
    if (topoType_ == TopoType::TOPO_TYPE_NP_DOUBLE_RING) {
        opInfoPtr = &graphModeOpInfo;
    }

    // 图模式opinfo不为空，但需要将数据从ccl input拷贝到ccl output上
    HcclResult ret = HCCL_SUCCESS;
    if (!DMAReduceFlag_) {
        ret = HcclD2DMemcpyAsync(dispatcher_, dstMem, execMem.inputMem, const_cast<Stream&>(param.stream));
        CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR("[CollAllGatherRingFor91093Executor][KernelRun]all gather double "
                        "ring memcpy Failed, Offset[%llu], Size[%llu]", dstMemOffset, inputMemSize), ret);
    } else {
        opInfoPtr = &opInfo;
        // 先做server间算法，带有消减拷贝场景数据需要从user input取，拷贝到ccl output上
        if (level1RankSize > 1 || level2RankSize > 1) {
            DeviceMem srcMem = DeviceMem::create(static_cast<u8 *>(execMem.inputPtr), inputMemSize);
            ret = HcclD2DMemcpyAsync(dispatcher_, dstMem, srcMem, const_cast<Stream&>(param.stream));
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[CollAllGatherRingFor91093Executor][KernelRun]all gather double "
                    "ring user memcpy Failed, Offset[%llu], Size[%llu]", dstMemOffset, inputMemSize), ret);
        }
    }
    if (level2RankSize > 1) {
        std::unique_ptr<AlgTemplateBase> level2AGExecutor;
        if (algType_.algoLevel2 == AlgTypeLevel2::ALG_LEVEL2_NB) {
            level2AGExecutor = AlgTemplateRegistry::Instance().GetAlgTemplate(
                TemplateType::TEMPLATE_ALL_GATHER_NB, dispatcher_);
            HCCL_INFO("allgather ring: using nonuniform-bruck algo inter-superPod.");
        } else if (algType_.algoLevel2 == AlgTypeLevel2::ALG_LEVEL2_NHR) {
            level2AGExecutor = AlgTemplateRegistry::Instance().GetAlgTemplate(
                TemplateType::TEMPLATE_ALL_GATHER_NHR, dispatcher_);
            HCCL_INFO("allgather ring: using nonuniform-hierarchical-ring algo inter-superPod.");
        } else {
            level2AGExecutor = AlgTemplateRegistry::Instance().GetAlgTemplate(
                TemplateType::TEMPLATE_ALL_GATHER_RING, dispatcher_);
            HCCL_INFO("allgather ring: using ring algo inter-superPod.");
        }
        CHK_SMART_PTR_NULL(level2AGExecutor);

        std::vector<Slice> level2DataSegsSlice;
        for (u32 i = 0; i < level2RankSize; i++) {
            Slice sliceTemp;
            sliceTemp.size = inputMemSize;
            sliceTemp.offset = i * level1RankSize * level0RankSize * inputMemSize +
                (level1ServerIndex * level0RankSize + level0ServerIndex) * inputMemSize;
            level2DataSegsSlice.push_back(sliceTemp);
        }
        CHK_RET(level2AGExecutor->Prepare(execMem.outputMem, execMem.outputMem, execMem.inputMem, execMem.count,
            param.DataDes.dataType, param.stream,
            HCCL_REDUCE_RESERVED, INVALID_VALUE_RANKID, level2DataSegsSlice, 0));

        CHK_RET(level2AGExecutor->RegisterProfiler((
            level2RankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level2CommInfo.localRank,
            PROF_STAGE_0, HCCL_EXEC_STEP_NOT_SET, param.stream));

        CHK_RET(RunTemplate(level2AGExecutor, level2CommInfo));
        HCCL_INFO("allgather double ring [superpod] level2 allgather run success");
    }
    if (level1RankSize > 1) {
        // 计算slice, 不同超节点相同slice
        std::vector<Slice> level1DataSegsSlice;
        for (u32 j = 0; j < level1RankSize; j++) {
            for (u32 i = 0; i < level2RankSize; i++) {
                Slice level1Slice;
                level1Slice.size = inputMemSize;
                level1Slice.offset =
                    (j * level0RankSize +  i * level1RankSize * level0RankSize + level0ServerIndex) * inputMemSize;
                level1DataSegsSlice.push_back(level1Slice);
            }
        }

        if (GetExternalInputEnableRdmaSdmaConcurrent() && (inputMemSize >= HCCL_SPLIT_SIZE_INTER_SERVER) 
            && !aicpuUnfoldMode_) {
            u32 syncTrans = (topoType_ == TopoType::TOPO_TYPE_NP_DOUBLE_RING) ? BEST_SPLIT_VALUE_DR :
                BEST_SPLIT_VALUE_SR;
            CHK_RET(Level1AllGatherConcurrent(execMem.inputMem, execMem.outputMem, execMem.count, param.DataDes.dataType,
                param.stream, PROF_STAGE_1, level1DataSegsSlice, syncTrans));
        } else {
            std::unique_ptr<AlgTemplateBase> level1AGExecutor;
            if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_RING) {
                level1AGExecutor = AlgTemplateRegistry::Instance().GetAlgTemplate(
                    TemplateType::TEMPLATE_ALL_GATHER_RING, dispatcher_);
                HCCL_INFO("allgather ring: using ring algo inter-server.");
            } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NB) {
                level1AGExecutor = AlgTemplateRegistry::Instance().GetAlgTemplate(
                    TemplateType::TEMPLATE_ALL_GATHER_NB, dispatcher_);
                HCCL_INFO("allgather ring: using nonuniform-bruck algo inter-server.");
            } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NHR) {
                level1AGExecutor = AlgTemplateRegistry::Instance().GetAlgTemplate(
                    TemplateType::TEMPLATE_ALL_GATHER_NHR, dispatcher_);
                HCCL_INFO("allgather ring: using nonuniform-hierarchical-ring algo inter-server.");
            } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_AHC || algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_AHC_BROKE) {
                // 获取通信域分组信息
                std::vector<std::vector<std::vector<u32>>> globalSubGroups;
                std::map<AHCConcOpType, TemplateType> ahcAlgOption;
                CHK_RET(topoMatcher_->GetGlobalSubGroups(commPlaneLevel1, globalSubGroups));
                topoMatcher_->GetAHCAlgOption(ahcAlgOption);
                if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_AHC) {
                    level1AGExecutor = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_GATHER_AHC, dispatcher_);
                    HCCL_INFO("algather comm: using ahc algo inter-server.");
                } else {
                    level1AGExecutor = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_GATHER_AHC_BROKE, dispatcher_);
                    HCCL_INFO("algather comm: using ahc-broke algo inter-server.");
                }
                CHK_SMART_PTR_NULL(level1AGExecutor);
                CHK_RET(level1AGExecutor->Prepare(execMem.count, globalSubGroups, ahcAlgOption));
            } else {
                HCCL_ERROR("allgather ring: unsupported algtype [%s].", AlgTypeToStr(algType_).c_str());
                return HCCL_E_NOT_SUPPORT;
            }
            CHK_SMART_PTR_NULL(level1AGExecutor);
            CHK_RET(level1AGExecutor->Prepare(execMem.outputMem, execMem.outputMem, execMem.inputMem, execMem.count,
                param.DataDes.dataType, param.stream,
                HCCL_REDUCE_RESERVED, INVALID_VALUE_RANKID, level1DataSegsSlice, 0));

            CHK_RET(level1AGExecutor->RegisterProfiler((
                level1RankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level2CommInfo.localRank,
                PROF_STAGE_1, HCCL_EXEC_STEP_NOT_SET, param.stream));

            CHK_RET(RunTemplate(level1AGExecutor, level1CommInfo));
            HCCL_INFO("allgather double ring [superpod] level1 allgather run success");
        }
    }
    // 节点内做all gather double ring
    std::vector<Slice> dataSegsSlice;
    std::vector<std::vector<Slice>> multRingsSliceZero; // 数据基于该rank上环0的偏移
    CHK_RET(PrepareAllgatherSlice(level0RankSize, inputMemSize, dataSegsSlice));

    //  多环数据切分
    if (topoType_ == TopoType::TOPO_TYPE_NP_DOUBLE_RING &&
        !IsSupportUnifiedMarch(param, topoType_, topoAttr_.serverNum, topoAttr_.superPodNum)) {
        multRingsSliceZero = PrepareMultiRingSlice(dataSegsSlice, param.tag, false, topoAttr_.nicList);
    } else {
        multRingsSliceZero.push_back(dataSegsSlice);
    }
    std::vector<std::vector<Slice>> multRingsSlice;
    for (u32 ringIndex = 0; ringIndex < multRingsSliceZero.size(); ringIndex++) {
        std::vector<Slice> level2DataSlice;
        CHK_RET(CalculateLevel2AllgatherSlice(inputMemSize, level0RankSize, level1RankSize, level2RankSize,
            multRingsSliceZero, level2DataSlice, ringIndex));
        multRingsSlice.push_back(level2DataSlice);
    }

    CHK_PRT_RET(0 < param.DataDes.strideCount && param.DataDes.strideCount < param.DataDes.count,
        HCCL_ERROR("[CollAllGatherRingFor91093Executor][KernelRun]strideCount[%llu] is smaller than opCount[%llu]",
        param.DataDes.strideCount, param.DataDes.count),
        HCCL_E_PARA);
    HCCL_DEBUG("[CollAllGatherRingFor91093Executor][KernelRun]strideCount[%llu], opCount[%llu]",
        param.DataDes.strideCount, param.DataDes.count);
    std::vector<std::vector<Slice>> multRingsUserMemSlice;
    if (!DMAReduceFlag_) {
        multRingsUserMemSlice = multRingsSlice;
        // 图模式，根据strideCount更新slice的offset
        if (param.DataDes.strideCount != 0) {
            CHK_RET(UpdateOffsetBasedOnStrideCount(param, multRingsUserMemSlice));
        }
    } else {
        for (u32 ringIndex = 0; ringIndex < multRingsSlice.size(); ringIndex++) {
            std::vector<Slice> userMemSlice;
            for (auto &cclSlice : multRingsSlice[ringIndex]) {
                Slice tmpSlice;
                u64 count = (param.DataDes.strideCount == 0) ? param.DataDes.count : param.DataDes.strideCount;
                tmpSlice.size = cclSlice.size;
                tmpSlice.offset =
                    (cclSlice.offset / inputMemSize) * count * perDataSize +
                    multRingsSliceZero[ringIndex][0].offset;
                userMemSlice.push_back(tmpSlice);
                HCCL_DEBUG("rank[%u], ringIndex[%u], tmpSlice.offset=[%llu], size=[%llu]",
                    topoAttr_.userRank, ringIndex, tmpSlice.offset, tmpSlice.size);
            }
            multRingsUserMemSlice.push_back(userMemSlice);
        }
    }
    if (DMAReduceFlag_ && (level1RankSize > 1 || level2RankSize > 1)) {
        // allgather输入放在CCL buffer上，通过设置nullptr指示要从CCL buffer获取输入
        opInfo.inputAddr = nullptr;
    }
    CHK_RET(RunIntraSeverAllGather(param.tag, execMem.inputMem, execMem.outputMem, execMem.count,
        param.DataDes.dataType, multRingsSlice, param.stream, PROF_STAGE_2, 0, opInfoPtr, multRingsUserMemSlice));
    HCCL_INFO("allgather double ring run success");
    return HCCL_SUCCESS;
}

HcclResult CollAllGatherRingFor91093Executor::KernelRunInterServer(const OpParam &param, ExecMem &execMem)
{
    HCCL_INFO("[CollAllGatherRingFor91093Executor][KernelRunInterServer] The AllGatherDoubleRingExecutor starts.");
    CHK_RET(ActiveSlaveStreams(param.stream));
    u32 perDataSize = 0;
    CHK_RET(SalGetDataTypeSize(param.DataDes.dataType, perDataSize));
    CHK_PRT_RET(perDataSize == 0,
        HCCL_ERROR("[CollAllGatherRingFor91093Executor][KernelRunInterServer]errNo[0x%016llx] datatype[%s] is invalid",
            HCCL_ERROR_CODE(HCCL_E_PARA), GetDataTypeEnumStr(param.DataDes.dataType).c_str()), HCCL_E_PARA);

    CHK_RET(CheckCommSize(COMM_LEVEL0, COMM_INDEX_0 + 1));
    SubCommInfo level0CommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);
    u32 level0ServerIndex = level0CommInfo.localRank;
    CHK_RET(CheckCommSize(COMM_LEVEL1, level0ServerIndex + 1));
    SubCommInfo level1CommInfo = GetSubCommInfo(COMM_LEVEL1, level0ServerIndex);
    CHK_RET(CheckCommSize(COMM_LEVEL2, COMM_INDEX_0 + 1));
    SubCommInfo level2CommInfo = GetSubCommInfo(COMM_LEVEL2, COMM_INDEX_0);

    //  第一步，将数据从input内存拷贝到output内存的对应位置
    u32 level1ServerIndex = level1CommInfo.localRank;
    u32 level1RankSize = level1CommInfo.localRankSize;
    u32 level2RankSize = level2CommInfo.localRankSize;

    u64 inputMemSize = execMem.inputMem.size();

    // exchange data between ranks to be prepared for continuous copying
    Stream stream = param.stream;
    if (level1RankSize > 1 || level2RankSize > 1) {
        CHK_RET(ExchangeData(stream, execMem, param.inputPtr));
    }

    if (level2RankSize > 1) {
        std::unique_ptr<AlgTemplateBase> level2AGExecutor;
        if (algType_.algoLevel2 == AlgTypeLevel2::ALG_LEVEL2_NB) {
            level2AGExecutor = AlgTemplateRegistry::Instance().GetAlgTemplate(
                TemplateType::TEMPLATE_ALL_GATHER_NB, dispatcher_);
            HCCL_INFO("allgather ring: using nonuniform-bruck algo inter-superPod.");
        } else if (algType_.algoLevel2 == AlgTypeLevel2::ALG_LEVEL2_NHR) {
            level2AGExecutor = AlgTemplateRegistry::Instance().GetAlgTemplate(
                TemplateType::TEMPLATE_ALL_GATHER_NHR, dispatcher_);
            HCCL_INFO("allgather ring: using nonuniform-hierarchical-ring algo inter-superPod.");
        } else {
            level2AGExecutor = AlgTemplateRegistry::Instance().GetAlgTemplate(
                TemplateType::TEMPLATE_ALL_GATHER_RING, dispatcher_);
            HCCL_INFO("allgather ring: using ring algo inter-superPod.");
        }
        CHK_SMART_PTR_NULL(level2AGExecutor);

        std::vector<Slice> level2DataSegsSlice;
        for (u32 i = 0; i < level2RankSize; i++) {
            u32 remoteRankRecv = level1ServerIndex * level2RankSize + i;
            u32 groupId = remoteRankRecv / level2RankSize;
            Slice sliceTemp;
            sliceTemp.size = inputMemSize;
            sliceTemp.offset = (groupId * level2RankSize + i) * inputMemSize;
            level2DataSegsSlice.push_back(sliceTemp);
        }
        CHK_RET(level2AGExecutor->Prepare(execMem.outputMem, execMem.outputMem, execMem.inputMem, execMem.count,
            param.DataDes.dataType, param.stream,
            HCCL_REDUCE_RESERVED, INVALID_VALUE_RANKID, level2DataSegsSlice, 0));

        CHK_RET(level2AGExecutor->RegisterProfiler((
            level2RankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level2CommInfo.localRank,
            PROF_STAGE_0, HCCL_EXEC_STEP_NOT_SET, param.stream));

        CHK_RET(RunTemplate(level2AGExecutor, level2CommInfo));
        HCCL_INFO("allgather double ring [superpod] level2 allgather run success");
    }
    if (level1RankSize > 1) {
        // 计算slice, 不同超节点相同slice
        std::vector<Slice> level1DataSegsSlice;
        u64 stride = inputMemSize * level2RankSize;
        for (u32 i = 0; i < level1RankSize; i++) {
            Slice level1Slice;
            level1Slice.size = stride;
            level1Slice.offset = i * stride;
            level1DataSegsSlice.push_back(level1Slice);
        }

        std::unique_ptr<AlgTemplateBase> level1AGExecutor;
        if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_RING) {
            level1AGExecutor = AlgTemplateRegistry::Instance().GetAlgTemplate(
                TemplateType::TEMPLATE_ALL_GATHER_RING, dispatcher_);
            HCCL_INFO("allgather ring: using ring algo inter-server.");
        } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NB) {
            level1AGExecutor = AlgTemplateRegistry::Instance().GetAlgTemplate(
                TemplateType::TEMPLATE_ALL_GATHER_NB, dispatcher_);
            HCCL_INFO("allgather ring: using nonuniform-bruck algo inter-server.");
        } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NHR) {
            level1AGExecutor = AlgTemplateRegistry::Instance().GetAlgTemplate(
                TemplateType::TEMPLATE_ALL_GATHER_NHR, dispatcher_);
            HCCL_INFO("allgather ring: using nonuniform-hierarchical-ring algo inter-server.");
        } else {
            HCCL_ERROR("allgather ring: unsupported algtype [%s].", AlgTypeToStr(algType_).c_str());
            return HCCL_E_NOT_SUPPORT;
        }
        CHK_SMART_PTR_NULL(level1AGExecutor);
        CHK_RET(level1AGExecutor->Prepare(execMem.outputMem, execMem.outputMem, execMem.inputMem, execMem.count,
            param.DataDes.dataType, param.stream,
            HCCL_REDUCE_RESERVED, INVALID_VALUE_RANKID, level1DataSegsSlice, 0));

        CHK_RET(level1AGExecutor->RegisterProfiler((
            level1RankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level2CommInfo.localRank,
            PROF_STAGE_1, HCCL_EXEC_STEP_NOT_SET, param.stream));

        CHK_RET(RunTemplate(level1AGExecutor, level1CommInfo));
        HCCL_INFO("allgather double ring [superpod] level1 allgather run success");
    }

    if (level1RankSize > 1 || level2RankSize > 1) {
        u32 unitSize = SIZE_TABLE[param.DataDes.dataType];
        u64 curSize = execMem.inputMem.size();
        for (u32 i = 0; i < level1RankSize * level2RankSize; i++) {
            DeviceMem dstMem = DeviceMem::create(static_cast<u8 *>(execMem.outputPtr) + param.DataDes.count * unitSize * (level0ServerIndex * level1RankSize * level2RankSize + i), curSize);
            DeviceMem srcMem = DeviceMem::create(static_cast<u8 *>(execMem.outputMem.ptr()) + i * curSize, curSize);
            CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dstMem, srcMem, const_cast<Stream&>(param.stream)));
        }
    }
    return HCCL_SUCCESS;
}


HcclResult CollAllGatherRingFor91093Executor::KernelRunIntraServer(const OpParam &param, ExecMem &execMem)
{
    HCCL_INFO("[CollAllGatherRingFor91093Executor][KernelRunIntraServer] The AllGatherDoubleRingExecutor starts.");
    CHK_RET(ActiveSlaveStreams(param.stream));
    u32 perDataSize = 0;
    CHK_RET(SalGetDataTypeSize(param.DataDes.dataType, perDataSize));
    CHK_PRT_RET(perDataSize == 0,
        HCCL_ERROR("[CollAllGatherRingFor91093Executor][KernelRunIntraServer]errNo[0x%016llx] datatype[%s] is invalid",
            HCCL_ERROR_CODE(HCCL_E_PARA), GetDataTypeEnumStr(param.DataDes.dataType).c_str()), HCCL_E_PARA);

    CHK_RET(CheckCommSize(COMM_LEVEL0, COMM_INDEX_0 + 1));
    SubCommInfo level0CommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);
    u32 level0ServerIndex = level0CommInfo.localRank;
    CHK_RET(CheckCommSize(COMM_LEVEL1, level0ServerIndex + 1));
    SubCommInfo level1CommInfo = GetSubCommInfo(COMM_LEVEL1, level0ServerIndex);
    CHK_RET(CheckCommSize(COMM_LEVEL2, COMM_INDEX_0 + 1));
    SubCommInfo level2CommInfo = GetSubCommInfo(COMM_LEVEL2, COMM_INDEX_0);

    //  第一步，将数据从input内存拷贝到output内存的对应位置
    u32 level0RankSize = level0CommInfo.localRankSize;
    u32 level1RankSize = level1CommInfo.localRankSize;
    u32 level2RankSize = level2CommInfo.localRankSize;
    u64 inputMemSize = execMem.inputMem.size();

    HcomCollOpInfo opInfo = {
        "", execMem.inputPtr, execMem.outputPtr, param.DataDes.count, param.DataDes.dataType, 0, HCCL_REDUCE_RESERVED,
        param.DataDes.strideCount
    };
    HcomCollOpInfo graphModeOpInfo = {
        "", execMem.inputMem.ptr(), execMem.outputMem.ptr(), param.DataDes.count, param.DataDes.dataType, 0,
        HCCL_REDUCE_RESERVED, param.DataDes.strideCount
    };
    HcomCollOpInfo *opInfoPtr = nullptr;
    if (topoType_ == TopoType::TOPO_TYPE_NP_DOUBLE_RING) {
        opInfoPtr = &graphModeOpInfo;
    }

    if (DMAReduceFlag_) {
        opInfoPtr = &opInfo;
    }
    
    // 节点内做all gather double ring
    std::vector<Slice> dataSegsSlice;
    std::vector<std::vector<Slice>> multRingsSliceZero; // 数据基于该rank上环0的偏移
    CHK_RET(PrepareAllgatherSlice(level0RankSize, inputMemSize * level2RankSize * level1RankSize, dataSegsSlice));

    //  多环数据切分
    if (topoType_ == TopoType::TOPO_TYPE_NP_DOUBLE_RING &&
        !IsSupportUnifiedMarch(param, topoType_, topoAttr_.serverNum, topoAttr_.superPodNum)) {
        multRingsSliceZero = PrepareMultiRingSlice(dataSegsSlice, param.tag, false, topoAttr_.nicList);
    } else {
        multRingsSliceZero.push_back(dataSegsSlice);
    }
    std::vector<std::vector<Slice>> multRingsSlice;
    for (u32 ringIndex = 0; ringIndex < multRingsSliceZero.size(); ringIndex++) {
        std::vector<Slice> level2DataSlice;
        level2DataSlice = multRingsSliceZero[ringIndex];
        multRingsSlice.push_back(level2DataSlice);
    }

    CHK_PRT_RET(0 < param.DataDes.strideCount && param.DataDes.strideCount < param.DataDes.count,
        HCCL_ERROR("[CollAllGatherRingFor91093Executor][KernelRunIntraServer]strideCount[%llu] is smaller than opCount[%llu]",
        param.DataDes.strideCount, param.DataDes.count),
        HCCL_E_PARA);
    std::vector<std::vector<Slice>> multRingsUserMemSlice;
    if (!DMAReduceFlag_) {
        multRingsUserMemSlice = multRingsSlice;
        // 图模式，根据strideCount更新slice的offset
        if (param.DataDes.strideCount != 0) {
            CHK_RET(UpdateOffsetBasedOnStrideCount(param, multRingsUserMemSlice));
        }
    } else {
        for (u32 ringIndex = 0; ringIndex < multRingsSlice.size(); ringIndex++) {
            std::vector<Slice> userMemSlice;
            for (auto &cclSlice : multRingsSlice[ringIndex]) {
                Slice tmpSlice;
                u64 count = (param.DataDes.strideCount == 0) ? param.DataDes.count : param.DataDes.strideCount;
                tmpSlice.size = cclSlice.size;
                tmpSlice.offset =
                    (cclSlice.offset / (inputMemSize * level1RankSize * level2RankSize)) * (count * level1RankSize * level2RankSize) * perDataSize +
                    multRingsSliceZero[ringIndex][0].offset;
                userMemSlice.push_back(tmpSlice);
                HCCL_DEBUG("rank[%u], ringIndex[%u], tmpSlice.offset=[%llu], size=[%llu], param.DataDes.strideCount=%d, param.DataDes.count=%d, cclSlice.offset=%d, inputMemSize=%d",
                    topoAttr_.userRank, ringIndex, tmpSlice.offset, tmpSlice.size, param.DataDes.strideCount, param.DataDes.count, cclSlice.offset, inputMemSize);
            }
            multRingsUserMemSlice.push_back(userMemSlice);
        }
    }
    if (DMAReduceFlag_ && (level1RankSize > 1 || level2RankSize > 1)) {
        // allgather输入放在CCL buffer上，通过设置nullptr指示要从CCL buffer获取输入
        opInfo.inputAddr = nullptr;
    }
    CHK_RET(RunIntraSeverAllGather(param.tag, execMem.inputMem, execMem.outputMem, execMem.count,
        param.DataDes.dataType, multRingsSlice, param.stream, PROF_STAGE_2, 0, opInfoPtr, multRingsUserMemSlice));
    return HCCL_SUCCESS;
}

REGISTER_EXEC("AllGatherRingFor91093Executor", AllGatherRingFor91093, CollAllGatherRingFor91093Executor);

} // namespace hccl
