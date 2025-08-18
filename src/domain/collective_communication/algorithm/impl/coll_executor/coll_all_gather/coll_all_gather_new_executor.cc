
/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "coll_all_gather_new_executor.h"
#include "all_gather_striped_pipeline_pub.h"

namespace hccl {
CollAllGatherNewExecutor::CollAllGatherNewExecutor(const HcclDispatcher dispatcher,
    std::unique_ptr<TopoMatcher> &topoMatcher)
    : CollAllGatherExecutor(dispatcher, topoMatcher)
{
    DMAReduceFlag_ = false;
}

HcclResult CollAllGatherNewExecutor::CalcStreamNum(u32& streamNum)
{
    u32 totalStreamNum = topoAttr_.deviceNumPerAggregation > 1U ? topoAttr_.deviceNumPerAggregation - 1U : 1U;
    streamNum = totalStreamNum - 1U;
    HCCL_INFO("[CollAllGatherMeshExecutor][CalcStreamNum] tag[%s] streamNum[%u]",
        tag_.c_str(), streamNum);
    return HCCL_SUCCESS;
}

HcclResult CollAllGatherNewExecutor::CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport)
{
    TransportMemType inputType = TransportMemType::CCL_INPUT;
    TransportMemType outputType = TransportMemType::CCL_OUTPUT;
    CHK_RET(CalcLevel0CommInfo(inputType, outputType, opTransport));
    CHK_RET(CalcLevel1CommInfo(inputType, outputType, opTransport));
    return HCCL_SUCCESS;
}

HcclResult CollAllGatherNewExecutor::CalcLevel0CommInfo(TransportMemType inputType, TransportMemType outputType,
    std::vector<LevelNSubCommTransport>& opTransport)
{
    CommParaInfo commParaLevel0(COMM_LEVEL0, CommType::COMM_TAG_MESH);
    CHK_RET(CalcCommPlaneInfo(tag_, commParaLevel0, opTransport[COMM_LEVEL0], inputType, outputType));
    return HCCL_SUCCESS;
}

//根据 CCL 缓冲区大小和数据单元大小，计算 AllGather 操作在单次循环中可处理的最大元素数量，确保数据传输不超过预分配的缓冲区容量。
//HCCL_MIN_SLICE_ALIGN是 HCCL 定义的最小数据分片对齐值
u64 CollAllGatherNewExecutor::CalcLoopMaxCount(const u64 cclBuffSize, const u32 unitSize)
{
    u64 maxCountPerLoop = cclBuffSize / topoAttr_.userRankSize / HCCL_MIN_SLICE_ALIGN
        * HCCL_MIN_SLICE_ALIGN / unitSize;
    return maxCountPerLoop;
}

    HcclResult CollAllGatherNewExecutor::KernelRun(const OpParam &param, ExecMem &execMem)
    {
        HCCL_INFO("[CollAllGatherNewExecutor][KernelRun] algorithm start");

        const HcclDataType dtype = GetDataType(param);
        const u64 inputMemSize   = execMem.inputMem.size();
        const u64 bytesPerRank   = inputMemSize;  // AllGather：每 rank 等长

        // —— 取 L0：用来确定 commIndex（与 Mesh 一致）——
        CHK_RET(CheckCommSize(COMM_LEVEL0, COMM_INDEX_0 + 1));
        SubCommInfo level0 = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);
        u32 commIndex      = level0.localRank;
        HCCL_INFO("[StripedPipeline] L0 rankSize=%u localRank=%u", level0.localRankSize, level0.localRank);

        // —— 取 L1：跨节点子平面（TopoMatcher 需返回 UB 子平面的 SubCommInfo）——
        CHK_RET(CheckCommSize(COMM_LEVEL1, commIndex + 1));
        SubCommInfo level1 = GetSubCommInfo(COMM_LEVEL1, commIndex);
        HCCL_INFO("[StripedPipeline] L1 rankSize=%u localRank=%u", level1.localRankSize, level1.localRank);

        // —— 准备 7 个子平面的 SubCommInfo（当前先全部复用同一份 L1；若 TopoMatcher 已能按子平面拆分，替换为 7 份即可）——
        constexpr size_t kPlaneNum = AllGatherStripedPipeline::kPlaneNum;
        std::array<SubCommInfo, kPlaneNum> commPlanes{};
        for (size_t p = 0; p < kPlaneNum; ++p) {
            if (CheckCommSize(COMM_LEVEL1, p + 1) == HCCL_SUCCESS) {
                commPlanes[p] = GetSubCommInfo(COMM_LEVEL1, p);
            } else {
                // 兜底：还没拆成7份就复用 commIndex 对应的那份
                commPlanes[p] = GetSubCommInfo(COMM_LEVEL1, commIndex);
            }
        }


        //debug
        auto dumpLinks = [](const SubCommInfo& ci, const char* tag) {
            size_t n = std::min<size_t>(ci.links.size(), 4);
            for (size_t i = 0; i < n; ++i) {
                auto &lk = ci.links[i];
                LinkType lt = lk->GetLinkType();
                u32 peer = lk->GetRemoteRank();                   // ← 这里替换
                HCCL_INFO("[L1/%s] link[%zu] type=%d remoteRank=%u", tag, i, (int)lt, peer);
            }
        };
        for (size_t p = 0; p < kPlaneNum; ++p) {
            dumpLinks(commPlanes[p], std::to_string(p).c_str());
        }


        // —— 取得模板，按“你给的 Prepare 签名”调用 —— 
        auto &slaveStreams = algResResp_->slaveStreams;
        auto &notifiesMain = algResResp_->notifiesMain;
        auto &notifiesAux  = algResResp_->notifiesAux;

        std::unique_ptr<AlgTemplateBase> basePtr =
            AlgTemplateRegistry::Instance().GetAlgTemplate(
                TemplateType::TEMPLATE_ALL_GATHER_STRIPED_PIPELINE, dispatcher_);
        CHK_SMART_PTR_NULL(basePtr);

        // debug
        if(!basePtr) {
            HCCL_ERROR("[CollAllGatherNewExecutor][KernelRun] failed to get alg template");
            return HCCL_E_UNAVAIL;
        }

        auto *tmpl = dynamic_cast<AllGatherStripedPipeline*>(basePtr.get());
        CHK_SMART_PTR_NULL(tmpl);

        //debug
        if(!tmpl) {
            HCCL_ERROR("[CollAllGatherNewExecutor][KernelRun] failed to get alg template");
            return HCCL_E_UNAVAIL;
        }
        HCCL_INFO("[StripedPipeline][KernelRun] start tag[%s] rank[%u/%u] slaveStreams=%zu",
          tag_.c_str(), topoAttr_.userRank, topoAttr_.userRankSize, algResResp_->slaveStreams.size());


        // 只要把 L1（跨节点 UB 子平面）塞给模板，模板就会在 7 个平面上并行环传
        CHK_RET(tmpl->Prepare(execMem.inputMem, execMem.outputMem, execMem.count, dtype,
                            param.stream, slaveStreams, notifiesMain, notifiesAux,
                            topoAttr_.userRank, topoAttr_.userRankSize,
                            /*localHop*/ (u32)(topoAttr_.deviceNumPerAggregation - 1),
                            commPlanes));

        HCCL_INFO("[StripedPipeline][KernelRun] Prepare done");

        CHK_RET(ActiveSlaveStreams(param.stream));

        HCCL_INFO("[StripedPipeline][KernelRun] ActiveSlaveStreams done");

        CHK_RET(tmpl->RunAsync());

        HCCL_INFO("[StripedPipeline][KernelRun] done");
        return HCCL_SUCCESS;
    }

REGISTER_EXEC("CollAllGatherNewExecutor", AllGatherNew, CollAllGatherNewExecutor);
} // namespace hccl
