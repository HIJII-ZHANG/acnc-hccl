#include "coll_all_gather_striped_pipeline.h"
#include "all_gather_striped_pipeline_pub.h"
#include "hccl/adapter_rts_common.h"

namespace hccl {
    CollAllGatherNewExecutor::CollAllGatherNewExecutor(const HcclDispatcher dispatcher,
        std::unique_ptr<TopoMatcher> &topoMatcher)
        : CollAllGatherExecutor(dispatcher, topoMatcher)
    {
        HCCL_INFO("[CollAllGatherNewExecutor] constructor called");
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
        TransportMemType in, out;
        CHK_RET(CalcTransportMemType(in, out));
        CHK_RET(CalcLevel0CommInfo(in, out, opTransport));
        CHK_RET(CalcLevel1CommInfo(in, out, opTransport));
        return HCCL_SUCCESS;
    }

    HcclResult CollAllGatherNewExecutor::CalcLevel0CommInfo(TransportMemType inputType,
        TransportMemType outputType,
        std::vector<LevelNSubCommTransport>& opTransport)
    {
        CommParaInfo commParaLevel0(COMM_LEVEL0, CommType::COMM_TAG_RING_INNER);
        CHK_RET(CalcCommPlaneInfo(tag_, commParaLevel0, opTransport[COMM_LEVEL0], inputType, outputType));
        return HCCL_SUCCESS;
    }
    HcclResult CollAllGatherNewExecutor::CalcTransportMemType(TransportMemType &inputType, TransportMemType &outputType)
    {
        inputType = TransportMemType::CCL_INPUT;
        outputType = TransportMemType::CCL_OUTPUT;
        HCCL_INFO("[CollAllGatherNewExecutor][CalcTransportMemType] tag[%s] inputType[%d], outputType[%d]",
            tag_.c_str(), inputType, outputType);
        return HCCL_SUCCESS;
    }
    u64 CollAllGatherNewExecutor::CalcLoopMaxCount(const u64 cclBuffSize, const u32 unitSize)
    {
        return (cclBuffSize / topoAttr_.userRankSize / HCCL_MIN_SLICE_ALIGN) * HCCL_MIN_SLICE_ALIGN / unitSize;
        //u64 perLoop = cclBuffSize / topoAttr_.userRankSize / HCCL_MIN_SLICE_ALIGN
        //          * HCCL_MIN_SLICE_ALIGN / unitSize;
        //return std::max<u64>(perLoop, 1);
    }

    HcclResult CollAllGatherNewExecutor::Orchestrate(OpParam &param, AlgResourceResponse &algRes)
    {
        HCCL_INFO("[AllGatherStripedPipeline][Orchestrate] start");

            HcclUs start = TIME_NOW();
    tag_ = param.tag;
    algResResp_ = &algRes;

    ExecMem m;
    m.count     = param.DataDes.count;
    m.inputPtr  = param.inputPtr;
    m.outputPtr = param.outputPtr;

    // 输入内存选择逻辑：与现有模板保持一致
    m.inputMem  = algRes.cclInputMem;     // 需要 CCL 提供的 staging buf 的场景

    // 输出走 AIV/CCL 管理的输出缓冲
    m.outputMem = algRes.cclOutputMem;

    // 【方案A关键点】如果需要预拷贝，也只在同一条通信流上做
    // 多数情况下 m.inputMem 已指到用户 buffer 的包装，预拷贝可以是 no-op
    HcclResult ret = PreCopyToCclUsingCommStream(param, m);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[CollAllGatherNewExecutor][Orchestrate] pre-copy failed, tag[%s]", tag_.c_str()), ret);

    // KernelRun 内部所有操作（包含可能的 D2D memcpy 到输出布局）也走 param.stream
    ret = KernelRun(param, m);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[CollAllGatherNewExecutor][Orchestrate] kernel run failed, tag[%s], err[0x%016llx]",
            tag_.c_str(), HCCL_ERROR_CODE(ret)), ret);

    HCCL_INFO("tag[%s], AllGather striped-pipeline orchestrate success, time [%lld]us.",
              tag_.c_str(), DURATION_US(TIME_NOW() - start));
    return HCCL_SUCCESS;
    }

HcclResult CollAllGatherNewExecutor::PreCopyToCclUsingCommStream(const OpParam& param, ExecMem& m)
{
    // 如果 m.inputMem 已经代表用户数据的 device 内存包装（最常见），这里不需要任何操作
    if (m.inputMem.size() == 0 || m.inputPtr == nullptr) {
        // 没有可拷贝的数据或无需预拷贝
        return HCCL_SUCCESS;
    }

    // 根据你的工程约定：当 workflowMode 是 OP_BASE 才需要把用户数据拷到 CCL staging
        // 将用户 device 指针视作源，按 m.inputMem 的大小做一次 D2D 拷贝到 CCL 输入缓冲
        const u64 copyBytes = m.inputMem.size();
        DeviceMem dst = m.inputMem.range(0, copyBytes);
        CHK_SMART_PTR_NULL(dst);

        // 将用户指针包装成 DeviceMem（如果已有帮助函数就用它；没有就改成工程已有的包装方式）
        DeviceMem src(reinterpret_cast<uint8_t*>(m.inputPtr), copyBytes);

        HcclResult ret = HcclD2DMemcpyAsync(dispatcher_, dst, src, const_cast<Stream&>(param.stream));
        CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR("[CollAllGatherNewExecutor][PreCopy] memcpy to CCL input failed, bytes[%llu]", copyBytes), ret);

    // 不做任何跨流事件/等待：后续 KernelRun 仍在 param.stream 顺序执行
    return HCCL_SUCCESS;
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
    HcclResult CollAllGatherNewExecutor::Getlevel1CommRank(SubCommInfo& level1CommInfo)
    {
        return HCCL_E_UNAVAIL;
    }
    HcclResult CollAllGatherNewExecutor::SelectTempAlg(std::unique_ptr<AlgTemplateBase> &level1TempAlg, u32 level1RankSize)
    {
        return HCCL_E_UNAVAIL;
    }

    REGISTER_EXEC("CollAllGatherNewExecutor", AllGatherNew, CollAllGatherNewExecutor);
} // namespace hccl
