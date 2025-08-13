#include "coll_all_gather_striped_pipeline.h"
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
        streamNum = 8 - 1;  // 7 HCCS + 1 RoCE
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
        for (size_t i = 0; i < kPlaneNum; ++i) commPlanes[i] = level1;

        // —— 取得模板，按“你给的 Prepare 签名”调用 —— 
        auto &slaveStreams = algResResp_->slaveStreams;
        auto &notifiesMain = algResResp_->notifiesMain;
        auto &notifiesAux  = algResResp_->notifiesAux;

        std::unique_ptr<AlgTemplateBase> basePtr =
            AlgTemplateRegistry::Instance().GetAlgTemplate(
                TemplateType::TEMPLATE_ALL_GATHER_STRIPED_PIPELINE, dispatcher_);
        CHK_SMART_PTR_NULL(basePtr);
        auto *tmpl = dynamic_cast<AllGatherStripedPipeline*>(basePtr.get());
        CHK_SMART_PTR_NULL(tmpl);

        // 只要把 L1（跨节点 UB 子平面）塞给模板，模板就会在 7 个平面上并行环传
        CHK_RET(tmpl->Prepare(execMem.inputMem, execMem.outputMem, execMem.count, dtype,
                            param.stream, slaveStreams, notifiesMain, notifiesAux,
                            topoAttr_.userRank, topoAttr_.userRankSize,
                            /*localHop*/ (u32)(topoAttr_.deviceNumPerAggregation - 1),
                            commPlanes));

        CHK_RET(ActiveSlaveStreams(param.stream));
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
