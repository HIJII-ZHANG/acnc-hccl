#include "coll_all_gather_striped_pipeline.h"

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
        return UINT64_MAX;
    }
    HcclResult CollAllGatherNewExecutor::KernelRun(const OpParam &param, ExecMem &execMem)
    {
        HCCL_INFO("[CollAllGatherNewExecutor][KernelRun] algorithm start");

        constexpr size_t kPlaneNum = 8;        // 7 HCCS + 1 RoCE
        // —— 把基类 RunContext 拆解后直接调用模板 ——
        std::array<SubCommInfo,kPlaneNum> subInfo{};
        CHK_RET(CheckCommSize(COMM_LEVEL0, COMM_INDEX_0 + 1));
        SubCommInfo level0 = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);
        u32 commIndex = level0.localRank;                 // Mesh 里就用这个索引到 Level-1
        CHK_RET(CheckCommSize(COMM_LEVEL1, commIndex + 1));
        SubCommInfo level1 = GetSubCommInfo(COMM_LEVEL1, commIndex);

        // 先把 8 个 plane 都填成同一个 CommInfo（确保能跑起来；之后再按真实 NIC 拆分）
        for (size_t i=0; i<kPlaneNum; ++i) subInfo[i] = level1;

        auto &slaveStreams = algResResp_->slaveStreams;
        auto &notifiesMain = algResResp_->notifiesMain;
        auto &notifiesAux  = algResResp_->notifiesAux;

        // 模板实例
        std::unique_ptr<AlgTemplateBase> tmpl =
            AlgTemplateRegistry::Instance().GetAlgTemplate(
                TemplateType::TEMPLATE_ALL_GATHER_STRIPED_PIPELINE, dispatcher_);
        CHK_SMART_PTR_NULL(tmpl);

        auto *bptr = dynamic_cast<AllGatherStripedPipelineTemplateBase *>(tmpl.get());
        CHK_SMART_PTR_NULL(bptr);

        // 准备并运行
        CHK_RET(bptr->Prepare(execMem.inputMem, execMem.outputMem, execMem.count, GetDataType(param),
                            param.stream, slaveStreams, notifiesMain, notifiesAux,
                            topoAttr_.userRank, topoAttr_.userRankSize, /*localHop=*/7, subInfo));

        CHK_RET(ActiveSlaveStreams(param.stream));
        CHK_RET(bptr->RunAsync());

        HCCL_INFO("[CollAllGatherNewExecutor][KernelRun] striped-pipeline done");
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

    REGISTER_EXEC("AllGatherNewExecutor", AllGatherNew, CollAllGatherNewExecutor);
} // namespace hccl
