#include "coll_all_gather_striped_pipeline.h"

namespace hccl {
    CollAllGatherNewExecutor::CollAllGatherNewExecutor(const HcclDispatcher dispatcher,
        std::unique_ptr<TopoMatcher> &topoMatcher)
        : CollAllGatherExecutor(dispatcher, topoMatcher)
    {
        DMAReduceFlag_ = false;
    }
    HcclResult CollAllGatherNewExecutor::CalcStreamNum(u32& streamNum) override
    {
        streamNum = 8 - 1;  // 7 HCCS + 1 RoCE
        return HCCL_SUCCESS;
    }
    HcclResult CollAllGatherNewExecutor::CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport) override
    {
        TransportMemType in, out;
        CHK_RET(CalcTransportMemType(in, out));
        // 将 plane0‑6(C‑tag HCCS) & plane7(C‑tag RoCE) 作为 Level‑1 的 8 个子通信域
        for(int p=0;p<8;++p){
            CommParaInfo paraLvl1(COMM_LEVEL1, CommType::COMM_TAG_RING_INNER);
            paraLvl1.planeId = p;      // 让框架在同一级不同 plane 上建 8 条链路
            CHK_RET(CalcCommPlaneInfo(tag_, paraLvl1, opTransport[COMM_LEVEL1+p], in, out));
        }
        return HCCL_SUCCESS;
    }
    HcclResult CollAllGatherNewExecutor::CalcLevel0CommInfo(TransportMemType inputType,
        TransportMemType outputType,
        std::vector<LevelNSubCommTransport>& opTransport)
    {
        return HCCL_SUCCESS;
    }
    HcclResult CollAllGatherNewExecutor::CalcTransportMemType(TransportMemType &inputType, TransportMemType &outputType) override
    {
        inputType = TransportMemType::PARAM_INPUT;
        outputType = TransportMemType::PARAM_OUTPUT;
        HCCL_INFO("[CollAllGatherNewExecutor][CalcTransportMemType] tag[%s] inputType[%d], outputType[%d]",
            tag_.c_str(), inputType, outputType);
        return HCCL_SUCCESS;
    }
    u64 CollAllGatherNewExecutor::CalcLoopMaxCount(const u64 cclBuffSize, const u32 unitSize) override
    {
        return UINT64_MAX;
    }
    HcclResult CollAllGatherNewExecutor::KernelRun(const OpParam &param, ExecMem &execMem) override
    {
        HCCL_INFO("[CollAllGatherNewExecutor][KernelRun] algorithm start");

        int kPlaneNum  = 8;        // 7 HCCS + 1 RoCE
        // —— 把基类 RunContext 拆解后直接调用模板 ——
        std::array<SubCommInfo,kPlaneNum> subInfo{};
        for(int p=0;p<7;++p) subInfo[p] = GetSubCommInfo(COMM_LEVEL1, p);
        subInfo[7] = GetSubCommInfo(COMM_LEVEL1, COMM_INDEX_ROCE);

        auto &slaveStreams = algResResp_->slaveStreams;
        auto &notifiesMain = algResResp_->notifiesMain;
        auto &notifiesAux  = algResResp_->notifiesAux;
        std::unique_ptr<AlgTemplateBase> tmpl = AlgTemplateRegistry::Instance().GetAlgTemplate(
            TemplateType::TEMPLATE_ALL_GATHER_STRIPED_PIPELINE, dispatcher_);
        CHK_SMART_PTR_NULL(tmpl);

        //std::vector<std::shared_ptr<LocalNotify>> start(kPlaneNum), done(kPlaneNum);
        //for(int i=0;i<kPlaneNum;++i) { start[i]=LocalNotify::Create(dispatcher_); done[i]=LocalNotify::Create(dispatcher_);}        


        CHK_RET(tmpl->Prepare(param, subInfo, param.stream, slaveStreams, notifiesMain, notifiesAux,
                             topoAttr_.userRank, topoAttr_.userRankSize, /*localHop=*/7));
        
        CHK_RET(ActiveSlaveStreams(param.stream));  // 基类 helper
        CHK_RET(tmpl->RunAsync());
        return HCCL_SUCCESS;
    }
    HcclResult CollAllGatherNewExecutor::Getlevel1CommRank(SubCommInfo& level1CommInfo) override
    {
        return HCCL_E_UNAVAIL;
    }
    HcclResult CollAllGatherNewExecutor::SelectTempAlg(std::unique_ptr<AlgTemplateBase> &level1TempAlg, u32 level1RankSize) override
    {
        return HCCL_E_UNAVAIL;
    }

    REGISTER_EXEC("AllGatherNewExecutor", AllGatherNew, CollAllGatherNewExecutor);
} // namespace hccl
