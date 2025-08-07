#include "coll_all_gather_new.h"

namespace hccl {
    CollAllGatherNewExecutor::CollAllGatherNewExecutor(const HcclDispatcher dispatcher,
        std::unique_ptr<TopoMatcher> &topoMatcher)
        : CollAllGatherExecutor(dispatcher, topoMatcher)
    {
    }
    HcclResult CollAllGatherNewExecutor::CalcStreamNum(u32& streamNum) override
    {
        streamNum = 8 - 1;  // 7 HCCS + 1 RoCE
        return HCCL_SUCCESS;
    }
    HcclResult CollAllGatherNewExecutor::CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport) override
    {
        return CollAllGatherExecutor::CalcCommInfo(opTransport);
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
        // —— 把基类 RunContext 拆解后直接调用模板 ——
        std::array<SubCommInfo,kPlaneNum> subInfo{};
        for(int p=0;p<7;++p) subInfo[p] = GetSubCommInfo(COMM_LEVEL1, p);
        subInfo[7] = GetSubCommInfo(COMM_LEVEL1, COMM_INDEX_ROCE);

        PrepareSubStreams(kPlaneNum);
        std::vector<std::shared_ptr<LocalNotify>> start(kPlaneNum), done(kPlaneNum);
        for(int i=0;i<kPlaneNum;++i) { start[i]=LocalNotify::Create(dispatcher_); done[i]=LocalNotify::Create(dispatcher_);}        

        AllGatherStripedPipeline tmpl(dispatcher_);
        CHK_RET(tmpl.Prepare(param, subInfo, param.stream, subStreams_, start, done,
                             topoAttr_.userRank, topoAttr_.userRankSize, /*localHop=*/7));
        CHK_RET(StartSlaveStreams(param.stream));  // 基类 helper
        CHK_RET(tmpl.RunAsync());
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
