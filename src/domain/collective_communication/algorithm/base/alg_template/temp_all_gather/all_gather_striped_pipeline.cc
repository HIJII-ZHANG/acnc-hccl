#include "all_gather_striped_pipeline.h"
#include "alg_template_register.h"

namespace hccl {
AllGatherStripedPipeline::AllGatherStripedPipeline(const HcclDispatcher dispatcher)
: AlgTemplateBase(dispatcher) {}

HcclResult AllGatherStripedPipeline::Prepare(
    DeviceMem &usrInMem, DeviceMem &usrOutMem, u64 totalCount, HcclDataType dataType,
    const Stream &mainStream, const std::vector<Stream> &slaveStreams,
    const std::vector<std::shared_ptr<LocalNotify>> &notifyMain,
    const std::vector<std::shared_ptr<LocalNotify>> &notifyAux,
    u32 userRank, u32 rankSize, u32 localHop,
    const std::array<SubCommInfo, kPlaneNum> &commPlanes)
{
    inMem_       = usrInMem;
    outMem_      = usrOutMem;
    count_       = totalCount;
    dataType_    = dataType;
    mainStream_  = mainStream;
    subStreams_  = slaveStreams;
    notifyMain_  = notifyMain;
    notifyAux_   = notifyAux;
    userRank_    = userRank;
    rankSize_    = rankSize;
    localHop_    = localHop;
    commPlanes_  = commPlanes;

    bytesPerRank_ = count_ * SIZE_TABLE[dataType_];
    for (int p=0;p<kPlaneNum;++p) segBytes_[p] = GetSegBytes(bytesPerRank_, p);
    return HCCL_SUCCESS;
}

HcclResult AllGatherStripedPipeline::StartSubs()
{
    for (size_t i=0;i<subStreams_.size();++i) {
        CHK_RET(LocalNotify::Post(mainStream_, dispatcher_, notifyAux_[i], INVALID_VALUE_STAGE));
        CHK_RET(LocalNotify::Wait(subStreams_[i], dispatcher_, notifyAux_[i], INVALID_VALUE_STAGE));
    }
    return HCCL_SUCCESS;
}
HcclResult AllGatherStripedPipeline::FinishSubs()
{
    for (size_t i=0;i<subStreams_.size();++i) {
        CHK_RET(LocalNotify::Post(subStreams_[i], dispatcher_, notifyMain_[i], INVALID_VALUE_STAGE));
        CHK_RET(LocalNotify::Wait(mainStream_, dispatcher_, notifyMain_[i], INVALID_VALUE_STAGE));
    }
    return HCCL_SUCCESS;
}

HcclResult AllGatherStripedPipeline::RunAsync()
{
    // STEP-0: 本地 copy input→output 自己段
    DeviceMem dstSelf = outMem_.range(bytesPerRank_*userRank_, bytesPerRank_);
    CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dstSelf, inMem_, mainStream_));

    // 启动从流
    CHK_RET(StartSubs());

    // STEP-1: 8 个 plane 并行
    for (int p=0; p<kPlaneNum; ++p) RunOnePlane(p);

    // 收尾
    CHK_RET(FinishSubs());
    return HCCL_SUCCESS;
}

void AllGatherStripedPipeline::RunOnePlane(int plane)
{
    const u64 seg   = segBytes_[plane];
    const u64 chunk = seg / rankSize_;
    const u64 slice = chunk / kSliceK;

    u8 *baseSend = static_cast<u8*>(inMem_.ptr())  + plane*seg + userRank_*chunk;
    u8 *baseRecv = static_cast<u8*>(outMem_.ptr()) + plane*seg;

    Stream s = subStreams_[plane];

    for (u32 r=0; r<rankSize_-1; ++r) {
        const u32 right = (userRank_ + 1) % rankSize_;
        const u32 left  = (userRank_ - 1 + rankSize_) % rankSize_;
        const u32 txIdx = (userRank_ + r) % rankSize_;
        const u32 rxIdx = (userRank_ - 1 - r + rankSize_) % rankSize_;
        u8 *txBase = baseSend + (txIdx - userRank_) * chunk;
        u8 *rxBase = baseRecv + rxIdx * chunk;

        // Node-first: 前 localHop_ 跳在本节点内
        const bool sameNodePhase = (r < localHop_);
        const HcclLinkType lt = sameNodePhase ? LINK_RDMA_LOCAL
                                : (plane < 7 ? LINK_RDMA_HCCS : LINK_RDMA_ROCE);

        for (int sl=0; sl<kSliceK; ++sl) {
            u8 *tx = txBase + sl*slice;
            u8 *rx = rxBase + sl*slice;
            // 发送到右邻，接收左邻
            CHK_RET(HcclSendAsync(dispatcher_, lt, right, tx, slice, s));
            CHK_RET(HcclRecvAsync(dispatcher_, lt, left , rx, slice, s));
        }
    }
}

REGISTER_TEMPLATE(TemplateType::TEMPLATE_ALL_GATHER_STRIPED_PIPELINE, AllGatherStripedPipeline);
}