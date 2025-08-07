#include "all_gather_striped_pipeline.h"
#include "alg_template_register.h"

namespace hccl {
AllGatherStripedPipeline::AllGatherStripedPipeline(int planeId, const HcclDispatcher dispatcher) : AlgTemplateBase(dispatcher), planeId_(planeId)
{
}
AllGatherStripedPipeline::~AllGatherStripedPipeline()
{
}


static constexpr int kPlaneNum  = 8;        // 7 HCCS + 1 RoCE
static constexpr int kSliceK    = 8;        // pipeline slices/segment
static constexpr float kBwHccs = 28.0f;
static constexpr float kBwRoce = 50.0f;
static constexpr float kSumBw  = 7*kBwHccs + kBwRoce;

inline uint64_t GetSegBytes(uint64_t total, int p){
    return static_cast<uint64_t>( total * ((p<7? kBwHccs:kBwRoce)/kSumBw) + 0.5f );
}

HcclResult AllGatherStripedPipeline::Prepare(const OpParam &param,
                       const std::array<SubCommInfo,kPlaneNum> &commPlanes,
                       const Stream &main,
                       const std::vector<Stream> &subStreams,
                       const std::vector<std::shared_ptr<LocalNotify>> &sigStart,
                       const std::vector<std::shared_ptr<LocalNotify>> &sigDone,
                       u32 rank, u32 rankSize, u32 localHop) {
        param_      = param;
        commPlanes_ = commPlanes;
        mainStream_ = main;
        subStreams_ = subStreams;
        sigStart_   = sigStart;
        sigDone_    = sigDone;
        rank_       = rank;
        rankSize_   = rankSize;
        localHop_   = localHop;
        bytesPerRank_ = param.count * SIZE_TABLE[param.dataType];
        for(int p=0;p<kPlaneNum;++p) segBytes_[p] = GetSegBytes(bytesPerRank_, p);
        return HCCL_SUCCESS;
}

HcclResult AllGatherStripedPipeline::StartSubs()
{
        for(int p=0;p<kPlaneNum;++p)
        {
            CHK_RET(LocalNotify::Post(mainStream_, dispatcher_, sigStart_[p], INVALID_VALUE_STAGE));
            CHK_RET(LocalNotify::Wait(subStreams_[p], dispatcher_, sigStart_[p], INVALID_VALUE_STAGE));
        }
        return HCCL_SUCCESS;
}
HcclResult AllGatherStripedPipeline::FinishSubs()
{
        for(int p=0;p<kPlaneNum;++p)
        {
            CHK_RET(LocalNotify::Post(subStreams_[p], dispatcher_, sigDone_[p], INVALID_VALUE_STAGE));
            CHK_RET(LocalNotify::Wait(mainStream_, dispatcher_, sigDone_[p], INVALID_VALUE_STAGE));
        }
        return HCCL_SUCCESS;
}
HcclResult AllGatherStripedPipeline::RunAsync( 
                        const std::array<LINK_LIST,kPlaneNum>& linksPerPlane)
{
        DeviceMem dstSelf = DeviceMem::create((u8*)param_.outputAddr + bytesPerRank_*rank_, bytesPerRank_);
        DeviceMem srcSelf = DeviceMem::create((u8*)param_.inputAddr, bytesPerRank_);
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dstSelf, srcSelf, mainStream_));

        // 启动 8 条辅流
        CHK_RET(StartSubs());

        // STEP‑1 8 条平面并行流水
        for(int plane=0; plane<kPlaneNum; ++plane) RunOnePlane(plane);

        CHK_RET(FinishSubs());
        return HCCL_SUCCESS;
}

void AllGatherStripedPipeline::RunOnePlane(int plane)
{
        uint64_t seg   = segBytes_[plane];
        uint64_t chunk = seg / rankSize_;               // 均匀 32/128 份
        uint64_t slice = chunk / kSliceK;
        u8* baseSend = (u8*)param_.inputAddr  + plane*seg + rank_*chunk;
        u8* baseRecv = (u8*)param_.outputAddr + plane*seg;
        for(u32 r=0;r<rankSize_-1;++r) {
            u32 txIdx = (rank_ + r) % rankSize_;
            u32 rxIdx = (rank_ - 1 - r + rankSize_) % rankSize_;
            u8* txBase = baseSend + (txIdx - rank_)*chunk;
            u8* rxBase = baseRecv + rxIdx*chunk;
            for(int s=0;s<kSliceK;++s) {
                HcclLinkType lt = (r < localHop_ ? LINK_RDMA_LOCAL : (plane<7? LINK_RDMA_HCCS : LINK_RDMA_ROCE));
                HcclSendAsync(dispatcher_, lt, (rank_+1)%rankSize_, txBase + s*slice, slice, subStreams_[plane]);
                HcclRecvAsync(dispatcher_, lt, (rank_-1+rankSize_)%rankSize_, rxBase + s*slice, slice, subStreams_[plane]);
            }
        }
    }

}