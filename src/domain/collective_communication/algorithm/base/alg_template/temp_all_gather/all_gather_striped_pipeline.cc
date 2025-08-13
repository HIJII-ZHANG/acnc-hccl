#include "all_gather_striped_pipeline.h"
#include "alg_template_register.h"
#include "coll_executor_base.h"

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
    DeviceMem dstSelf = outMem_.range(bytesPerRank_ * userRank_, bytesPerRank_);
    CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dstSelf, inMem_, mainStream_));

    CHK_RET(StartSubs());

    // STEP-1：8 个 plane 并行（每个 plane 走一次 RING 模板）
    for (int p = 0; p < kPlaneNum; ++p) {
        CHK_RET(RunOnePlaneWithRingTemplate(p));
    }

    CHK_RET(FinishSubs());
    return HCCL_SUCCESS;
}

HcclResult AllGatherStripedPipeline::RunOnePlaneWithRingTemplate(int plane)
{
    // 每个 plane 的条纹大小 / 切块
    const u64 seg   = segBytes_[plane];
    const u64 chunk = seg / rankSize_;                // 均匀切成 rankSize 份
    // 这个 plane 在用户输出内存中的基址
    DeviceMem outSeg = outMem_.range(plane * seg, seg);
    CHK_SMART_PTR_NULL(outSeg);

    // 生成 RING 模板所需的 Slice 列表（每个 rank 一块）
    std::vector<Slice> slices(rankSize_);
    for (u32 i = 0; i < rankSize_; ++i) {
        slices[i].offset = i * chunk;
        slices[i].size   = chunk;
    }

    // 取一份 RING 模板（仓库已有）
    std::unique_ptr<AlgTemplateBase> ringTmpl =
        AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_GATHER_RING, dispatcher_);
    CHK_SMART_PTR_NULL(ringTmpl);

    // 准备：把“plane 条纹”映射成一次子 AllGather
    // 注意：这里把 input 也指向 outSeg，让模板在 outSeg 内部搬运各 rank 的块（符合 AllGather）
    // 若你的 RING 模板需要 input/out 分离，可把 input 指向 inMem_ 对应 plane 的条纹：
    //   DeviceMem inSeg = inMem_.range(plane * seg, seg);
    //   然后把下面的第1、2参数分别改成 inSeg / outSeg。
    CHK_RET(ringTmpl->Prepare(
        /*outputMem*/ outSeg,
        /*outputMem*/ outSeg,
        /*inputMem */ outSeg,                   // 或 inSeg（取决于你们 RING 模板签名）
        /*count    */ seg / SIZE_TABLE[dataType_],
        /*dtype    */ dataType_,
        /*stream   */ subStreams_[plane],
        /*reduce   */ HCCL_REDUCE_RESERVED,
        /*bridgeId */ INVALID_VALUE_RANKID,
        /*slices   */ slices,
        /*baseOff  */ 0));

    // 按该 plane 的通信域运行（把 commPlanes_[plane] 传进去）
    return CollExecutorBase::RunTemplate(ringTmpl, commPlanes_[plane]);
}

REGISTER_TEMPLATE(TemplateType::TEMPLATE_ALL_GATHER_STRIPED_PIPELINE, AllGatherStripedPipeline);
}