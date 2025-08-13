#ifndef ALL_GATHER_STRIPED_PUB__H
#define ALL_GATHER_STRIPED_PUB_H

#include "alg_template_base_pub.h"
#include <sstream>

namespace hccl {
class AllGatherStripedPipeline : public AlgTemplateBase {
public:
    explicit AllGatherStripedPipeline(const HcclDispatcher dispatcher);
    ~AllGatherStripedPipeline() override = default;

    // 常量（在头里自带，避免未定义）
    static constexpr int  kPlaneNum = 8;     // 7×HCCS + 1×RoCE
    static constexpr int  kSliceK   = 8;
    static constexpr float kBwHccs  = 28.0f; // GB/s
    static constexpr float kBwRoce  = 50.0f; // GB/s
    static constexpr float kSumBw   = 7*kBwHccs + kBwRoce;

    static inline uint64_t GetSegBytes(uint64_t total, int p) {
        return static_cast<uint64_t>( total * ((p < 7 ? kBwHccs : kBwRoce)/kSumBw) + 0.5 );
    }

    // 与 fe401bf8db 模板风格一致：不用 OpParam；直接传入内存/流/通知/秩信息/comm 列表
    HcclResult Prepare(
        DeviceMem &usrInMem,
        DeviceMem &usrOutMem,
        u64 totalCount,
        HcclDataType dataType,
        const Stream &mainStream,
        const std::vector<Stream> &slaveStreams,
        const std::vector<std::shared_ptr<LocalNotify>> &notifyMain,
        const std::vector<std::shared_ptr<LocalNotify>> &notifyAux,
        u32 userRank, u32 rankSize, u32 localHop,
        const std::array<SubCommInfo, kPlaneNum> &commPlanes  // 每个平面的 links 都在这里
    );

    // 本模板不走 RunTemplate() 注入 links；直接用 Prepare 里保存的 commPlanes_ 自己跑
    HcclResult RunAsync();

private:
    // 简单的主从流同步（仿照你们 Mesh 模板）
    HcclResult StartSubs();
    HcclResult FinishSubs();
    HcclResult RunOnePlaneWithRingTemplate(int plane);
    // ===== 成员 =====
    DeviceMem  inMem_{};
    DeviceMem  outMem_{};
    u64        count_{};               // 元素个数
    HcclDataType dataType_{};
    u64        bytesPerRank_{};        // 本 rank 字节数
    u64        segBytes_[kPlaneNum]{}; // 每 plane 条纹大小

    Stream     mainStream_{};
    std::vector<Stream>                         subStreams_;
    std::vector<std::shared_ptr<LocalNotify>>   notifyMain_, notifyAux_;

    u32 userRank_{}, rankSize_{}, localHop_{};

    std::array<SubCommInfo, kPlaneNum>          commPlanes_{};
};
}  // namespace hccl

#endif /* * ALL_GATHER_STRIPED_PUB_H */
