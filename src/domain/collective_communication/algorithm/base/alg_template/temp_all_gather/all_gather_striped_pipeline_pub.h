#ifndef ALL_GATHER_STRIPED_PUB__H
#define ALL_GATHER_STRIPED_PUB_H

#include "alg_template_base_pub.h"
#include <sstream>

namespace hccl {
class AllGatherStripedPipeline : public AlgTemplateBase {
public:
    explicit AllGatherStripedPipeline(int planeId, const HcclDispatcher dispatcher);
    ~AllGatherStripedPipeline() override;

    HcclResult Prepare(const OpParam &param,
                       const std::array<SubCommInfo,kPlaneNum> &commPlanes,
                       const Stream &main,
                       const std::vector<Stream> &subStreams,
                       const std::vector<std::shared_ptr<LocalNotify>> &sigStart,
                       const std::vector<std::shared_ptr<LocalNotify>> &sigDone,
                       u32 rank, u32 rankSize, u32 localHop) override;
    HcclResult RunAsync(const std::array<LINK_LIST,kPlaneNum>& linksPerPlane) override;

protected:
private:
    HcclResult StartSubs();
    HcclResult FinishSubs();
    HcclResult RunOnePlane(int plane)

    OpParam                                     param_{};
    std::array<SubCommInfo,kPlaneNum>           commPlanes_{};
    Stream                                      mainStream_{};
    std::vector<Stream>                         subStreams_;
    std::vector<std::shared_ptr<LocalNotify>>   sigStart_, sigDone_;
    u32         rank_{};
    u32         rankSize_{};
    u32         localHop_{};
    uint64_t    bytesPerRank_{};
    uint64_t    segBytes_[kPlaneNum]{};
};
}  // namespace hccl

#endif /* * ALL_GATHER_STRIPED_PUB_H */
