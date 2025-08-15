// coll_alltoall_cm128slice_executor.h
#ifndef COLL_ALLTOALL_CM128SLICE_EXECUTOR_H
#define COLL_ALLTOALL_CM128SLICE_EXECUTOR_H
#include "coll_all_to_all_executor.h"
#include "alg_template_register.h"
#include <memory>
#include <vector>
#include <cstdint>

namespace hccl {

class CollAlltoAllCM128SliceExecutor final : public CollAlltoAllExecutor {
public:
  explicit CollAlltoAllCM128SliceExecutor(const HcclDispatcher dispatcher,
                                          std::unique_ptr<TopoMatcher>& topoMatcher)
    : CollAlltoAllExecutor(dispatcher, topoMatcher) {}

private:
  // 资源/建链（手册要求的两个关键钩子）
  HcclResult CalcStreamNum(u32& streamNum) override;                           // 7 个子平面 → 7 从流
  HcclResult CalcCommInfo(std::vector<LevelNSubCommTransport>& opT) override;  // Level0/Level1

  // 三段式执行：Gather(机内) → Inter(跨机, 聚合者间) → Scatter(机内)
  HcclResult KernelRun(const OpParam& param, ExecMem& execMem) override;
  HcclResult Orchestrate(OpParam& param, AlgResourceResponse& algRes) override;
};

} // namespace hccl
#endif
