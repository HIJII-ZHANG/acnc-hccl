// coll_alltoall_cm128slice_executor.cc
#include "coll_alltoall_new.h"
#include <array>

namespace hccl {

HcclResult CollAlltoAllCM128SliceExecutor::CalcStreamNum(u32& streamNum) override
{
  // 7 个子平面各一条从流（主流+7从流）；最少也允许 3（gather/inter/scatter）
  streamNum = 7;
  return HCCL_SUCCESS;
}

std::array<u32,7> CollAlltoAllCM128SliceExecutor::PickAggregatorsByPlane(const SubCommInfo& level0) const
{
  std::array<u32,7> aggs{};
  const u32 localSize =
#if defined(HAVE_SUBCOMMINFO_LOCAL_SIZE)
      level0.localRankSize;
#else
      level0.rankSize;  // 若结构不同，请替换为你们的字段
#endif
  for (u32 k=0; k<7; ++k) aggs[k] = (localSize > 0) ? (k % localSize) : 0;
  return aggs;
}

HcclResult CollAlltoAllCM128SliceExecutor::KernelRun(const OpParam& param, ExecMem& execMem)
{
  // 取 Template 实例（按模板类型枚举值）
  auto tempAlg = AlgTemplateRegistry::Instance()
                   .GetAlgTemplate(TemplateType::TEMPLATE_ALL_TO_ALL_CM128SLICE, dispatcher_);
  CHK_SMART_PTR_NULL(tempAlg);

  // 取得分层通信域
  SubCommInfo level0 = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0); // 机内
  SubCommInfo level1 = GetSubCommInfo(COMM_LEVEL1, COMM_INDEX_0); // 跨机（聚合者间）

  // 向模板传递多从流与聚合者表
  if (auto* my = dynamic_cast<AlltoAllCM128Slice*>(tempAlg.get())) {
    (void)my->SetSlaveStreams(algResResp_->slaveStreams);
    (void)my->SetAggregators(PickAggregatorsByPlane(level0));
  }

  // 标准 Prepare：数据/主流/计数等（手册接口）
  CHK_RET(tempAlg->Prepare(execMem.outputMem, execMem.outputMem, execMem.inputMem,
                           execMem.count, param.DataDes.dataType, param.stream,
                           HCCL_REDUCE_RESERVED, INVALID_VALUE_RANKID,
                           std::vector<Slice>{}, /*baseOffset*/0));

  // 三段：Gather → Inter → Scatter（每段在相应子通信域上 RunTemplate）
  CHK_RET(tempAlg->SetMode(0));  // Gather（机内）
  CHK_RET(RunTemplate(tempAlg, level0));

  CHK_RET(tempAlg->SetMode(1));  // Inter（聚合者↔聚合者）
  CHK_RET(RunTemplate(tempAlg, level1));

  CHK_RET(tempAlg->SetMode(2));  // Scatter（机内）
  CHK_RET(RunTemplate(tempAlg, level0));

  return HCCL_SUCCESS;
}

} // namespace hccl
