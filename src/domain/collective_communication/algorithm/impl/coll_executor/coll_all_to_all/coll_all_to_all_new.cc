// coll_all_to_all_new.cc
#include "coll_all_to_all_new.h"
#include <array>
#include "stream_utils.h"
#include "alltoall_new_pub.h"

namespace hccl {

CollAlltoAllCM128SliceExecutor::CollAlltoAllCM128SliceExecutor(
    const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher>& topoMatcher)
    : CollAlltoAllExecutor(dispatcher, topoMatcher)
{
}


HcclResult CollAlltoAllCM128SliceExecutor::CalcStreamNum(u32& streamNum)
{
  // 7 个子平面各一条从流（主流+7从流）；最少也允许 3（gather/inter/scatter）
  streamNum = 7;
  return HCCL_SUCCESS;
}

HcclResult CollAlltoAllCM128SliceExecutor::CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport)
{
  // 确保下标可写：这里只用到 LEVEL0/LEVEL1，按需扩到 max(level)+1
  const u32 need = (std::max)(static_cast<u32>(COMM_LEVEL0), static_cast<u32>(COMM_LEVEL1)) + 1;
  if (opTransport.size() < need) opTransport.resize(need);

  TransportMemType inT  = TransportMemType::CCL_INPUT;
  TransportMemType outT = TransportMemType::CCL_OUTPUT;

  // Level0：机内 Mesh
  CommParaInfo l0(COMM_LEVEL0, CommType::COMM_TAG_MESH);
  CHK_RET(CalcCommPlaneInfo(tag_, l0, opTransport[COMM_LEVEL0], inT, outT));

  // Level1：跨机（用基类实现）
  CHK_RET(CollNativeExecutorBase::CalcLevel1CommInfo(inT, outT, opTransport));

  return HCCL_SUCCESS;
}

HcclResult CollAlltoAllCM128SliceExecutor::Orchestrate(OpParam& param, AlgResourceResponse& algRes)
{
    HcclUs startTs = TIME_NOW();
    HcclResult ret = HCCL_SUCCESS;

    // 1) 记录 tag、资源指针、参数镜像（与示例一致）
    tag_        = param.tag;
    algResResp_ = &algRes;
    AlltoAllVParam_ = param;  // 后续 KernelRun / Template 可能需要其中的 DataDes/stream 等

    // 2) 打点：以 stream id + tag 关联 profiler
    HCCL_PROFILER_ADD_STREAM_BY_STREAMID(param.stream.id(), param.tag, 0, algType_);

    // 3) 组装 ExecMem（按你示例的最小必需字段；其余字段按存在与否补充）
    ExecMem execMem;
    execMem.count     = 0;       // 等量 all-to-all：每对端元素数
    execMem.inputPtr  = param.inputPtr;    // 用户输入指针
    execMem.outputPtr = param.outputPtr;   // 用户输出指针

    // —— 可选：如果 ExecMem/DeviceMem 版本存在这些字段，就一并补齐 —— //
#ifdef HCCL_EXECMEM_HAS_DEVICEMEM
    // WrapUserMem/MakeDeviceMem 为示例名；替换为你工程内创建 DeviceMem 的实际帮助函数
    execMem.inputMem   = DeviceMem(param.inputPtr, /*bytes=*/param.count * SizeOf(param.DataDes.dataType));
    execMem.outputMem  = DeviceMem(param.outputPtr, /*bytes=*/param.count * SizeOf(param.DataDes.dataType));
    //  scratch 由资源阶段分配（若 AlgResourceResponse 暴露了对应字段）
    execMem.scratchInputMem  = algRes.cclInputMem;   // 若字段名不同请替换
    execMem.scratchOutputMem = algRes.cclOutputMem;  // 若字段名不同请替换
#endif

    // 4) 进入核心执行（模板的 Prepare/RunAsync 都在 KernelRun 里完成）
    ret = KernelRun(param, execMem);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[CollRunAlltoAllNew][Orchestrate] errNo[0x%016llx] executor run failed",
            HCCL_ERROR_CODE(ret)), ret);

    // 5) 结束 profiler / 日志
    HCCL_PROFILER_DEL_STREAM_BY_STREAMID(param.stream.id());
    HCCL_INFO("tag[%s], CollRunAlltoAllNew orchestrate success, take time [%lld]us.",
        param.tag.c_str(), DURATION_US(TIME_NOW() - startTs));

    return HCCL_SUCCESS;
}

std::array<u32,7> CollAlltoAllCM128SliceExecutor::PickAggregatorsByPlane(const SubCommInfo& level0) const
{
  std::array<u32,7> aggs{};
  const u32 localSize =
#if defined(HAVE_SUBCOMMINFO_LOCAL_SIZE)
      level0.localRankSize;
#else
      level0.rankSize;
#endif
  for (u32 k=0; k<7; ++k) aggs[k] = (localSize > 0) ? (k % localSize) : 0;
  return aggs;
}

HcclResult CollAlltoAllCM128SliceExecutor::KernelRun(const OpParam &param, ExecMem &execMem)
{
 // 取模板实例（模板类型枚举需与你工程匹配）
  auto tempAlg = AlgTemplateRegistry::Instance()
                   .GetAlgTemplate(TemplateType::TEMPLATE_ALL_TO_ALL_CM128SLICE, dispatcher_);
  CHK_SMART_PTR_NULL(tempAlg);

  // 子通信域
  SubCommInfo level0 = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);
  SubCommInfo level1 = GetSubCommInfo(COMM_LEVEL1, COMM_INDEX_0);

  // 12参 Prepare：AlltoAllVStagedMesh 风格
  StageAlltoAllVAddrInfo sendAddrInfo{};
  StageAlltoAllVAddrInfo recvAddrInfo{};

  // 如果 ExecMem 只有一个 scratchMem，就两处都传它
  DeviceMem scratchA = execMem.scratchMem;
  DeviceMem scratchB = execMem.scratchMem;

  // 从资源获取主/从流与信号
  Stream mainStream = param.stream; // 非 const 引用
  std::vector<Stream> &ubStreams = algResResp_->slaveStreams;
  auto &notifyMainToSub = algResResp_->notifiesMain;
  auto &notifySubToMain = algResResp_->notifiesAux;

  // send/recv 使用用户 input/output 的 DeviceMem（如无，可简单包一层）
  DeviceMem sendMem = execMem.inputMem;   // 若没有 inputMem 字段，请按你工程的封装方式创建
  DeviceMem recvMem = execMem.outputMem;  // 同上

  // 如果你的 ExecMem 里没有 inputMem/outputMem，且模板需要 DeviceMem，
  // 可以用 param.inputPtr/param.outputPtr 封装一个 DeviceMem（略）

  CHK_RET(tempAlg->Prepare(
      /*sendMem          */ sendMem,
      /*recvMem          */ recvMem,
      /*scratchInputMem  */ scratchA,
      /*scratchOutputMem */ scratchB,
      /*sendAddrInfo     */ sendAddrInfo,
      /*recvAddrInfo     */ recvAddrInfo,
      /*isAlltoAllZCopy  */ isAlltoAllZCopyMode_,
      /*userRank         */ topoAttr_.userRank,
      /*mainStream       */ mainStream,
      /*subStreams       */ subStreams,
      /*notifyMainToSub  */ notifyMainToSub,
      /*notifySubToMain  */ notifySubToMain));

  // （可选）把 plane→聚合者下发到模板（仅当模板暴露此接口）
  //if (auto* t = dynamic_cast<AlltoAllCM128Slice *>(tempAlg.get())) {
  //  (void)t->SetAggregators(PickAggregatorsByPlane(level0));
  //}

  // 三段：Gather(机内) → Inter(跨机) → Scatter(机内)
  // 如果模板没有 SetMode，可直接三次 RunTemplate，模板根据子通信域自行决策
  // (void)tempAlg->SetMode(0);
  CHK_RET(RunTemplate(tempAlg, level0));
  // (void)tempAlg->SetMode(1);
  CHK_RET(RunTemplate(tempAlg, level1));
  // (void)tempAlg->SetMode(2);
  CHK_RET(RunTemplate(tempAlg, level0));

  HCCL_INFO("[CollRunAlltoAllNew] executor run success.");
  return HCCL_SUCCESS;
}

REGISTER_EXEC("CollAlltoAllNew", AlltoAllNew, CollAlltoAllCM128SliceExecutor);

} // namespace hccl
