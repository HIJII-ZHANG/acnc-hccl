// coll_all_to_all_new.cc
#include "coll_all_to_all_new.h"
#include <array>
#include "stream_utils.h"

namespace hccl {

CollAlltoAllCM128SliceExecutor::CollAlltoAllCM128SliceExecutor(
    const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher>& topoMatcher)
    : CollAlltoAllExecutor(dispatcher, topoMatcher)
{
  DMAReduceFlag_ = false;  // 默认不使用 DMA Reduce
}


HcclResult CollAlltoAllCM128SliceExecutor::CalcStreamNum(u32& streamNum) override
{
  // 7 个子平面各一条从流（主流+7从流）；最少也允许 3（gather/inter/scatter）
  streamNum = 7;
  return HCCL_SUCCESS;
}

HcclResult CollAlltoAllCM128SliceExecutor::CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport)
{
  // === 方案 A：打平通信域（推荐）===
  // 将 level0/level1 合并为一个 COMBINE 平面，模板内部一次 RunAsync 完成三段编排
  if (opTransport.size() < COMM_LEVEL_MAX) opTransport.resize(COMM_LEVEL_MAX);
  TransportMemType inputType  = TransportMemType::CCL_INPUT;
  TransportMemType outputType = TransportMemType::CCL_OUTPUT;

  CommParaInfo commCombinePara(COMM_COMBINE_ORDER, CommType::COMM_TAG_MESH);
  CHK_RET(CalcCommPlaneInfo(tag_, commCombinePara,
          opTransport[COMM_COMBINE_ORDER], inputType, outputType));
  return HCCL_SUCCESS;

  // === 方案 B：分层通信域（如需三段分别 RunTemplate，解开下面注释并注释掉上面的“打平”）===
  /*
  if (opTransport.size() < COMM_LEVEL_MAX) opTransport.resize(COMM_LEVEL_MAX);
  TransportMemType inputType  = TransportMemType::CCL_INPUT;
  TransportMemType outputType = TransportMemType::CCL_OUTPUT;

  // level0：机内 Mesh（用于 Gather/Scatter）
  CommParaInfo commParaLevel0(COMM_LEVEL0, CommType::COMM_TAG_MESH);
  CHK_RET(CalcCommPlaneInfo(tag_, commParaLevel0,
          opTransport[COMM_LEVEL0], inputType, outputType));

  // level1：跨机（聚合 rank 间）— 复用基类实现
  CHK_RET(CollNativeExecutorBase::CalcLevel1CommInfo(inputType, outputType, opTransport));
  return HCCL_SUCCESS;
  */
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
    execMem.count     = param.count;       // 等量 all-to-all：每对端元素数
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

HcclResult CollAlltoAllCM128SliceExecutor::KernelRun(const OpParam &param, ExecMem &execMem)
{
  // 1) 取模板实例（注册名在 Operator 挑选时用 `"RunAlltoAllNew"`，这里按模板枚举拿）
  std::unique_ptr<AlgTemplateBase> tempAlg =
      AlgTemplateRegistry::Instance().GetAlgTemplate(
          TemplateType::TEMPLATE_ALL_TO_ALL_CM128SLICE, dispatcher_);
  CHK_SMART_PTR_NULL(tempAlg);

  // 2) 组装 AlltoAllVStagedMesh（12 参）版 Prepare 的参数
  //    如果你们需要自定义“每对端位移/计数”，可在 sendAddrInfo/recvAddrInfo 中填入映射
  StageAlltoAllVAddrInfo sendAddrInfo {};
  StageAlltoAllVAddrInfo recvAddrInfo {};

  // 主流/从流/信号：父类在 CalcResRequest 阶段已放入 algResResp_
  Stream &mainStream = const_cast<Stream&>(param.stream); // 模板需要非 const 引用
  std::vector<Stream>              &subStreams      = algResResp_->slaveStreams;
  std::vector<std::shared_ptr<LocalNotify>> &notifyMainToSub = algResResp_->notifiesMain;
  std::vector<std::shared_ptr<LocalNotify>> &notifySubToMain = algResResp_->notifiesAux;

  // 3) 调用模板 Prepare（12 参数）
  CHK_RET(tempAlg->Prepare(
      /*sendMem          */ execMem.inputMem,
      /*recvMem          */ execMem.outputMem,
      /*scratchInputMem  */ execMem.scratchInputMem,
      /*scratchOutputMem */ execMem.scratchOutputMem,
      /*sendAddrInfo     */ sendAddrInfo,
      /*recvAddrInfo     */ recvAddrInfo,
      /*isAlltoAllZCopy  */ isAlltoAllZCopyMode_,
      /*userRank         */ topoAttr_.userRank,
      /*mainStream       */ mainStream,
      /*subStreams       */ subStreams,
      /*notifyMainToSub  */ notifyMainToSub,
      /*notifySubToMain  */ notifySubToMain));

  // （可选）若你的模板支持 plane→聚合者 的外部配置，在此注入：
  // if (auto* ext = dynamic_cast<YourAlltoAllVNewImpl*>(tempAlg.get())) {
  //   std::array<u32,7> aggs{};
  //   for (u32 k=0; k<7; ++k) aggs[k] = k % std::max<u32>(1, topoAttr_.intraRankSize);
  //   (void)ext->SetAggregators(aggs);
  // }

  // 4) 下发执行
  // === 方案 A：打平通信域 — 一次 RunTemplate 覆盖三段 ===
  {
    SubCommInfo combine = GetSubCommInfo(COMM_COMBINE_ORDER, COMM_INDEX_0);
    CHK_RET(RunTemplate(tempAlg, combine));
  }

  // === 方案 B：分层通信域 — 三段分开 RunTemplate（解开注释以启用）===
  /*
  {
    SubCommInfo level0 = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);
    SubCommInfo level1 = GetSubCommInfo(COMM_LEVEL1, COMM_INDEX_0);

    // 如果模板提供 SetMode，可在三段之间切换；否则可由模板内部依据子通信域自行判别
    // (void)tempAlg->SetMode(0); // Gather
    CHK_RET(RunTemplate(tempAlg, level0));

    // (void)tempAlg->SetMode(1); // Inter
    CHK_RET(RunTemplate(tempAlg, level1));

    // (void)tempAlg->SetMode(2); // Scatter
    CHK_RET(RunTemplate(tempAlg, level0));
  }
  */

  HCCL_INFO("[CollRunAlltoAllNew] executor run success.");
  return HCCL_SUCCESS;
}

REGISTER_EXEC("CollAlltoAllNew", AlltoAllNew, CollAlltoAllCM128SliceExecutor);

} // namespace hccl
