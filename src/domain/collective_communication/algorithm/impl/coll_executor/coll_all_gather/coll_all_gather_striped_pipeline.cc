#include "coll_all_gather_striped_pipeline.h"
#include "all_gather_striped_pipeline_pub.h"
#include "hccl/adapter_rts_common.h"

namespace hccl {
    CollAllGatherNewExecutor::CollAllGatherNewExecutor(const HcclDispatcher dispatcher,
        std::unique_ptr<TopoMatcher> &topoMatcher)
        : CollAllGatherExecutor(dispatcher, topoMatcher)
    {
        HCCL_INFO("[CollAllGatherNewExecutor] constructor called");
        DMAReduceFlag_ = false;
    }
    HcclResult CollAllGatherNewExecutor::CalcStreamNum(u32& streamNum)
    {
        u32 totalStreamNum = topoAttr_.deviceNumPerAggregation > 1U ? topoAttr_.deviceNumPerAggregation - 1U : 1U;
        streamNum = totalStreamNum - 1U;
        HCCL_INFO("[CollAllGatherMeshExecutor][CalcStreamNum] tag[%s] streamNum[%u]",
            tag_.c_str(), streamNum);
        return HCCL_SUCCESS;
    }
    HcclResult CollAllGatherNewExecutor::CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport)
    {
        TransportMemType in, out;
        CHK_RET(CalcTransportMemType(in, out));
        CHK_RET(CalcLevel0CommInfo(in, out, opTransport));
        CHK_RET(CalcLevel1CommInfo(in, out, opTransport));
        return HCCL_SUCCESS;
    }

    HcclResult CollAllGatherNewExecutor::CalcLevel0CommInfo(TransportMemType inputType,
        TransportMemType outputType,
        std::vector<LevelNSubCommTransport>& opTransport)
    {
        CommParaInfo commParaLevel0(COMM_LEVEL0, CommType::COMM_TAG_RING_INNER);
        CHK_RET(CalcCommPlaneInfo(tag_, commParaLevel0, opTransport[COMM_LEVEL0], inputType, outputType));
        return HCCL_SUCCESS;
    }
    HcclResult CollAllGatherNewExecutor::CalcTransportMemType(TransportMemType &inputType, TransportMemType &outputType)
    {
        inputType = TransportMemType::CCL_INPUT;
        outputType = TransportMemType::CCL_OUTPUT;
        HCCL_INFO("[CollAllGatherNewExecutor][CalcTransportMemType] tag[%s] inputType[%d], outputType[%d]",
            tag_.c_str(), inputType, outputType);
        return HCCL_SUCCESS;
    }
    u64 CollAllGatherNewExecutor::CalcLoopMaxCount(const u64 cclBuffSize, const u32 unitSize)
    {
        return (cclBuffSize / topoAttr_.userRankSize / HCCL_MIN_SLICE_ALIGN) * HCCL_MIN_SLICE_ALIGN / unitSize;
        //u64 perLoop = cclBuffSize / topoAttr_.userRankSize / HCCL_MIN_SLICE_ALIGN
        //          * HCCL_MIN_SLICE_ALIGN / unitSize;
        //return std::max<u64>(perLoop, 1);
    }

    HcclResult CollAllGatherNewExecutor::Orchestrate(OpParam &param, AlgResourceResponse &algRes)
    {
  HCCL_INFO("[AllGatherStripedPipeline][Orchestrate] start");

  tag_ = param.tag;
  algResResp_ = &algRes;

  ExecMem m;
  m.count     = param.DataDes.count;
  m.inputPtr  = param.inputPtr;
  m.outputPtr = param.outputPtr;

  const u64 typeSize   = GetTypeSize(GetDataType(param));
  const u64 sendBytes  = m.count * typeSize;
  const u64 recvBytes  = sendBytes * topoAttr_.userRankSize;

  // 只给出“视图”范围，不要把整块 200MiB 都暴露给本次 op
  CHK_PRT_RET(algRes.cclInputMem.size()  < sendBytes, HCCL_E_INTERNAL,
              HCCL_ERROR("cclInputMem too small, need %llu", sendBytes), HCCL_E_INTERNAL);
  CHK_PRT_RET(algRes.cclOutputMem.size() < recvBytes, HCCL_E_INTERNAL,
              HCCL_ERROR("cclOutputMem too small, need %llu", recvBytes), HCCL_E_INTERNAL);

  m.inputMem  = algRes.cclInputMem.range(0, sendBytes);
  m.outputMem = algRes.cclOutputMem.range(0, recvBytes);

  // 预拷只拷“本 rank 的 sendBytes”
  CHK_RET(PreCopyToCclUsingCommStream(param, m, sendBytes));

  CHK_RET(KernelRun(param, m));

  // 把聚合后的 recvBytes 从 CCL 输出拷回用户输出
  CHK_RET(PostCopyFromCclUsingCommStream(param, m, recvBytes));

  HCCL_INFO("tag[%s], AllGather striped-pipeline orchestrate success.", tag_.c_str());
  return HCCL_SUCCESS;
    }

HcclResult CollAllGatherNewExecutor::PreCopyToCclUsingCommStream(const OpParam& param, ExecMem& m)
{
  if (sendBytes == 0 || m.inputPtr == nullptr) return HCCL_SUCCESS;

  DeviceMem dst = m.inputMem; // CCL staging (sendBytes 视图)
  DeviceMem src(reinterpret_cast<uint8_t*>(m.inputPtr), sendBytes); // 用户输入(设备指针)
  return HcclD2DMemcpyAsync(dispatcher_, dst, src, const_cast<Stream&>(param.stream)); // 用户→CCL
}

HcclResult CollAllGatherNewExecutor::PostCopyFromCclUsingCommStream(
    const OpParam& param, ExecMem& m, u64 recvBytes) {
  if (recvBytes == 0 || m.outputPtr == nullptr) return HCCL_SUCCESS;

  DeviceMem src = m.outputMem; // CCL 输出( recvBytes 视图 )
  DeviceMem dst(reinterpret_cast<uint8_t*>(m.outputPtr), recvBytes); // 用户输出(设备指针)
  return HcclD2DMemcpyAsync(dispatcher_, dst, src, const_cast<Stream&>(param.stream)); // CCL→用户
}


    HcclResult CollAllGatherNewExecutor::KernelRun(const OpParam &param, ExecMem &execMem)
    {
        HCCL_INFO("[CollAllGatherNewExecutor][KernelRun] algorithm start");

        const HcclDataType dtype = GetDataType(param);
        const u64 inputMemSize   = execMem.inputMem.size();
        const u64 bytesPerRank   = inputMemSize;  // AllGather：每 rank 等长

        // —— 取 L0：用来确定 commIndex（与 Mesh 一致）——
        CHK_RET(CheckCommSize(COMM_LEVEL0, COMM_INDEX_0 + 1));
        SubCommInfo level0 = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);
        u32 commIndex      = level0.localRank;
        HCCL_INFO("[StripedPipeline] L0 rankSize=%u localRank=%u", level0.localRankSize, level0.localRank);

        // —— 取 L1：跨节点子平面（TopoMatcher 需返回 UB 子平面的 SubCommInfo）——
        CHK_RET(CheckCommSize(COMM_LEVEL1, commIndex + 1));
        SubCommInfo level1 = GetSubCommInfo(COMM_LEVEL1, commIndex);
        HCCL_INFO("[StripedPipeline] L1 rankSize=%u localRank=%u", level1.localRankSize, level1.localRank);

        // —— 准备 7 个子平面的 SubCommInfo（当前先全部复用同一份 L1；若 TopoMatcher 已能按子平面拆分，替换为 7 份即可）——
        constexpr size_t kPlaneNum = AllGatherStripedPipeline::kPlaneNum;
        std::array<SubCommInfo, kPlaneNum> commPlanes{};
        for (size_t p = 0; p < kPlaneNum; ++p) {
            if (CheckCommSize(COMM_LEVEL1, p + 1) == HCCL_SUCCESS) {
                commPlanes[p] = GetSubCommInfo(COMM_LEVEL1, p);
            } else {
                // 兜底：还没拆成7份就复用 commIndex 对应的那份
                commPlanes[p] = GetSubCommInfo(COMM_LEVEL1, commIndex);
            }
        }


        //debug
        auto dumpLinks = [](const SubCommInfo& ci, const char* tag) {
            size_t n = std::min<size_t>(ci.links.size(), 4);
            for (size_t i = 0; i < n; ++i) {
                auto &lk = ci.links[i];
                LinkType lt = lk->GetLinkType();
                u32 peer = lk->GetRemoteRank();                   // ← 这里替换
                HCCL_INFO("[L1/%s] link[%zu] type=%d remoteRank=%u", tag, i, (int)lt, peer);
            }
        };
        for (size_t p = 0; p < kPlaneNum; ++p) {
            dumpLinks(commPlanes[p], std::to_string(p).c_str());
        }


        // —— 取得模板，按“你给的 Prepare 签名”调用 —— 
        auto &slaveStreams = algResResp_->slaveStreams;
        auto &notifiesMain = algResResp_->notifiesMain;
        auto &notifiesAux  = algResResp_->notifiesAux;

        std::unique_ptr<AlgTemplateBase> basePtr =
            AlgTemplateRegistry::Instance().GetAlgTemplate(
                TemplateType::TEMPLATE_ALL_GATHER_STRIPED_PIPELINE, dispatcher_);
        CHK_SMART_PTR_NULL(basePtr);

        // debug
        if(!basePtr) {
            HCCL_ERROR("[CollAllGatherNewExecutor][KernelRun] failed to get alg template");
            return HCCL_E_UNAVAIL;
        }

        auto *tmpl = dynamic_cast<AllGatherStripedPipeline*>(basePtr.get());
        CHK_SMART_PTR_NULL(tmpl);

        //debug
        if(!tmpl) {
            HCCL_ERROR("[CollAllGatherNewExecutor][KernelRun] failed to get alg template");
            return HCCL_E_UNAVAIL;
        }
        HCCL_INFO("[StripedPipeline][KernelRun] start tag[%s] rank[%u/%u] slaveStreams=%zu",
          tag_.c_str(), topoAttr_.userRank, topoAttr_.userRankSize, algResResp_->slaveStreams.size());


        // 只要把 L1（跨节点 UB 子平面）塞给模板，模板就会在 7 个平面上并行环传
        CHK_RET(tmpl->Prepare(execMem.inputMem, execMem.outputMem, execMem.count, dtype,
                            param.stream, slaveStreams, notifiesMain, notifiesAux,
                            topoAttr_.userRank, topoAttr_.userRankSize,
                            /*localHop*/ (u32)(topoAttr_.deviceNumPerAggregation - 1),
                            commPlanes));

        HCCL_INFO("[StripedPipeline][KernelRun] Prepare done");

        CHK_RET(ActiveSlaveStreams(param.stream));

        HCCL_INFO("[StripedPipeline][KernelRun] ActiveSlaveStreams done");

        CHK_RET(tmpl->RunAsync());

        HCCL_INFO("[StripedPipeline][KernelRun] done");
        return HCCL_SUCCESS;
    }
    HcclResult CollAllGatherNewExecutor::Getlevel1CommRank(SubCommInfo& level1CommInfo)
    {
        return HCCL_E_UNAVAIL;
    }
    HcclResult CollAllGatherNewExecutor::SelectTempAlg(std::unique_ptr<AlgTemplateBase> &level1TempAlg, u32 level1RankSize)
    {
        return HCCL_E_UNAVAIL;
    }

    REGISTER_EXEC("CollAllGatherNewExecutor", AllGatherNew, CollAllGatherNewExecutor);
} // namespace hccl
