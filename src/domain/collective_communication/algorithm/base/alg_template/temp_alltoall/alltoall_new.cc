#include "alltoall_new.h"
#include "alg_template_register.h"

namespace hccl {

HcclResult AlltoAllCM128Slice::Prepare(
    DeviceMem &sendMem, DeviceMem &recvMem, DeviceMem &scratchInputMem,
    DeviceMem &scratchOutputMem, StageAlltoAllVAddrInfo &/*sendAddrInfo*/,
    StageAlltoAllVAddrInfo &/*recvAddrInfo*/, bool isAlltoAllZCopyMode,
    u32 userRank, Stream &mainStream, std::vector<Stream> &subStreams,
    std::vector<std::shared_ptr<LocalNotify>> &meshSignalMainToSub,
    std::vector<std::shared_ptr<LocalNotify>> &meshSignalSubToMain)
{
  // 保存运行期对象（与手册“Prepare 接参 → RunAsync 执行”一致）
  sendMem_   = sendMem;
  recvMem_   = recvMem;
  scratchIn_ = scratchInputMem;
  scratchOut_= scratchOutputMem;
  zcopy_     = isAlltoAllZCopyMode;
  userRank_  = userRank;
  mainStream_= mainStream;
  slaveStreams_ = subStreams;            // 7 个从流
  notifyM2S_    = meshSignalMainToSub;   // 可选使用
  notifyS2M_    = meshSignalSubToMain;

  return HCCL_SUCCESS;
}

void AlltoAllCM128Slice::BuildSlices(u64 bytesPerPair, SliceMap& sm) const
{
  const u64 avg = std::max<u64>(1, bytesPerPair / 128);
  u64 acc = 0;
  for (u32 s=0; s<128; ++s) {
    sm.plane[s] = s % 7;
    sm.len[s]   = avg;
    sm.off[s]   = acc;
    acc        += avg;
  }
  if (acc < bytesPerPair) sm.len[127] += (bytesPerPair - acc); // 余数补到最后一片（可做 4KB 对齐）
}

inline HcclResult AlltoAllCM128Slice::TxOne(const LINK &link, u64 off, u64 sz, Stream &s)
{
  DeviceMem src = sendMem_.range(off, sz);
  return link->TxAsync(UserMemType::OUTPUT_MEM, /*userAddr*/ off, src.ptr(), sz, s);
}

inline HcclResult AlltoAllCM128Slice::RxOne(const LINK &link, u64 off, u64 sz, Stream &s)
{
  DeviceMem dst = recvMem_.range(off, sz);
  return link->RxAsync(UserMemType::OUTPUT_MEM, /*userAddr*/ off, dst.ptr(), sz, s);
}

HcclResult AlltoAllCM128Slice::RunAsync(const u32 my, const u32 size, const std::vector<LINK> &links)
{
  // 这里按“每对卡等长数据”的 All-to-All：bytesPerPair = count_ * sizeof(elem)
  // 如果你们框架把 count_/dataType_ 交给其他 Prepare，请根据实际来源获取
  const u64 elemSz       = DataUnitSize(dataType_);
  const u64 bytesPerPair = count_ * elemSz;

  SliceMap sm; BuildSlices(bytesPerPair, sm);

  auto planeAgg  = [&](u32 k)->u32 { return aggsByPlane_[k]; };
  auto& main     = mainStream_;

  // —— 0) Gather（机内：非聚合者 → 各 plane 的聚合者）——
  if (mode_ == 0) {
    for (u32 s=0; s<128; ++s) if (sm.len[s]) {
      const u32 k   = sm.plane[s];
      const u32 agg = planeAgg(k);
      Stream &sub   = PlaneStream(k);
      const u64 off = PairBase(/*peer=*/my, bytesPerPair) + SliceOff(sm, s);
      if (my != agg) {
        CHK_RET(TxOne(links[agg], off, sm.len[s], sub));
      } else {
        // 聚合者接收本地域其它成员（如果自环无意义，可在外层由执行器安排对等 Rx）
      }
    }
    for (u32 p=0; p<size; ++p) { CHK_RET(links[p]->RxWaitDone(main)); CHK_RET(links[p]->TxWaitDone(main)); }
    return HCCL_SUCCESS;
  }

  // —— 1) Inter（跨机：仅“我是该 plane 的聚合者”的切片参与）——
  if (mode_ == 1) {
    for (u32 s=0; s<128; ++s) if (sm.len[s]) {
      const u32 k   = sm.plane[s];
      const u32 agg = planeAgg(k);
      if (my != agg) continue;
      Stream &sub = PlaneStream(k);
      for (u32 peer=0; peer<size; ++peer) if (peer != my) {
        const u64 off = PairBase(/*peer=*/peer, bytesPerPair) + SliceOff(sm, s);
        CHK_RET(TxOne(links[peer], off, sm.len[s], sub));
        CHK_RET(RxOne(links[peer], off, sm.len[s], sub));
        CHK_RET(links[peer]->TxWaitDone(sub));
        CHK_RET(links[peer]->RxWaitDone(sub));
      }
    }
    return HCCL_SUCCESS;
  }

  // —— 2) Scatter（机内：聚合者 → 本地其他 NPU；非聚合者只接收）——
  if (mode_ == 2) {
    for (u32 s=0; s<128; ++s) if (sm.len[s]) {
      const u32 k   = sm.plane[s];
      const u32 agg = planeAgg(k);
      Stream &sub   = PlaneStream(k);
      if (my == agg) {
        for (u32 dst=0; dst<size; ++dst) if (dst != agg) {
          const u64 offDst = PairBase(/*peer=*/dst, bytesPerPair) + SliceOff(sm, s);
          CHK_RET(TxOne(links[dst], offDst, sm.len[s], sub));
        }
      } else {
        const u64 offMy = PairBase(/*peer=*/my, bytesPerPair) + SliceOff(sm, s);
        CHK_RET(RxOne(links[agg], offMy, sm.len[s], sub));
      }
    }
    for (u32 p=0; p<size; ++p) { CHK_RET(links[p]->TxWaitDone(main)); CHK_RET(links[p]->RxWaitDone(main)); }
    return HCCL_SUCCESS;
  }

  return HCCL_E_INTERNAL;
}

/* 模板注册 */
REGISTER_TEMPLATE(TemplateType::TEMPLATE_ALL_TO_ALL_CM128SLICE, AlltoAllCM128Slice);

} // namespace hccl
