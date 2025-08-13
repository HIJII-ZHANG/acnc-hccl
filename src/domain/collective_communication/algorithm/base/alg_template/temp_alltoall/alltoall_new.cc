#include "alltoall_new.h"
#include "alg_template_register.h"

namespace hccl {

HcclResult AlltoAllCM128Slice::Prepare(
    DeviceMem &inputMem, DeviceMem &outputMem, DeviceMem &scratchMem,
    const u64 count, const HcclDataType dataType, const Stream &stream,
    const HcclReduceOp, const u32, const std::vector<Slice> &, const u64 baseOffset)
{
  inputMem_   = inputMem;
  outputMem_  = outputMem;
  scratchMem_ = scratchMem;
  count_      = count;
  dataType_   = dataType;
  stream_     = stream;
  baseOffset_ = baseOffset;
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
  if (acc < bytesPerPair) sm.len[127] += (bytesPerPair - acc);  // 余数补到最后一片（可按 4KB 对齐）
}

inline HcclResult AlltoAllCM128Slice::TxOne(const LINK &link, u64 off, u64 sz, const Stream &s)
{
  DeviceMem src = outputMem_.range(off, sz);
  return link->TxAsync(UserMemType::OUTPUT_MEM, baseOffset_ + off, src.ptr(), sz, s);
}

inline HcclResult AlltoAllCM128Slice::RxOne(const LINK &link, u64 off, u64 sz, const Stream &s)
{
  DeviceMem dst = outputMem_.range(off, sz);
  return link->RxAsync(UserMemType::OUTPUT_MEM, baseOffset_ + off, dst.ptr(), sz, s);
}

HcclResult AlltoAllCM128Slice::RunAsync(const u32 my, const u32 size, const std::vector<LINK> &links)
{
  const u64 elemSz       = DataUnitSize(dataType_);
  const u64 bytesPerPair = count_ * elemSz;

  SliceMap sm; BuildSlices(bytesPerPair, sm);

  auto planeAgg     = [&](u32 k)->u32 { return aggsByPlane_[k]; };
  auto planeStream  = [&](u32 k)->const Stream& { return PlaneStream(k); };

  // —— 阶段 0：Gather（机内，各 plane 的非聚合者 → 对应 plane 的聚合者）——
  if (mode_ == 0) {
    for (u32 s=0; s<128; ++s) if (sm.len[s]) {
      const u32 k   = sm.plane[s];
      const u32 agg = planeAgg(k);
      const u64 off = PairBase(/*peer=*/my, bytesPerPair) + SliceOff(sm, s);
      if (my != agg) {
        CHK_RET(TxOne(links[agg], off, sm.len[s], planeStream(k)));
      } else {
        CHK_RET(RxOne(links[my],  off, sm.len[s], planeStream(k))); // 有的实现使用自环/内部搬运；按你们链接口替换
      }
    }
    // 统一等待（必要时；也可分 peer 等待）
    for (u32 p=0; p<size; ++p) {
      CHK_RET(links[p]->RxWaitDone(stream_));
      CHK_RET(links[p]->TxWaitDone(stream_));
    }
    return HCCL_SUCCESS;
  }

  // —— 阶段 1：Inter（跨机，仅“我是该 plane 的聚合者”的切片参与）——
  if (mode_ == 1) {
    // 只处理我是聚合者的那些 plane
    for (u32 s=0; s<128; ++s) if (sm.len[s]) {
      const u32 k   = sm.plane[s];
      const u32 agg = planeAgg(k);
      if (my != agg) continue;

      // 与所有对端聚合者互发（pairwise 全双工）
      for (u32 peer=0; peer<size; ++peer) if (peer != my) {
        const u64 off = PairBase(/*peer=*/peer, bytesPerPair) + SliceOff(sm, s);
        const Stream& sub = planeStream(k);
        CHK_RET(TxOne(links[peer], off, sm.len[s], sub));
        CHK_RET(RxOne(links[peer], off, sm.len[s], sub));
        CHK_RET(links[peer]->TxWaitDone(sub));
        CHK_RET(links[peer]->RxWaitDone(sub));
      }
    }
    return HCCL_SUCCESS;
  }

  // —— 阶段 2：Scatter（机内，聚合者 → 本地其他 NPU；非聚合者只接收）——
  if (mode_ == 2) {
    for (u32 s=0; s<128; ++s) if (sm.len[s]) {
      const u32 k   = sm.plane[s];
      const u32 agg = planeAgg(k);
      const u64 off = PairBase(/*peer=*/my, bytesPerPair) + SliceOff(sm, s);
      const Stream& sub = planeStream(k);

      if (my == agg) {
        // 从聚合者向所有本地域成员“按对端 rank 的区域”分发
        for (u32 dst=0; dst<size; ++dst) if (dst != agg) {
          const u64 offDst = PairBase(/*peer=*/dst, bytesPerPair) + SliceOff(sm, s);
          CHK_RET(TxOne(links[dst], offDst, sm.len[s], sub));
        }
      } else {
        // 非聚合者只收属于自己的区域
        CHK_RET(RxOne(links[agg], off, sm.len[s], sub));
      }
    }
    // 可选：等待完成
    for (u32 p=0; p<size; ++p) {
      CHK_RET(links[p]->TxWaitDone(stream_));
      CHK_RET(links[p]->RxWaitDone(stream_));
    }
    return HCCL_SUCCESS;
  }

  return HCCL_E_INTERNAL;
}

/* 模板注册 */
REGISTER_TEMPLATE(TemplateType::TEMPLATE_ALL_TO_ALL_CM128SLICE, AlltoAllCM128Slice);

} // namespace hccl
