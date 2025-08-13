#ifndef ALLTOALL_CM128SLICE_PUB_H
#define ALLTOALL_CM128SLICE_PUB_H
#include "alg_template_base_pub.h"

namespace hccl {

class AlltoAllCM128Slice : public AlgTemplateBase {
public:
  explicit AlltoAllCM128Slice(const HcclDispatcher dispatcher) : AlgTemplateBase(dispatcher) {}
  ~AlltoAllCM128Slice() override = default;

  // 标准 Prepare（数据/主流/计数/类型等）
  HcclResult Prepare(DeviceMem &inputMem, DeviceMem &outputMem, DeviceMem &scratchMem,
                     const u64 count, const HcclDataType dataType, const Stream &stream,
                     const HcclReduceOp reductionOp, const u32 root,
                     const std::vector<Slice> &slices, const u64 baseOffset) override;

  // 在单一子通信域 links 上执行（框架通过 RunTemplate 触达此处）
  HcclResult RunAsync(const u32 rank, const u32 rankSize, const std::vector<LINK> &links) override;

  // —— 扩展配置，由 Executor 设置 ——
  HcclResult SetMode(int m)                       { mode_ = m; return HCCL_SUCCESS; } // 0/1/2
  HcclResult SetSlaveStreams(const std::vector<Stream>& sts) {
    slaveStreams_ = sts; return HCCL_SUCCESS;
  }
  HcclResult SetAggregators(const std::array<u32,7>& aggs) {
    aggsByPlane_ = aggs; return HCCL_SUCCESS;
  }

private:
  // 128 切片与 7 平面映射
  struct SliceMap {
    std::array<u64,128> len{};   // 每片长度
    std::array<u64,128> off{};   // 片内偏移（前缀和）
    std::array<u32,128> plane{}; // 片所属 plane : s % 7
  };
  void BuildSlices(u64 bytesPerPair, SliceMap& sm) const;

  // 简化 Tx/Rx 封装
  inline HcclResult TxOne(const LINK &link, u64 off, u64 sz, const Stream &s);
  inline HcclResult RxOne(const LINK &link, u64 off, u64 sz, const Stream &s);

  // plane → 从流映射；不足则回退主流
  inline const Stream& PlaneStream(u32 plane) const {
    if (!slaveStreams_.empty() && plane < slaveStreams_.size()) return slaveStreams_[plane];
    return stream_;
  }

  // 偏移：按“每对端连续区域”布局（pair-major）
  inline u64 PairBase(u32 peer, u64 bytesPerPair) const { return static_cast<u64>(peer) * bytesPerPair; }
  inline u64 SliceOff(const SliceMap& sm, u32 s) const { return sm.off[s]; }

private:
  int mode_ = 1;                                 // 0:Gather 1:Inter 2:Scatter
  std::vector<Stream> slaveStreams_;             // 7 个从流
  std::array<u32,7>   aggsByPlane_{0,1,2,3,4,5,6}; // 每 plane 的聚合者（子通信域内 rank）

  // 继承自基类：
  // inputMem_, outputMem_, scratchMem_, count_, dataType_, stream_, baseOffset_
};

};
#endif // ALLTOALL_CM128SLICE_PUB_H