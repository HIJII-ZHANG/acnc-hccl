#ifndef ALLTOALL_CM128SLICE_PUB_H
#define ALLTOALL_CM128SLICE_PUB_H
#include "alg_template_base_pub.h"

namespace hccl {

class AlltoAllCM128Slice : public AlgTemplateBase {
public:
  explicit AlltoAllCM128Slice(const HcclDispatcher dispatcher) : AlgTemplateBase(dispatcher) {}
  ~AlltoAllCM128Slice() override = default;

  // —— 使用你们的 AlltoAllVStagedMesh 版 Prepare（12 参数）——
  HcclResult Prepare(DeviceMem &sendMem, DeviceMem &recvMem, DeviceMem &scratchInputMem,
                     DeviceMem &scratchOutputMem, StageAlltoAllVAddrInfo &sendAddrInfo,
                     StageAlltoAllVAddrInfo &recvAddrInfo, bool isAlltoAllZCopyMode,
                     u32 userRank, Stream &mainStream, std::vector<Stream> &subStreams,
                     std::vector<std::shared_ptr<LocalNotify>> &meshSignalMainToSub,
                     std::vector<std::shared_ptr<LocalNotify>> &meshSignalSubToMain) override;

  // 在单一子通信域 links 上执行
  HcclResult RunAsync(const u32 rank, const u32 rankSize, const std::vector<LINK> &links) override;

  // 扩展配置（由执行器设置）
  HcclResult SetMode(int m)                       { mode_ = m; return HCCL_SUCCESS; } // 0/1/2
  HcclResult SetSlaveStreams(const std::vector<Stream>& sts) { slaveStreams_ = sts; return HCCL_SUCCESS; }
  HcclResult SetAggregators(const std::array<u32,7>& aggs)   { aggsByPlane_ = aggs; return HCCL_SUCCESS; }

private:
  // 128 切片与 7 平面映射
  struct SliceMap {
    std::array<u64,128> len{};   // 每片长度
    std::array<u64,128> off{};   // 片内偏移（前缀和）
    std::array<u32,128> plane{}; // 片所属 plane : s % 7
  };
  void BuildSlices(u64 bytesPerPair, SliceMap& sm) const;

  // 简化 Tx/Rx 封装（注意：需要 Stream & 非 const 引用）
  inline HcclResult TxOne(const LINK &link, u64 off, u64 sz, Stream &s);
  inline HcclResult RxOne(const LINK &link, u64 off, u64 sz, Stream &s);

  // plane → 从流映射；不足则回退主流
  inline Stream& PlaneStream(u32 plane) {
    if (!slaveStreams_.empty() && plane < slaveStreams_.size()) return slaveStreams_[plane];
    return mainStream_;
  }

  // 偏移：按“每对端连续区域”布局
  inline u64 PairBase(u32 peer, u64 bytesPerPair) const { return static_cast<u64>(peer) * bytesPerPair; }
  inline u64 SliceOff(const SliceMap& sm, u32 s) const { return sm.off[s]; }

private:
  // 运行期配置
  int mode_ = 1;                                // 0:Gather 1:Inter 2:Scatter
  std::vector<Stream> slaveStreams_;            // 7 个从流
  std::array<u32,7>   aggsByPlane_{0,1,2,3,4,5,6};

  // 这版 Prepare 里我们保存的对象
  DeviceMem sendMem_{}, recvMem_{}, scratchIn_{}, scratchOut_{};
  bool      zcopy_{false};
  u32       userRank_{0};
  Stream    mainStream_{};                      // 注意：非常量，可下传到 Tx/Rx
  std::vector<std::shared_ptr<LocalNotify>> notifyM2S_, notifyS2M_;

  // 继承自基类：dataType_/count_ 等如需可再扩展
};

}
#endif // ALLTOALL_CM128SLICE_PUB_H