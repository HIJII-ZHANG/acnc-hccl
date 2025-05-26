/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ZERO_COPY_ACL_GRAPH_H
#define ZERO_COPY_ACL_GRAPH_H
#include <atomic>
#include <memory>
#include <hccl/hccl_types.h>
#include "hccl_communicator_attrs.h"
#include "hccl/base.h"
#include "hccl_impl_pub.h"
namespace hccl {
class ZeroCopyAclGraph {
public:
    ZeroCopyAclGraph();
    ~ZeroCopyAclGraph();
    bool SetAclGraphZeroCopyMode(
        DevType deviceType, HcclCMDType opType, OpParam &opParam, u64 bufferSize, u32 userRankSize);
    bool SetAclGraphZeroCopyModeAllocAlgResource(DevType deviceType, const OpParam &opParam,
        AlgResourceResponse &algResResponse, DeviceMem &outputBuffer, std::vector<DeviceMem> &deviceResOrigMem,
        u32 userRankSize);

private:
    bool IsAlgoSupportAclGraphZeroCopyMode(HcclCMDType opType, OpParam &opParam, u64 bufferSize, u32 userRankSize);
    bool IsReduceScatterSupportAclGraphZeroCopyMode(const OpParam &opParam, u64 bufferSize, u32 userRankSize);
    bool SetGraphMode(HcclCMDType opType, OpParam &opParam, u64 bufferSize, u32 userRankSize);
    bool ReduceScatterAllocAlgResource(const OpParam &opParam, AlgResourceResponse &algResResponse,
        DeviceMem &outputBuffer, std::vector<DeviceMem> &deviceResOrigMem, u32 userRankSize);

private:
    std::atomic<u32> tagResourceIndex_;
};
}
#endif //end of ZERO_COPY_ACL_GRAPH_H