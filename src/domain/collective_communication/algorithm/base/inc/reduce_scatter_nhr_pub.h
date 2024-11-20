/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef REDUCE_SCATTER_NHR_PUB_H
#define REDUCE_SCATTER_NHR_PUB_H

#include <cmath>

#include "nonuniform_hierarchical_ring_base_pub.h"

#include "mem_host_pub.h"
#include "mem_device_pub.h"
#include "stream_pub.h"

#include "reducer_pub.h"
#include "sender_pub.h"
#include "comm_base_pub.h"

namespace hccl {
class ReduceScatterNHR : public NHRBase {
public:
    explicit ReduceScatterNHR(const HcclDispatcher dispatcher, const u64 reduceAttrBitMap);
    ~ReduceScatterNHR() override;

    HcclResult RunAsync(const u32 rank, const u32 rankSize, const std::vector<LINK> &links) override;

    HcclResult RunAsyncWithReorder(const u32 rank, const u32 rankSize, const std::vector<LINK> &links);

protected:
private:
    HcclResult SimpleCheck(const u32 rank, const u32 rankSize, const std::vector<LINK> &links);
    HcclResult CheckSlices(const std::vector<Slice> &checkSlices, const u32 rankSize);

    HcclResult RunReduceScatterNHR(const u32 rank, const u32 rankSize, const std::vector<LINK> &links,
        const std::vector<Slice> &inputSlices, const std::vector<Slice> &outputSlices);
    HcclResult RunSourceSender(const LINK &link, InterServerAlgoStep &stepInfo, const std::vector<Slice> &inputSlices,
        const std::vector<Slice> &outputSlices);
    HcclResult RunDestReducer(const LINK &link, InterServerAlgoStep &stepInfo, const std::vector<Slice> &inputSlices,
        const std::vector<Slice> &outputSlices);

    HcclResult GetStepInfo(u32 step, u32 nSteps, u32 rank, u32 rankSize, InterServerAlgoStep &stepInfo) override;

    HcclResult InlineReducer(const LINK &linkLeft, const std::vector<ReducerMemoryInfo> &rxReduceMems);

    HcclResult InlineReduceRx(const LINK &linkLeft, std::vector<Slice> &rxSlices, std::vector<Slice> &rxSlicestemp);

    HcclResult InlineReduceRxLastStep(const LINK &linkLeft, InterServerAlgoStep &stepInfo,
        const std::vector<Slice> &inputSlices, const std::vector<Slice> &outputSlices);

    HcclResult TbeReduceRx(const LINK &linkLeft, std::vector<Slice> &rxSlices, std::vector<Slice> &rxSlicestemp);

    HcclResult TbeReduceRxLastStep(const LINK &linkLeft, InterServerAlgoStep &stepInfo,
        const std::vector<Slice> &inputSlices, const std::vector<Slice> &outputSlices);

    HcclResult RunDestReducerLastStep(const LINK &linkLeft, InterServerAlgoStep &stepInfo,
        const std::vector<Slice> &inputSlices, const std::vector<Slice> &outputSlices);

    HcclResult GetRxSlices(std::vector<Slice> &rxSlices, std::vector<Slice> &rxSlicestemp,
        InterServerAlgoStep &stepInfo, const std::vector<Slice> &inputSlices, const std::vector<Slice> &outputSlices);

    HcclResult SdmaReducer(const u32 nSteps, const LINK &linkLeft, InterServerAlgoStep &stepInfo,
        const std::vector<Slice> &inputSlices, const std::vector<Slice> &outputSlices);

    const u64 reduceAttr_; /* 0x1:表示data_type + reduce_type支持inlinereduce  */

    std::unique_ptr<Sender> senderInfo_;
    std::unique_ptr<Reducer> reducerInfo_;
};
} // hccl

#endif /* REDUCE_SCATTER_NHR_PUB_H */