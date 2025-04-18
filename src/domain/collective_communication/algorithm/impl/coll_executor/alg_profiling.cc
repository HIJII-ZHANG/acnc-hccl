/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "alg_profiling.h"

thread_local TaskCallBack g_aivCallBack = nullptr; 
thread_local void* g_aivCallBackUserPtr = nullptr;

HcclResult RegisterAlgCallBack(void* userPtr, TaskCallBack callback)
{
    g_aivCallBackUserPtr = userPtr;
    g_aivCallBack = callback;
    return HCCL_SUCCESS;
}

void SetupTaskParaAiv(AivTaskPara& taskPara, TaskParaAiv& para, HcclRtStream stream, u64 beginTime)
{
    taskPara.isMainStream = true;
    taskPara.stream = stream;
    taskPara.beginTime = beginTime;
    taskPara.aiv = para;
}

HcclResult TaskAivProfiler(HcclCMDType cmdType, u32 tag, u64 size, u32 blockDim, u32 rankSize,
     void* flagMem, rtStream_t stream, s32 aivRdmaStep, uint64_t beginTime)
{
    if(g_aivCallBack == nullptr){
        return HCCL_E_PTR;
    }

    TaskParaAiv para(cmdType, tag, size, blockDim, rankSize, aivRdmaStep, flagMem);
    AivTaskPara taskPara;

    SetupTaskParaAiv(taskPara, para, stream, beginTime);
    g_aivCallBack(g_aivCallBackUserPtr, (void *)&taskPara, sizeof(struct AivTaskPara));

    return HCCL_SUCCESS;
}

