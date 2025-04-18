/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_ENV_CONFIG_H
#define HCCL_ENV_CONFIG_H

#include <vector>
#include <hccl/hccl_types.h>
#include "base.h"

/*************** Interfaces ***************/
using HcclSocketPortRange = struct HcclSocketPortRangeDef {
    u32 min;
    u32 max;
};

enum SocketLocation {
    SOCKET_HOST = 0,
    SOCKET_NPU = 1
};

HcclResult InitEnvConfig();

bool GetExternalInputHostPortSwitch();

bool GetExternalInputNpuPortSwitch();

const std::vector<HcclSocketPortRange> &GetExternalInputHostSocketPortRange();

const std::vector<HcclSocketPortRange> &GetExternalInputNpuSocketPortRange();

/*************** For Internal Use ***************/

struct EnvConfig {
    // 初始化标识
    bool initialized;

    // 环境变量参数
    bool hostSocketPortSwitch; // HCCL_HOST_SOCKET_PORT_RANGE 环境变量配置则开启；否则关闭
    bool npuSocketPortSwitch; // HCCL_NPU_SOCKET_PORT_RANGE 环境变量配置则开启；否则关闭
    std::vector<HcclSocketPortRange> hostSocketPortRange;
    std::vector<HcclSocketPortRange> npuSocketPortRange;

    EnvConfig()
    : hostSocketPortSwitch(false),
    npuSocketPortSwitch(false),
    hostSocketPortRange(),
    npuSocketPortRange()
    {
    }
};

HcclResult InitEnvParam();

HcclResult ParseHostSocketPortRange();

HcclResult ParseNpuSocketPortRange();

HcclResult CheckSocketPortRangeValid(const std::string &envName, const std::vector<HcclSocketPortRange> &portRanges);

HcclResult PortRangeSwitchOn(const SocketLocation &socketLoc);

void PrintSocketPortRange(const std::string &envName, const std::vector<HcclSocketPortRange> &portRangeVec);

#endif // HCCL_ENV_INPUT_H