/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "env_config.h"
#include <mutex>
#include <regex>
#include <sstream>
#include <string>
#include "adapter_error_manager_pub.h"
#include "log.h"
#include "sal_pub.h"

using namespace hccl;

static std::mutex g_envConfigMutex;
static EnvConfig g_envConfig;

constexpr char ENV_EMPTY_STRING[] = "EmptyString";

constexpr char HCCL_AUTO_PORT_CONFIG[] = "auto"; // 端口范围配置为auto时，由OS分配浮动监听端口
constexpr u32 MAX_PORT_NUMBER = 65535; // 合法端口号的上限
constexpr u32 HCCL_SOCKET_PORT_RANGE_AUTO = 0; // 需要保留的

HcclResult InitEnvConfig()
{
    std::lock_guard<std::mutex> lock(g_envConfigMutex);
    if (g_envConfig.initialized) {
        return HCCL_SUCCESS;
    }
    // 初始化环境变量
    CHK_RET(InitEnvParam());

    g_envConfig.initialized = true;

    return HCCL_SUCCESS;
}

bool GetExternalInputHostPortSwitch()
{
    std::lock_guard<std::mutex> lock(g_envConfigMutex);
    return g_envConfig.hostSocketPortSwitch;
}

bool GetExternalInputNpuPortSwitch()
{
    std::lock_guard<std::mutex> lock(g_envConfigMutex);
    return g_envConfig.npuSocketPortSwitch;
}


const std::vector<HcclSocketPortRange> &GetExternalInputHostSocketPortRange()
{
    std::lock_guard<std::mutex> lock(g_envConfigMutex);
    return g_envConfig.hostSocketPortRange;
}

const std::vector<HcclSocketPortRange> &GetExternalInputNpuSocketPortRange()
{
    std::lock_guard<std::mutex> lock(g_envConfigMutex);
    return g_envConfig.npuSocketPortRange;
}
HcclResult InitEnvParam()
{
    HcclResult ret = ParseHostSocketPortRange();
    RPT_ENV_ERR(ret != HCCL_SUCCESS, "EI0001", std::vector<std::string>({"env", "tips"}),
        std::vector<std::string>({"HCCL_HOST_SOCKET_PORT_RANGE", "Please check whether the port range is valid."}));
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[InitEnvParam]errNo[0x%016llx] In init environtment param, parse "
            "HCCL_HOST_SOCKET_PORT_RANGE failed. errorno[%d]", HCCL_ERROR_CODE(ret), ret), ret);

    ret = ParseNpuSocketPortRange();
    RPT_ENV_ERR(ret != HCCL_SUCCESS, "EI0001", std::vector<std::string>({"env", "tips"}),
        std::vector<std::string>({"HCCL_NPU_SOCKET_PORT_RANGE", "Please check whether the port range is valid."}));
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[InitEnvParam]errNo[0x%016llx] In init environtment param, parse "
            "HCCL_NPU_SOCKET_PORT_RANGE failed. errorno[%d]", HCCL_ERROR_CODE(ret), ret), ret);

    return HCCL_SUCCESS;
}

HcclResult SetDefaultSocketPortRange(const SocketLocation &socketLoc, std::vector<HcclSocketPortRange> &portRangeVec)
{
    if (socketLoc == SOCKET_HOST) {
        g_envConfig.hostSocketPortSwitch = false;
        portRangeVec.clear();
        HCCL_RUN_WARNING("HCCL_HOST_SOCKET_PORT_RANGE is not set. Multi-process will not be supported!");
    } else if (socketLoc == SOCKET_NPU) {
        g_envConfig.npuSocketPortSwitch = false;
        portRangeVec.clear();
        HCCL_RUN_WARNING("HCCL_NPU_SOCKET_PORT_RANGE is not set. Multi-process will not be supported!");
    } else {
        HCCL_ERROR("[SetDefaultSocketPortRange] undefined socket location, fail to init socket port range by default.");
        return HCCL_E_NOT_SUPPORT;
    }
    return HCCL_SUCCESS;
}

HcclResult CheckSocketPortRangeValid(const std::string &envName, const std::vector<HcclSocketPortRange> &portRanges)
{
    std::vector<HcclSocketPortRange> rangeVec(portRanges.begin(), portRanges.end());
    std::sort(rangeVec.begin(), rangeVec.end(), [](auto &a, auto &b) {
        return a.min == b.min ? a.max < b.max : a.min < b.min;
    });
    for (size_t i = 0; i < rangeVec.size(); ++i) {
        // the socket range should not be inverted
        CHK_PRT_RET(rangeVec[i].min > rangeVec[i].max,
            HCCL_ERROR("[Check][PortRangeValid] In %s, in socket port range [%u, %u], "
                "the lower bound is greater than the upper bound.",
                envName.c_str(), rangeVec[i].min, rangeVec[i].max),
            HCCL_E_PARA);

        // the socket range should not include the reserved port for auto listening.
        CHK_PRT_RET(rangeVec[i].min <= HCCL_SOCKET_PORT_RANGE_AUTO && rangeVec[i].max >=  HCCL_SOCKET_PORT_RANGE_AUTO,
            HCCL_ERROR("[Check][PortRangeValid] In %s, socket port range [%u, %u] includes "
                "the reserved port number [%u]. please do not use port [%u] in socket port range.",
                envName.c_str(), rangeVec[i].min, rangeVec[i].max, HCCL_SOCKET_PORT_RANGE_AUTO,
                HCCL_SOCKET_PORT_RANGE_AUTO),
            HCCL_E_PARA);

        // the socket range should not exceed the maximum port number
        CHK_PRT_RET(rangeVec[i].max > MAX_PORT_NUMBER,
            HCCL_ERROR("[Check][PortRangeValid] In %s, in socket port range [%u, %u], "
                "the upper bound exceed max port number[%u].",
                envName.c_str(), rangeVec[i].min, rangeVec[i].max, MAX_PORT_NUMBER),
            HCCL_E_PARA);

        // the socket range should not be overlapped
        CHK_PRT_RET(i != 0 && rangeVec[i - 1].max >= rangeVec[i].min,
            HCCL_ERROR("[Check][PortRangeValid] In %s, "
                "socket port range [%u, %u] is conflict with socket port range [%u, %u].",
                envName.c_str(), rangeVec[i - 1].min, rangeVec[i - 1].max, rangeVec[i].min, rangeVec[i].max),
            HCCL_E_PARA);
    }

    return HCCL_SUCCESS;
}

HcclResult SplitHcclSocketPortRange(const std::string &envName, std::string &portRangeConfig,
    std::vector<HcclSocketPortRange> &portRangeVec)
{
    portRangeConfig += ",";
    // verify whether the string format is valid
    std::regex inputFormatPattern(R"((\d{1,5}((-\d{1,5})?,))+)");
    bool validFormat = std::regex_match(portRangeConfig.begin(), portRangeConfig.end(), inputFormatPattern);
    CHK_PRT_RET(!validFormat,
        HCCL_ERROR("[Split][HcclSocketPortRange]errNo[0x%016llx] %s is invalid, please check the format.",
            HCCL_ERROR_CODE(HCCL_E_PARA), envName.c_str()), HCCL_E_PARA);

    // load socket port range one by one
    std::regex rangePattern(R"(\d{1,5}(-\d{1,5})?)");
    std::sregex_iterator iter(portRangeConfig.begin(), portRangeConfig.end(), rangePattern);
    std::sregex_iterator end;
    for (std::sregex_iterator it = iter; it != end; ++it) {
        std::smatch match = *it;
        std::string rangeStr = match.str();
        std::size_t found = rangeStr.find("-");
        HcclSocketPortRange portRange = {};
        if (found == std::string::npos) {
            SalStrToULong(rangeStr, HCCL_BASE_DECIMAL, portRange.min);
            portRange.max = portRange.min;
        } else {
            SalStrToULong(rangeStr.substr(0, found), HCCL_BASE_DECIMAL, portRange.min);
            SalStrToULong(rangeStr.substr(found + 1), HCCL_BASE_DECIMAL, portRange.max);
        }
        portRangeVec.emplace_back(portRange);
        HCCL_INFO("[Split][HcclSocketPortRange] Load hccl socket port range [%u, %u] from %s",
            portRange.min, portRange.max, envName.c_str());
    }
    CHK_RET(CheckSocketPortRangeValid(envName, portRangeVec));
    return HCCL_SUCCESS;
}

HcclResult PortRangeSwitchOn(const SocketLocation &socketLoc)
{
    if (socketLoc == SOCKET_HOST) {
        g_envConfig.hostSocketPortSwitch = true;
        HCCL_INFO("HCCL_HOST_SOCKET_PORT_RANGE is set, switch on.");
    } else if (socketLoc == SOCKET_NPU) {
        g_envConfig.npuSocketPortSwitch = true;
        HCCL_INFO("HCCL_NPU_SOCKET_PORT_RANGE is set, switch on.");
    } else {
        HCCL_ERROR("[PortRangeSwitchOn] undefined socket location, fail to init socket port range by default.");
        return HCCL_E_NOT_SUPPORT;
    }
    return HCCL_SUCCESS;
}

void PrintSocketPortRange(const std::string &envName, const std::vector<HcclSocketPortRange> &portRangeVec)
{
    // assemble port ranges into a string to print the result range
    std::ostringstream portRangeOss;
    for (auto range : portRangeVec) {
        portRangeOss << " [" << std::to_string(range.min) << ", " << std::to_string(range.max) << "]";
    }
    HCCL_RUN_INFO("%s is set to%s.", envName.c_str(), portRangeOss.str().c_str());
}

HcclResult SetSocketPortRange(const std::string &envName, const std::string &socketPortRange,
    const SocketLocation &socketLoc, std::vector<HcclSocketPortRange> &portRangeVec)
{
    portRangeVec.clear();

    // the environment variable is not set
    if (!socketPortRange.compare(ENV_EMPTY_STRING)) {
        CHK_RET(SetDefaultSocketPortRange(socketLoc, portRangeVec));
        return HCCL_SUCCESS;
    }

    // the socket port range is set to auto, then the os will listen on the ports dymamically and automatically.
    if (!socketPortRange.compare(HCCL_AUTO_PORT_CONFIG)) {
        HcclSocketPortRange autoSocketPortRange = {
            HCCL_SOCKET_PORT_RANGE_AUTO,
            HCCL_SOCKET_PORT_RANGE_AUTO
        };
        portRangeVec.emplace_back(autoSocketPortRange);
        CHK_RET(PortRangeSwitchOn(socketLoc));
        HCCL_RUN_INFO("%s is set to %s as [%u, %u].", envName.c_str(), HCCL_AUTO_PORT_CONFIG,
            autoSocketPortRange.min, autoSocketPortRange.max);
        return HCCL_SUCCESS;
    }

    std::string portRangeConfig = socketPortRange;
    // the environment variable is set to an empty string
    portRangeConfig.erase(std::remove(portRangeConfig.begin(), portRangeConfig.end(), ' '), portRangeConfig.end());
    if (portRangeConfig.empty()) {
        CHK_RET(SetDefaultSocketPortRange(socketLoc, portRangeVec));
        return HCCL_SUCCESS;
    }
    // load ranges from string
    CHK_RET(SplitHcclSocketPortRange(envName, portRangeConfig, portRangeVec));
    if (portRangeVec.size() == 0) {
        HCCL_ERROR("Load empty port range from %s, please check.", envName.c_str());
        return HCCL_E_PARA;
    }
    CHK_RET(PortRangeSwitchOn(socketLoc));
    (void) PrintSocketPortRange(envName, portRangeVec);
    return HCCL_SUCCESS;
}

HcclResult ParseHostSocketPortRange()
{
    std::string hostSocketPortRangeEnv = SalGetEnv("HCCL_HOST_SOCKET_PORT_RANGE");
    CHK_RET(SetSocketPortRange("HCCL_HOST_SOCKET_PORT_RANGE", hostSocketPortRangeEnv, SOCKET_HOST,
        g_envConfig.hostSocketPortRange));
    return HCCL_SUCCESS;
}

HcclResult ParseNpuSocketPortRange()
{
    std::string npuSocketPortRangeEnv = SalGetEnv("HCCL_NPU_SOCKET_PORT_RANGE");
    CHK_RET(SetSocketPortRange("HCCL_NPU_SOCKET_PORT_RANGE", npuSocketPortRangeEnv, SOCKET_NPU,
        g_envConfig.npuSocketPortRange));
    return HCCL_SUCCESS;
}