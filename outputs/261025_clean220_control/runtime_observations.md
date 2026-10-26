# 运行时间异常与处理

2026-10-25，对照组seed43的第8、40轮分别记录约515秒和410秒，通常约12秒。独立等待55秒的调用也曾经过约426秒墙钟时间。Windows系统事件日志确认空闲超时进入Modern Standby，随后退出待机，见`power_event_observations.txt`。时间戳是UTC，用户当地时区America/New_York。

训练恢复后损失为有限值，未重启、改变参数或丢弃已有轮次。原先线程级ES_SYSTEM_REQUIRED请求在该系统上未阻止上述暂停。检查时电池100%、已接通电源。

为避免再次空闲待机，增加`prevent_idle_standby.py`：仅在原训练PID仍存在、且接通电源时请求SYSTEM_REQUIRED和DISPLAY_REQUIRED；轮询进程结束、电源切换，结束时清除请求并关闭句柄，最多12小时。不修改永久电源设置；用户主动休眠、合盖等操作仍可能暂停训练。原训练内的请求也在进程结束时释放。实际释放记录见`standby_guard.log`和`run.log`。

这是主机运行保障，不改变模型、数据、训练参数或选择规则。训练耗时包含主机暂停，不能直接作为纯GPU速度基准。全轮次数值和最终产物由`verify_delivery.py`验收。

参考微软API文档：[SetThreadExecutionState](https://learn.microsoft.com/en-us/windows/win32/api/winbase/nf-winbase-setthreadexecutionstate)、[PowerSetRequest](https://learn.microsoft.com/en-us/windows/win32/api/winbase/nf-winbase-powersetrequest)。后者说明Modern Standby及用户主动休眠对请求的限制；本次辅助程序使用前者，不声称绕过系统所有休眠策略。
