# teleop_command

`teleop_command` 是 ELRS/CRSF 遥控输入节点。它从串口读取 CRSF 通道数据，将通道转换为 `sensor_msgs/msg/Joy`，并根据当前拨杆/按钮组合发布机器人速度、姿态和有限状态机命令。

对应核心源码：

- `src/teleop_command_node.cpp`
- `include/teleop_command/teleop_command_node.hpp`
- `config/param.yaml`

## 启动方式

编译并加载环境：

```bash
colcon build --symlink-install --packages-up-to teleop_command
source install/setup.bash
```

启动节点：

```bash
ros2 launch teleop_command teleop_command.launch.py
```

默认参数文件为 `config/param.yaml`。默认串口为 `/dev/ttyTHS1`，如硬件连接不同，需要修改 `uart_interface`。

## 发布话题

| 话题 | 类型 | 说明 |
| --- | --- | --- |
| `joy` | `sensor_msgs/msg/Joy` | 归一化后的遥控器输入 |
| `command/cmd_twist` | `geometry_msgs/msg/Twist` | 速度命令 |
| `command/cmd_pose` | `geometry_msgs/msg/PoseStamped` | 姿态/位姿命令 |
| `command/cmd_key` | `std_msgs/msg/String` | 状态机目标命令 |

`command/cmd_key` 使用 transient local QoS，只在状态机命令变化时发布。

## 主要参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `uart_interface` | `/dev/ttyTHS1` | CRSF 串口设备 |
| `update_rate` | `10` | 状态机命令检查频率，单位 Hz |
| `use_sdk` | `false` | 是否切换到 SDK 输入模式 |
| `joystick_deadzone` | `0.013` | CH1-CH4 摇杆死区 |
| `speed_ratio` | `[0.33, 0.66, 1.0]` | 低/中/高速档速度比例 |
| `max_twist_linear` | `3.0` | 最大线速度比例 |
| `max_twist_angular` | `6.0` | 最大角速度比例 |
| `max_roll` | `0.2` | 最大 roll 命令 |
| `max_pitch` | `0.4` | 最大 pitch 命令 |

## 通道与操作映射

代码中将 CRSF 通道转换成 `Joy` 的 `axes`。下表中的 `+1/0/-1` 是节点内部的 `Joy` 值，实际物理拨杆位置取决于遥控器混控配置。

| CRSF 通道 | Joy 字段 | 操作 |
| --- | --- | --- |
| CH1 | `axes[0]` | CH6=0 时控制左右速度 `linear.y`；CH6=-1 时控制 `roll` |
| CH2 | `axes[1]` | CH6=0 时控制高度速度 `linear.z`；CH6=-1 时控制 `pitch` |
| CH3 | `axes[2]` | 控制前后速度 `linear.x` |
| CH4 | `axes[3]` | 控制偏航角速度 `angular.z` |
| CH5 | `axes[4]` | 状态机主拨杆 |
| CH6 | `axes[5]` | 运动/姿态/状态机组合拨杆 |
| CH7 | `axes[6]` | 跳跃触发拨杆 |
| CH8 | `axes[7]` | 速度档位选择 |
| CH9 | button mode | 模式按钮，值为 `8` 时切换 `use_sdk` |

### 速度与姿态

| 条件 | 操作 |
| --- | --- |
| CH6 `axes[5] == 0` | CH1 控制 `linear.y`，CH2 控制 `linear.z` |
| CH6 `axes[5] == -1` | CH1 控制 `roll`，CH2 控制 `pitch` |
| CH8 `axes[7] == +1` | 使用 `speed_ratio[0]`，低速档 |
| CH8 `axes[7] == 0` | 使用 `speed_ratio[1]`，中速档 |
| CH8 `axes[7] == -1` | 使用 `speed_ratio[2]`，高速档 |

### 状态机命令

| 条件 | 发布到 `command/cmd_key` 的命令 |
| --- | --- |
| CH5 `axes[4] == +1` | `transform_down` |
| CH5 `axes[4] == -1` 且 CH6 `axes[5] == +1` | `transform_up` |
| CH5 `axes[4] == -1` 且 CH6 `axes[5] == 0/-1` | `rl_0` |
| 上述 `rl_0` 条件下，CH7 `axes[6] == -1` | `jump` |
| 上述 `rl_0` 条件下，`Joy button[2] == 1` | `rl_1`，预留/SDK 输入；当前 CRSF 转换只主动写入 `buttons[0]` |
| 上述 `rl_0` 条件下，`Joy button[3] == 1` | `rl_2`，预留/SDK 输入；当前 CRSF 转换只主动写入 `buttons[0]` |
| 上述 `rl_0` 条件下，`Joy button[4] == 1` | `rl_3`，预留/SDK 输入；当前 CRSF 转换只主动写入 `buttons[0]` |

## 终端提示

节点启动后会在终端打印一份操作提示，包含通道、按钮与对应动作。运行过程中，当 `command/cmd_key` 状态机命令发生变化时，也会打印当前命令；当 CH9 模式值为 `8` 时，会打印 `use_sdk` 输入模式的启用/关闭状态。

## 调试建议

查看归一化后的遥控输入：

```bash
ros2 topic echo /joy
```

查看速度命令：

```bash
ros2 topic echo /command/cmd_twist
```

查看状态机命令：

```bash
ros2 topic echo /command/cmd_key
```
