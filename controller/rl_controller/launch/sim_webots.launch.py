#!/usr/bin/env python
import os
import re
import launch
from launch import LaunchDescription
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from webots_ros2_driver.webots_launcher import WebotsLauncher, Ros2SupervisorLauncher
from webots_ros2_driver.webots_controller import WebotsController

from webots_ros2_driver.urdf_spawner import URDFSpawner, get_webots_driver_node
from launch.actions import OpaqueFunction
import xacro


def _extract_extern_robot_names(world_path):
    with open(world_path, "r", encoding="utf-8") as world_file:
        lines = world_file.readlines()

    robot_names = []
    depth = 0
    in_robot_block = False
    robot_name = None
    has_extern_controller = False

    for line in lines:
        stripped = line.strip()
        if not in_robot_block and re.match(r"(DEF\s+\w+\s+)?Robot\s*{", stripped):
            in_robot_block = True
            depth = line.count("{") - line.count("}")
            robot_name = None
            has_extern_controller = False
            if depth == 0 and has_extern_controller and robot_name:
                robot_names.append(robot_name)
                in_robot_block = False
            continue

        if not in_robot_block:
            continue

        if depth == 1:
            name_match = re.match(r'name\s+"([^"]+)"', stripped)
            if name_match:
                robot_name = name_match.group(1)
            if stripped == 'controller "<extern>"':
                has_extern_controller = True

        depth += line.count("{") - line.count("}")
        if depth == 0:
            if has_extern_controller and robot_name:
                robot_names.append(robot_name)
            in_robot_block = False

    return robot_names


def _resolve_world_robot_name(requested_robot_name, extern_robot_names):
    non_supervisor_robots = [name for name in extern_robot_names if name != "Ros2Supervisor"]
    if not non_supervisor_robots:
        return requested_robot_name, False

    candidates = [
        requested_robot_name,
        f"{requested_robot_name}_webots",
    ]
    for candidate in candidates:
        if candidate in non_supervisor_robots:
            return candidate, True

    if len(non_supervisor_robots) == 1:
        return non_supervisor_robots[0], True

    raise RuntimeError(
        f"World has multiple extern robots {non_supervisor_robots}, unable to select one for "
        f'"{requested_robot_name}".'
    )


def launch_setup(context, *args, **kwargs):
    robot_name = LaunchConfiguration("robot").perform(context)
    ns = LaunchConfiguration("ns").perform(context)
    terrain = LaunchConfiguration("terrain").perform(context)
    controllers_file = LaunchConfiguration("controllers_file").perform(context)
    world_path = os.path.join(
        get_package_share_directory("webots_bridge"),
        "worlds",
        terrain + ".wbt",
    )
    extern_robot_names = _extract_extern_robot_names(world_path)
    webots_robot_name, use_embedded_robot = _resolve_world_robot_name(robot_name, extern_robot_names)
    has_embedded_supervisor = "Ros2Supervisor" in extern_robot_names

    robot_xacro_path = os.path.join(
        get_package_share_directory(robot_name + "_description"),
        "xacro",
        "robot.xacro",
    )

    robot_description = xacro.process_file(
        robot_xacro_path, mappings={"hw_env": "webots"}
    ).toxml()
    spawn_robot = None
    if not use_embedded_robot:
        spawn_robot = URDFSpawner(
            name=robot_name,
            robot_description=robot_description,
            # relative_path_prefix=os.path.join(robot_name + "_description", 'resource'),
            translation="0 0 0.4",
            rotation="0 0 0 0",
        )

    webots = WebotsLauncher(
        world=world_path,
        ros2_supervisor=not has_embedded_supervisor,
    )
    supervisor = Ros2SupervisorLauncher() if has_embedded_supervisor else webots._supervisor

    robot_controllers = os.path.join(
        get_package_share_directory("rl_controller"),
        "config",
        robot_name,
        controllers_file,
    )

    tita_driver = WebotsController(
        robot_name=webots_robot_name,
        parameters=[
            {"robot_description": robot_description},
            {"xacro_mappings": ["name:=" + robot_name]},
            {"use_sim_time": True},
            {"set_robot_state_publisher": False},
            robot_controllers,
        ],
        respawn=True,
        namespace=ns,
    )

    robot_state_pub_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="both",
        parameters=[
            {"robot_description": robot_description},
            # {"robot_description": '<robot name=""><link name=""/></robot>'},
            {"frame_prefix": ns + "/"},
        ],
        namespace=ns,
    )
    joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "joint_state_broadcaster",
            "--controller-manager",
            ns + "/controller_manager",
        ],
    )

    imu_sensor_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "imu_sensor_broadcaster",
            "--controller-manager",
            ns + "/controller_manager",
        ],
    )
    rl_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            robot_name + "_rl_controller",
            "--controller-manager",
            ns + "/controller_manager",
        ],
    )

    def get_ros2_nodes(*args):
        ros2_nodes = [
            robot_state_pub_node,
            tita_driver,
            joint_state_broadcaster_spawner,
            imu_sensor_broadcaster_spawner,
            rl_controller_spawner,
        ]
        if spawn_robot is None:
            return ros2_nodes

        return [
            spawn_robot,
            launch.actions.RegisterEventHandler(
                event_handler=launch.event_handlers.OnProcessIO(
                    target_action=spawn_robot,
                    on_stdout=lambda event: get_webots_driver_node(event, ros2_nodes),
                )
            ),
        ]

    webots_event_handler = launch.actions.RegisterEventHandler(
        event_handler=launch.event_handlers.OnProcessExit(
            target_action=webots,
            on_exit=[launch.actions.EmitEvent(event=launch.events.Shutdown())],
        )
    )

    ros2_reset_handler = launch.actions.RegisterEventHandler(
        event_handler=launch.event_handlers.OnProcessExit(
            target_action=supervisor,
            on_exit=get_ros2_nodes,
        )
    )

    return [
        webots,
        supervisor,
        webots_event_handler,
        ros2_reset_handler,
    ] + get_ros2_nodes()


def generate_launch_description():
    declared_arguments = []
    declared_arguments.append(
        launch.actions.DeclareLaunchArgument(
            "robot",
            default_value="tita",
            description="Path to the robot description file",
        )
    )
    declared_arguments.append(
        launch.actions.DeclareLaunchArgument(
            "ns",
            default_value="",
            description="Namespace of launch",
        )
    )
    declared_arguments.append(
        launch.actions.DeclareLaunchArgument(
            "terrain",
            default_value="empty_world",
            description="Terrain of webots world",
            choices=[
                "empty_world",
                "tita",
                "stairs",
                "uneven",
            ]
        )
    )
    declared_arguments.append(
        launch.actions.DeclareLaunchArgument(
            "controllers_file",
            default_value="controllers.yaml",
            description="Controller YAML file under rl_controller/config/<robot>",
        )
    )
    return LaunchDescription(
        declared_arguments + [OpaqueFunction(function=launch_setup)]
    )
