"""Python logging -> ROS topic bridge -- publishes agent logs to ROS so they can be recorded to a rosbag and replayed for debugging.

Framework side: ros_source calls install(node) once the node is ready,
attaching a handler to the root logger. Not installed when the dev
machine has no ROS; logs continue to flow through the platform logger
unaffected.

Topic: /soccer/agent_log, type rcl_interfaces/msg/Log (a standard ROS log
message, natively supported by Studio).
"""

from __future__ import annotations

import logging

_log = logging.getLogger(__name__)

_TOPIC = "/soccer/agent_log"


def install(node) -> None:
    """Attach the ROS log publisher to the Python root logger (Docker-only)."""
    try:
        handler = _RosLogHandler(node)
        # attach to the root logger to capture all modules; level inherits from root (default INFO)
        logging.getLogger().addHandler(handler)
        _log.info("log_publisher installed, publishing to %s", _TOPIC)
    except Exception as exc:
        _log.warning("log_publisher install failed (ROS log disabled): %s", exc)


# logging level -> ROS Log constant mapping
_LEVEL_MAP = {
    logging.DEBUG: 10,     # Log.DEBUG
    logging.INFO: 20,      # Log.INFO
    logging.WARNING: 30,   # Log.WARN
    logging.ERROR: 40,     # Log.ERROR
    logging.CRITICAL: 50,  # Log.FATAL
}


class _RosLogHandler(logging.Handler):
    """logging.Handler: converts a LogRecord into rcl_interfaces/msg/Log and publishes it to /rosout."""

    def __init__(self, node) -> None:
        super().__init__()
        from rcl_interfaces.msg import Log
        from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy

        # standard /rosout QoS: TRANSIENT_LOCAL + RELIABLE, history=1000
        qos = QoSProfile(
            depth=1000,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE,
        )
        self._node = node
        self._pub = node.create_publisher(Log, _TOPIC, qos)

    def emit(self, record: logging.LogRecord) -> None:
        """Convert to a ROS Log message and publish to /rosout. Exceptions here can't be logged (would recurse), so swallow them."""
        try:
            from rcl_interfaces.msg import Log

            msg = Log()
            msg.stamp = self._node.get_clock().now().to_msg()
            msg.level = _LEVEL_MAP.get(record.levelno, 20)  # default INFO
            msg.name = record.name
            msg.msg = record.getMessage()
            msg.file = record.pathname
            msg.function = record.funcName
            msg.line = record.lineno
            self._pub.publish(msg)
        except Exception:
            pass  # can't call logging again (would recurse), so swallow the exception
