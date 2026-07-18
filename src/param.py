"""Central tuning entry point.

Tunable parameters for strategy, movement, kicking, and obstacle avoidance
live here. Values were migrated from the code files that used them; prefer
adjusting parameters here going forward.
"""

from __future__ import annotations

import math


# ======================================================================
# Kick power
# ======================================================================

# Player.kick() clamps power into this range. Exceeding it isn't supported by
# the control interface and may cause the robot to fall.
KICK_POWER_MIN = 1.0
KICK_POWER_MAX = 10.0

# Kick power for normal play.
KICK_POWER_DEFAULT = 5.0
KICK_POWER_BACKFIELD = 5.0
KICK_POWER_OUR_KICKOFF = 5.0

# Kick power tiers (QW4). All clamped to [KICK_POWER_MIN, KICK_POWER_MAX].
KICK_POWER_SHOT = 8.0          # full shot on goal
KICK_POWER_CLEAR = 8.0         # hard clearance upfield when in our own half
KICK_POWER_CARRY = 3.5         # gentle nudge to advance the ball (dribble upfield)
# Passing is deliberately SOFT so the receiver can actually collect it (a hard
# pass overshoots and the receiver can't get to it). Tune against sim physics.
PASS_POWER_MIN = 2.0
PASS_POWER_MAX = 4.5
PASS_POWER_PER_M = 0.4         # pass power scales gently with distance to the target


# ======================================================================
# Player movement control
# ======================================================================

# SPEED NOTES: the SDK doc recommends vx <= ~0.3 m/s for early debugging; the
# robot's stable gait max is well below MAX_LINEAR, so the real speed ceiling is
# the gait, not these caps. The biggest perceived-speed win is spending less
# time NOT moving: don't stop to rotate in place unless the target is well
# off-heading (TURN_THRESHOLD), and stay in omni mode longer near targets
# (OMNI_DIST) so robots strafe-while-facing instead of turn-then-walk.
ARRIVE_DIST = 0.15             # Arrival threshold (m)
OMNI_DIST = 1.5                # Below this distance to target, use omnidirectional close-range control (raised 1.0->1.5: strafe-while-facing near targets; also helps the keeper slide to saves while facing the ball)
TURN_THRESHOLD = 1.0           # Long-distance walking: only fully turn in place when heading error exceeds this (rad, ~57deg). Below it, walk while turning (forward speed scaled by cos(error)). Raised 0.5->1.0 to cut stop-and-turn stutter.
MAX_LINEAR = 2.0               # Commanded forward speed cap (m/s). Likely above the gait's real max — raising further probably won't help; lower it if high commands cause instability.
MAX_ANGULAR = 2.0              # Commanded turn speed cap (rad/s)
LINEAR_GAIN = 2.0              # Translation P gain (raised 1.5->2.0 to reach top speed sooner, less ramp-down)
ANGULAR_GAIN = 2.0             # Turning P gain


# ======================================================================
# Player kicking / shot planning
# ======================================================================

KICK_ENTER_M = 2.0             # Enter kick state when closer to the ball than this
KICK_EXIT_M = 2.5              # While kicking, only exit once farther from the ball than this
CHASE_BEHIND_M = 0.35          # Distance behind the ball to stand at while chasing it


# ======================================================================
# World model / ball velocity estimation
# ======================================================================

# Ball velocity is derived by differencing ball position across adjacent
# frames, then EMA-smoothed.
BALL_VEL_SMOOTH_ALPHA = 0.4       # EMA smoothing factor (0..1): higher = more responsive but noisier
BALL_VEL_RESET_DT_S = 0.4         # If the gap between observations exceeds this, discard the diff and reset velocity (s)
BALL_VEL_STATIONARY_SPEED = 0.2   # Below this speed, treat the ball as stationary (m/s), to avoid noise-driven jitter


# ======================================================================
# Player technical action parameters: goalkeeping / support
# ======================================================================

GUARD_FACE_BALL = True
GUARD_THREAT_ENTER_X = -1.0
GUARD_THREAT_EXIT_X = -0.7

# --- Goalkeeper positioning: arc + angle-closing step-out (QW1) ---
# The keeper stands on the line from the ball to own-goal-center, at a
# step-out distance from goal center. The closer/more central the ball, the
# farther it steps out (closing the angle); the farther the ball, the closer
# it hugs the goal line (guarding against chips/rebounds). Step-out distance
# is linearly interpolated based on ball-to-goal distance.
KEEPER_STEP_OUT_MIN_M = 0.35      # Minimum step-out distance (m): hug the goal line when ball is far
KEEPER_STEP_OUT_MAX_M = 1.2       # Maximum step-out distance (m): close the angle when ball is near
KEEPER_BALL_NEAR_M = 2.5          # Ball-to-goal distance <= this -> use max step-out distance (m)
KEEPER_BALL_FAR_M = 6.5           # Ball-to-goal distance >= this -> use min step-out distance (m)
KEEPER_LATERAL_MARGIN_M = 0.35    # Lateral range of motion: half goal width + this extension (m)

# --- Save: when the ball moves fast toward goal, slide to the predicted crossing point to block it ---
KEEPER_SAVE_LINE_M = 0.4          # Shallow stance line the keeper drops back to during a save (from goal center, m)
KEEPER_SAVE_BALL_SPEED = 0.7      # Ball speed >= this and heading at goal -> enter save mode (m/s)
KEEPER_SAVE_EXIT_SPEED = 0.35     # Save hysteresis: ball speed must drop below this to exit save mode (m/s)
KEEPER_SAVE_HORIZON_S = 2.0       # Only save shots predicted to reach the goal line within this time (s)
KEEPER_SAVE_MOUTH_MARGIN_M = 0.5  # Still save if the predicted crossing point is within this range outside the posts (m)

# --- Fixed keeper assignment (QW2) ---
# One robot is permanently the goalkeeper. If that robot is unavailable
# (penalized / fallen / pose unknown) the closest active player to our goal is
# promoted so the net is never abandoned.
KEEPER_PLAYER_ID = 1              # Player id permanently assigned as goalkeeper

# --- Keeper repositioning + sweeping ---
# Repositioning speed: the keeper strafes (slow, heel-to-heel) when it faces the
# ball while moving. So for a large move it faces the travel direction and walks
# forward (fast); only once within KEEPER_SETTLE_DIST_M of its spot does it
# square up to the ball (ready to react, small strafes only).
KEEPER_SETTLE_DIST_M = 0.5       # > this from target -> face travel (fast walk); <= this -> face ball (square)
# Sweeping/claiming: come out to grab a loose ball rather than sit on the line.
KEEPER_CLAIM_DIST_M = 2.0        # come out to claim a loose ball within this range, IF we're the closest robot to it
KEEPER_CLAIM_EXIT_M = 2.7        # keep claiming until the ball is beyond this (hysteresis, avoids in/out flapping)

SUPPORT_DIST_M = 3.0

# ======================================================================
# Normal-phase strategy
# ======================================================================

ATTACKER_KEEP_DIST_MARGIN_M = 0.3  # Prevents attacker selection from oscillating

FALLEN_COST = 10.0  # Distance penalty applied to a fallen player (meters)


# ======================================================================
# Team tactics — possession-aware field-player roles (QW3)
# ======================================================================

# The keeper is fixed; the two field players take dynamic roles. The nearer one
# always goes for the ball (press when defending, attack when we have it); the
# other's role depends on team "mode":
#   DEFEND mode (opponent has the ball in OUR half): second player holds a
#       second-line cover position (defensive support()).
#   ATTACK mode (we have it, or the ball is in their half): second player pushes
#       up as an advanced passing outlet / shooting threat (support_attack()).
POSSESSION_MARGIN_M = 0.4          # Possession is "ours"/"theirs" only when one side is this much closer to the ball; otherwise "contested" (mode is held, avoiding flapping)

SUPPORT_ATTACK_AHEAD_M = 2.0       # Attacking outlet stands this far ahead of the ball (toward opp goal)
SUPPORT_ATTACK_WIDE_M = 1.8        # ...and this far to the open (fewer-opponents) side

# --- Attacker shot / pass selection (QW4/QW6) ---
# Shooting is the PRIORITY: shoot from anywhere with a clear angle to goal, even
# long range. Only pass when there's no shooting angle.
SHOT_RANGE_M = 8.0                 # shoot within this distance of the opponent goal (covers ~most of the field)
SHOT_LANE_RADIUS_M = 0.45          # shot lane counts as blocked if an opponent is within this of the line
PASS_ADVANCE_MARGIN_M = 1.5        # pass only to a teammate at least this much closer to the opp goal
PASS_LANE_RADIUS_M = 0.45          # pass lane blocked radius

# --- Rebound crash + pass reception (QW6) ---
REBOUND_DEPTH_M = 1.2              # when a shot is on, the other robot crashes to this far in front of the opp goal
PASS_RECEIVE_WINDOW_S = 1.2        # after a pass, the intended receiver commits to collecting for this long
BALL_COLLECT_DIST_M = 0.6         # ball counts as collected once a (non-passing) field player is this close

# --- Possession / pressure (QW5) ---
# Retain the ball rather than boot it away: only CLEAR when genuinely pressured
# AND deep; otherwise carry (dribble) or pass to keep possession.
PRESSURE_DIST_M = 1.5              # an opponent within this of the ball = we're under pressure
DANGER_RADIUS_M = 3.5             # ball within this of our own goal = danger zone (clear under pressure; escalate defense)
SUPPORT_DEEP_DIST_M = 1.3         # tight second-line distance from the ball when defending inside the danger zone


# ======================================================================
# Kickoff / set-play strategy
# ======================================================================

# Kickoff
KICKOFF_STAGE_M = 2.0
KICKOFF_FRONT_MARGIN = 0.1
KICKOFF_LATERAL_TOL = 0.35

CENTER_LEAVE_DIST_M = 0.15 # How far the ball must move from the center spot to be considered "in play"

OPP_SET_WALL_DIST_M = 2.0 # Distance beyond which we block during the opponent's set play

# When defending an opponent's set play/kickoff, field players must stay at
# least this far from the ball, or they're sent off for 30s (and the set piece
# is retaken). Rule minimum is 1.45 m; we use 1.7 m to leave a buffer for
# control error and the robot's own radius (over-retreating is always legal for
# the defending side). The separate "ball within 3 m" inactivity rule is a
# different penalty and not this distance.
SET_PLAY_KEEP_CLEAR_M = 1.7
SET_PLAY_DEFENDER_SPREAD_M = 0.8  # Lateral spacing between retreating defenders (m)


# ======================================================================
# Positioning / avoidance
# ======================================================================

OPPONENT_RESTART_AVOID_M = 1.65
CIRCLE_MARGIN_M = 0.3


# ======================================================================
# Kick target geometry
# ======================================================================

GOAL_TARGET_DEPTH_M = 0.25              # Depth of the kick target point inside the goal (m)

# ======================================================================
# Obstacle geometry parameters
# ======================================================================

BALL_OBSTACLE_RADIUS = 0.5              # Ball's obstacle radius (m); used for avoidance calculations
OPPONENT_RADIUS = 0.55                  # Opponent robot radius (m)
TEAMMATE_RADIUS = 0.48                  # Teammate robot radius (m)
SAFETY_MARGIN = 0.22                    # General safety margin (m)

GOAL_DEPTH = 0.6                        # Goal depth (m); used for modeling the goal as an obstacle
POST_RADIUS = 0.18                      # Goalpost radius (m)
NET_RADIUS = 0.20                       # Net radius (m)
NET_STEP = 0.35                         # Net discretization step (m)

START_IGNORE = 0.0                      # Start-point ignore radius (m); obstacles near the start don't affect planning
TARGET_IGNORE = 0.0                     # Target-point ignore radius (m); obstacles near the target don't affect planning

# ======================================================================
# Global path planner (A* Grid Planner)
# ======================================================================

USE_GLOBAL_PATH_PLANNER = True          # Whether to enable the global planner; falls back to local VFH if False
GLOBAL_GRID_RESOLUTION_M = 0.35         # Grid resolution (m/cell); smaller = finer paths but more compute
GLOBAL_FIELD_MARGIN_M = 0.25            # Field boundary expansion margin (m); ensures paths near the boundary are feasible
GLOBAL_OBSTACLE_MARGIN_M = 0.10         # Obstacle inflation margin (m); extra padding added on top of obstacle radius
GLOBAL_PATH_LOOKAHEAD_M = 0.9           # How far ahead along the planned path to extract a waypoint (m)

# ======================================================================
# Local path planner (VFH Direction Scan)
# ======================================================================

PLAN_LOOKAHEAD = 1.2                    # Forward probe ray length (m); the range the planner "looks" ahead
PLAN_CLEARANCE = 0.35                   # Minimum safety clearance (m); candidate direction must be clearer than this ahead
PLAN_STEP = math.radians(15)            # Candidate direction scan step (rad); smaller = finer direction resolution
PLAN_MAX_OFFSET = math.radians(100)     # Maximum deviation from the target direction (rad); scan range is +/- this value

# ======================================================================
# Visualization
# ======================================================================

KICK_TARGET_MARK_SIZE_M = 0.18
