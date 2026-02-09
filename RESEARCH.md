# Reachy Mini Animation Research

## SDK Interpolation Methods

The SDK has 4 built-in interpolation methods in `goto_target()`:

```python
mini.goto_target(head=pose, duration=0.5, method="minjerk")  # or "linear", "ease", "cartoon"
```

### Interpolation Curves (t=0 to t=1)

| Method     | t=0.25 | t=0.50 | t=0.75 | Character |
|------------|--------|--------|--------|-----------|
| LINEAR     | 0.250  | 0.500  | 0.750  | Robotic, constant speed |
| MIN_JERK   | 0.104  | 0.500  | 0.896  | Natural, slow-fast-slow |
| EASE_IN_OUT| 0.125  | 0.500  | 0.875  | Similar to minjerk |
| CARTOON    | -0.100 | 0.500  | 1.100  | **Anticipation + Overshoot!** |

### CARTOON is key for "pleasant" animation!
- **Anticipation**: Goes negative (-0.1) before moving forward (like winding up)
- **Overshoot**: Goes past target (1.1) then settles back (1.0) (follow-through)
- This implements Disney's animation principles natively!

## How Official Face Tracking Works

From `camera_worker.py`:
1. Detect face position (u, v) in image
2. Call `look_at_image(u, v, duration=0, perform_movement=False)` to get target pose
3. Extract translation/rotation, **scale by 0.6**
4. Store as offsets for MovementManager

From `moves.py` (MovementManager):
1. 100Hz control loop
2. Read face tracking offsets
3. Create secondary pose with `create_head_pose(x, y, z, roll, pitch, yaw)`
4. Compose with primary pose: `compose_world_offset(primary, secondary)`
5. Send via `set_target(head=combined)`

## Key Insight: set_target vs goto_target

- **`set_target()`**: Instant position command. The daemon just sends this to servos.
- **`goto_target()`**: Interpolated movement. SDK handles smooth trajectory.

For face tracking, the official code uses `set_target()` at 100Hz, NOT `goto_target()`.
The smoothness comes from:
1. Offset scaling (0.6)
2. The compose_world_offset math
3. High update rate (100Hz)
4. Servo internal controllers

## Motor Limits

| Joint | Range |
|-------|-------|
| Head Pitch/Roll | ±40° |
| Head Yaw | ±180° |
| Body Yaw | ±160° |

## What Makes Motion "Pleasant"

Based on Disney animation principles + robotics research:

1. **No sudden starts/stops**: Use min_jerk or ease curves
2. **Anticipation**: Slight movement opposite before main action
3. **Follow-through**: Slight overshoot, then settle
4. **Slow in/slow out**: Accelerate/decelerate gradually
5. **Arcs**: Natural motion follows curved paths
6. **Secondary motion**: Antennas can add personality

## Recommended Approach for Face Tracking

1. Use YOLO detection at ~25Hz
2. Smooth detection coordinates (low-pass filter on u,v)
3. Use `look_at_image(perform_movement=False)` to get target pose
4. Scale offsets by 0.5-0.7
5. Apply offsets via `set_target()` at 100Hz
6. OR use `goto_target(duration=0.1, method="minjerk")` at lower rate

## The Simplest Pleasant Motion

For a single smooth movement:
```python
# This uses minjerk internally - smooth!
reachy.goto_target(head=target_pose, duration=0.5, method="minjerk")
```

For continuous tracking, try CARTOON method with short durations for playful feel:
```python
reachy.goto_target(head=pose, duration=0.15, method="cartoon")
```

## Next Steps to Try

1. Use `goto_target()` with `method="cartoon"` for playful motion
2. Increase duration (0.15-0.3s) for smoother feel
3. Reduce detection rate to ~10Hz so movements complete before new target
4. Add idle animations (breathing, glancing) from official BreathingMove
