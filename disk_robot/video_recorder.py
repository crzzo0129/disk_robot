"""Small streaming video recorder for MuJoCo simulation scripts."""
from __future__ import annotations

from pathlib import Path


class MujocoVideoRecorder:
    def __init__(
        self,
        model,
        output_path: Path,
        track_body_id: int,
        *,
        fps: int = 30,
        width: int = 960,
        height: int = 540,
        azimuth: float = 90.0,
        elevation: float = -10.0,
        distance: float = 1.4,
    ) -> None:
        import imageio.v2 as imageio
        import mujoco

        self._mujoco = mujoco
        self._track_body_id = int(track_body_id)
        self._frame_period = 1.0 / fps
        self._next_frame_time = 0.0
        output_path = Path(output_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path = output_path
        model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), width)
        model.vis.global_.offheight = max(int(model.vis.global_.offheight), height)
        self._renderer = mujoco.Renderer(model, height=height, width=width)
        self._writer = imageio.get_writer(
            str(output_path),
            fps=fps,
            codec="libx264",
            pixelformat="yuv420p",
            macro_block_size=1,
            quality=8,
        )
        self._camera = mujoco.MjvCamera()
        self._camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        self._camera.trackbodyid = self._track_body_id
        self._camera.azimuth = azimuth
        self._camera.elevation = elevation
        self._camera.distance = distance
        self.frame_count = 0

    def __enter__(self):
        return self

    def capture(self, data) -> None:
        if float(data.time) + 1e-9 < self._next_frame_time:
            return
        self._camera.lookat[:] = data.xpos[self._track_body_id]
        self._renderer.update_scene(data, camera=self._camera)
        self._writer.append_data(self._renderer.render())
        self.frame_count += 1
        while self._next_frame_time <= float(data.time) + 1e-9:
            self._next_frame_time += self._frame_period

    def close(self) -> None:
        self._writer.close()
        self._renderer.close()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
