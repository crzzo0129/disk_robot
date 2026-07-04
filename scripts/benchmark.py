from mujoco.mjx import benchmark
import mujoco

model = mujoco.MjModel.from_xml_path("assets/disk_quadruped_extreme.xml")
jit_time, run_time, steps = benchmark(model, batch_size=8192)
print("jit_time:", jit_time)
print("physics steps/sec:", steps / run_time)
