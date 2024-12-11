import numpy as np
from simulator import Simulator
import pinocchio as pin
from pathlib import Path
import matplotlib.pyplot as plt
import os
from numpy import sin,cos,pi

current_dir = os.path.dirname(os.path.abspath(__file__))
xml_path = os.path.join(current_dir, "robots/universal_robots_ur5e/ur5e.xml")
model = pin.buildModelFromMJCF(xml_path)
data = model.createData()

def Rx(q):
    Rx = np.array([[1,         0,         0 ],
                  [0, np.cos(q), -np.sin(q) ],
                  [0, np.sin(q),  np.cos(q) ],], dtype=float)
    return Rx


def Ry(q):
    Ry = np.array([[ np.cos(q),0, np.sin(q)],
                  [         0, 1,         0],
                  [-np.sin(q), 0, np.cos(q)],], dtype=float)
    return Ry


def Rz(q):
    Rz = np.array([[np.cos(q), -np.sin(q),0],
                  [np.sin(q),  np.cos(q), 0],
                  [        0,          0, 1]], dtype=float)
    return Rz


def desired_pose_func(center, radius, angular_velocity, t):
    # desired end effector position (circular trajectory)
    r_desired = np.array([center[0] + radius*cos(angular_velocity * t), 
                          center[1] + radius*sin(angular_velocity * t), 
                          center[2]])

    # desired end effector velocity (position and orientation)
    dr_desired = np.array([-angular_velocity * radius * sin(angular_velocity * t), 
                           angular_velocity * radius * cos(angular_velocity * t), 
                           0,
                           0,
                           0,
                           0])
    # desired end effector acceleration (position and orientation)
    ddr_desired = np.array([-(angular_velocity)**2 * radius * cos(angular_velocity *t), 
                            -(angular_velocity)**2 * radius * sin(angular_velocity *t), 
                            0,
                            0,
                            0,
                            0])
    # desired end effector orientation

    #look_vector = center - r_desired
    #look_vector /= np.linalg.norm(look_vector)
    #y_axis = look_vector
    #x_axis = dr_desired[:3]
    #x_axis /= np.linalg.norm(x_axis)
    #z_axis = np.cross(x_axis, y_axis)
    #z_axis /= np.linalg.norm(z_axis)
    
    R_desired = Rx(pi/2)

    return r_desired, dr_desired, ddr_desired, R_desired



def plot_results(times: np.ndarray, positions: np.ndarray, velocities: np.ndarray, controls: np.ndarray):
    """Plot and save simulation results."""
    # Joint positions plot
    plt.figure(figsize=(10, 6))
    for i in range(positions.shape[1]):
        plt.plot(times, positions[:, i], label=f'Joint {i+1}')
    plt.xlabel('Time [s]')
    plt.ylabel('Joint Positions [rad]')
    plt.title('Joint Positions over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig('logs/plots/Joint positions.png')
    plt.close()
    
    # Joint velocities plot
    plt.figure(figsize=(10, 6))
    for i in range(velocities.shape[1]):
        plt.plot(times, velocities[:, i], label=f'Joint {i+1}')
    plt.xlabel('Time [s]')
    plt.ylabel('Joint Velocities [rad/s]')
    plt.title('Joint Velocities over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig('logs/plots/Joint velocities.png')
    plt.close()

    # Controls plot
    plt.figure(figsize=(10, 6))
    for i in range(controls.shape[1]):
        plt.plot(times, controls[:, i], label = f'Joint {i+1}')
    plt.xlabel('Time [s]')
    plt.ylabel('Joint Controls')
    plt.title('Joint Controls over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig('logs/plots/Controls.png')
    plt.close()
    
def plot_convergence(times: np.ndarray, pose_errors: np.ndarray, dpose_errors: np.ndarray, ddpose_errors: np.ndarray):
    # Joint positions plot
    plt.figure(figsize=(10, 6))
    for i in range(pose_errors.shape[1]):
        plt.plot(times, pose_errors[:, i], label=f'Pose error {i+1} (1-3 — position, 4-6 — angle)')
    plt.xlabel('Time [s]')
    plt.ylabel('Errors')
    plt.title('Pose error over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig('logs/plots/Pose errors.png')
    plt.close()
    
    # Joint velocities plot
    plt.figure(figsize=(10, 6))
    for i in range(dpose_errors.shape[1]):
        plt.plot(times, dpose_errors[:, i], label=f'Velocity error {i+1} (1-3 — tangential velocity, 4-6 — angular velocity)')
    plt.xlabel('Time [s]')
    plt.ylabel('Errors')
    plt.title('Velocity errors over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig('logs/plots/Endeffector velocity errors.png')
    plt.close()

    # Controls plot
    plt.figure(figsize=(10, 6))
    for i in range(ddpose_errors.shape[1]):
        plt.plot(times, ddpose_errors[:, i], label = f'Acceleration error {i+1} (1-3 — tangential acceleration, 4-6 — angular acceleration)')
    plt.xlabel('Time [s]')
    plt.ylabel('Errors')
    plt.title('Acceleration errors over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig('logs/plots/Acceleration errors.png')
    plt.close()

def error_calculation(q: np.ndarray, dq: np.ndarray, ddq, t: float, desired):
    pin.computeAllTerms(model, data, q, dq)
    ee_frame_id = model.getFrameId("end_effector")
    frame = pin.LOCAL

    
    #circular trajectory
    radius = 0.2 # circle radius
    center = np.array([0.3, 0.3, 0.6]) # center cooridnates
    angular_velocity = np.pi/2 # angular velocity
    r_desired, dr_desired, ddr_desired, R_desired = desired_pose_func(center, radius, angular_velocity, t)

    J_l = pin.getFrameJacobian(model, data, ee_frame_id, frame) # local frame jacobian
    dJ_l = pin.getFrameJacobianTimeVariation(model, data, ee_frame_id, frame) # local frame jacobian time derivative
    J_wa = pin.getFrameJacobian(model, data, ee_frame_id, pin.LOCAL_WORLD_ALIGNED) # world aligned frame jacobian
    dJ_wa = pin.getFrameJacobianTimeVariation(model, data, ee_frame_id, pin.LOCAL_WORLD_ALIGNED) # world aligned frame jacobian time derivative

    J = np.zeros((6,6))
    dJ = np.zeros((6,6))
    # Finding Jacobian to control positional velocity in world aligned space and angular velocity in local space 
    J[:3,:] = J_wa[:3,:]
    J[3:,:] = J_l[3:,:]
    dJ[:3,:] = dJ_wa[:3,:]
    dJ[3:,:] = dJ_l[3:,:]

    ee_pose = data.oMf[ee_frame_id]
    ee_position = ee_pose.translation
    ee_rotation = ee_pose.rotation

    pose_error = np.zeros(6)
    pose_error[:3] = r_desired - ee_position # position error calculation
    pose_error[3:] = pin.log3(ee_rotation.T@R_desired) # orientation error calculation

    dpose_error = dr_desired - J @ dq # velocity error calculation. dr_desired is the desired velocity (both tangential and angular)
    ddpose_error = ddr_desired - (J @ ddq + dJ @ dq)

    return pose_error, dpose_error, ddpose_error

def task_space_controller(q: np.ndarray, dq: np.ndarray, ddq, t: float, desired) -> np.ndarray:
    
    #circular trajectory
    radius = 0.2 # circle radius
    center = np.array([0.3, 0.3, 0.6]) # center cooridnates
    angular_velocity = np.pi/2 # angular velocity
    r_desired, dr_desired, ddr_desired, R_desired = desired_pose_func(center, radius, angular_velocity, t)

    pin.computeAllTerms(model, data, q, dq)
    pin.forwardKinematics(model, data, q, dq)

    ee_frame_id = model.getFrameId("end_effector")
    frame = pin.LOCAL
    
    pin.updateFramePlacement(model, data, ee_frame_id)

    J_l = pin.getFrameJacobian(model, data, ee_frame_id, frame) # local frame jacobian
    dJ_l = pin.getFrameJacobianTimeVariation(model, data, ee_frame_id, frame) # local frame jacobian time derivative
    J_wa = pin.getFrameJacobian(model, data, ee_frame_id, pin.LOCAL_WORLD_ALIGNED) # world aligned frame jacobian
    dJ_wa = pin.getFrameJacobianTimeVariation(model, data, ee_frame_id, pin.LOCAL_WORLD_ALIGNED) # world aligned frame jacobian time derivative

    J = np.zeros((6,6))
    dJ = np.zeros((6,6))
    # Finding Jacobian to control positional velocity in world aligned space and angular velocity in local space 
    J[:3,:] = J_wa[:3,:]
    J[3:,:] = J_l[3:,:]
    dJ[:3,:] = dJ_wa[:3,:]
    dJ[3:,:] = dJ_l[3:,:]

    ee_pose = data.oMf[ee_frame_id]
    ee_position = ee_pose.translation
    ee_rotation = ee_pose.rotation

    pose_error = np.zeros(6)
    pose_error[:3] = r_desired - ee_position # position error calculation
    pose_error[3:] = pin.log3(ee_rotation.T@R_desired) # orientation error calculation

    dpose_error = dr_desired - J @ dq # velocity error calculation. dr_desired is the desired velocity (both tangential and angular)
    ddpose_error = ddr_desired - (J @ ddq + dJ @ dq) 

    Kp = 100
    Kd = 20

    # Singular Jacobian handling
    if np.linalg.det(J) == 0:
        j_inv = np.linalg.pinv(J)
    else:
        j_inv = np.linalg.inv(J)

    aq = j_inv @ (ddr_desired + Kp * pose_error + Kd * dpose_error - dJ @ dq) # outer loop aq calculation

    tau = data.M @ aq + data.nle # joint space control calculation

    return tau

def main():
    Path("logs/videos").mkdir(parents=True, exist_ok=True)
    Path("logs/plots").mkdir(parents=True, exist_ok=True)
    
    print("\nRunning task space controller...")
    sim = Simulator(
        xml_path="robots/universal_robots_ur5e/scene.xml",
        enable_task_space=True,
        show_viewer=True,
        record_video=True,
        video_path="logs/videos/HomeAssignmentSimulationVideo.mp4",
        fps=30,
        width=1920,
        height=1080
    )
    sim.set_controller(task_space_controller)
    sim.reset()
    # Simulation parameters
    t = 0
    dt = sim.dt
    time_limit = 20.0

    # Data collection
    times = []
    positions = []
    velocities = []
    controls = []
    pose_errors = []
    dpose_errors = []
    ddpose_errors = []

    while t < time_limit:
        state = sim.get_state()
        times.append(t)
        positions.append(state['q'])
        velocities.append(state['dq'])

        tau = task_space_controller(q=state['q'], dq=state['dq'], ddq=state['ddq'], t=t, desired=state['desired'])
        controls.append(tau)
        sim.step(tau)

        if sim.record_video and len(sim.frames) < sim.fps * t:
            sim.frames.append(sim._capture_frame())
        t += dt

        pose_error, dpose_error, ddpose_error = error_calculation(q=state['q'], dq=state['dq'], ddq=state['ddq'], t=t, desired=state['desired'])

        pose_errors.append(pose_error)
        dpose_errors.append(dpose_error)
        ddpose_errors.append(ddpose_error)

    # Process and save results
    times = np.array(times)
    positions = np.array(positions)
    velocities = np.array(velocities)
    controls = np.array(controls)

    print(f"Simulation completed: {len(times)} steps")
    print(f"Final joint positions: {positions[-1]}")

    pose_errors = np.array(pose_errors)
    dpose_errors = np.array(dpose_errors)
    ddpose_errors = np.array(ddpose_errors)

    sim._save_video()
    plot_results(times, positions, velocities, controls)
    plot_convergence(times, pose_errors, dpose_errors, ddpose_errors)

if __name__ == "__main__":
    main()