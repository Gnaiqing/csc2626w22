import sys
import pickle
import gym
import gym_thing
import numpy as np
import pathlib
import matplotlib.pyplot as plt
import time
from dmp import DMP
from policy import BallCatchDMPPolicy

PATH_TO_HOME = str(pathlib.Path(__file__).parent.resolve())
PATH_TO_NLOPTJSON = str((
    pathlib.Path(__file__).parent /
    "ballcatch_env" /
    "gym_thing" /
    "nlopt_optimization" /
    "config" /
    "nlopt_config_stationarybase.json"
).resolve())

TRAINED_DMP_PATH = "results/trained_dmp.pkl"

def make_env(online=True, dataset=None, n_substeps = 1, gravity_factor_std=0., n_substeps_std=0.,
             mass_std = 0., act_gain_std = 0., joint_noise_std=0., vicon_noise_std=0., control_noise_std=0., random_training_index=False):
    # Create thing env
    initial_arm_qpos = np.array([1, -.3, -1, 0, 1.6, 0])
    initial_ball_qvel = np.array([-1, 1.5, 4.5])
    initial_ball_qpos = np.array([1.22, -1.6, 1.2])
    zero_base_qpos = np.array([0.0, 0.0, 0.0])

    training_data = None
    if not online:
        training_data = np.load(dataset)
        print("OFFLINE")

    env = gym.make('ballcatch-v0',
                    model_path="robot_with_cup.xml",
                    nlopt_config_path=PATH_TO_NLOPTJSON,
                    initial_arm_qpos=initial_arm_qpos,
                    initial_ball_qvel=initial_ball_qvel,
                    initial_ball_qpos=initial_ball_qpos,
                    initial_base_qpos=zero_base_qpos,
                    pos_rew_weight=.1,
                    rot_rew_weight=0.01,
                    n_substeps=n_substeps,
                    online=online,
                    training_data=training_data,
                    gravity_factor_std = gravity_factor_std,
                    n_substeps_std = n_substeps_std,
                    mass_std = mass_std,
                    act_gain_std = act_gain_std,
                    joint_noise_std = joint_noise_std,
                    vicon_noise_std = vicon_noise_std,
                    control_noise_std = control_noise_std,
                    random_training_index = random_training_index)
    env.reset()
    return env

def test_policy(eval_env, policy, eval_episodes=5, render_freq=1, seed=1):
    # Set seeds
    eval_env.seed(seed)
    np.random.seed(seed)

    avg_reward = 0.
    successes = 0


    for eps in range(eval_episodes):
        print(f"\rEvaluating Episode {eps+1}/{eval_episodes}", end='')
        state, done = eval_env.reset(), False
        policy.set_goal(state=state, goal=eval_env.env.goal)
        while not done:
            action = policy.select_action(np.array(state))
            state, reward, done, info = eval_env.step(action)
            if render_freq != 0 and eps % render_freq == 0:
                eval_env.render()
                # time.sleep(0.01)
            avg_reward += reward
        if "TimeLimit.truncated" in info:
            actual_done = not info["TimeLimit.truncated"]
        else:
            actual_done = done

        if actual_done:
            successes += 1


    avg_reward /= eval_episodes
    success_pct = float(successes)/eval_episodes

    print("")
    print("---------------------------------------")
    print("Evaluation over {} episodes: {:.3f}, success%: {:.3f}".format(eval_episodes, avg_reward, success_pct))
    print("---------------------------------------")
    print("")
    return avg_reward, success_pct

def load_dataset(dataset_path="data/demos.pkl"):
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)
    return [traj[:, :6] for traj in dataset["trajectories"]] # 1st 6 elements are joint angle

def q2_recons():
    trajectories = load_dataset()
    # TODO: Train a DMP on trajectories[0]
    X = trajectories[0][np.newaxis,:]   # (1, n_timesteps, n_dof)
    dt = 0.04
    x0 = X[0, 0, :]
    g = X[0, -1, :]
    T = dt * np.arange(X.shape[1]).reshape(1,-1)  # (1, n_timesteps)
    tau = T[0,-1]
    dmp = DMP()
    dmp.learn(X, T)


    rollout = dmp.rollout(dt, tau, x0, g)
    demo = trajectories[0]
    demo_time = np.arange(X.shape[1]) * dt
    rollout_time = np.arange(rollout.shape[0]) * dt

    for k in range(6):
        plt.figure()
        plt.plot(demo_time, demo[:, k], label='GT')
        plt.plot(rollout_time, rollout[:, k], label='DMP')
        plt.legend()
        plt.savefig(f'results/recons_{k}.png')


def q2_tuning():
    trajectories = load_dataset()
    
    # TODO: select the best settings for fitting the demos
    mse = 1e6
    best_nbasis = None
    best_k = None
    for nbasis in [15, 30, 45, 60, 75, 90]:
        for k in [10, 20, 40, 80]:
            dmp = DMP(nbasis=nbasis, K_vec=k*np.ones(6,))
            initial_dt = 0.04
            X, T = dmp._interpolate(trajectories, initial_dt)
            dmp.learn(X, T)
            # evaluate the learned dmp for fitting the demos
            se = 0
            for i in range(len(trajectories)):
                demo = X[i,:,:]  # (n_steps, n_dofs)
                demo_time = T[i,:]  # (n_steps)
                dt = demo_time[1] - demo_time[0]
                tau = demo_time[-1]
                x0 = demo[0,:]  # (n_dofs)
                g = demo[-1,:]   # (n_dofs)
                rollout = dmp.rollout(dt, tau, x0, g)  # (n_steps * n_dofs)
                if rollout.shape[0] != demo.shape[0]:
                    # due to numerical issues the rollout might be longer than demo
                    rollout = rollout[:demo.shape[0],:]

                se += ((rollout - demo) ** 2).sum()

            if se < mse:
                mse = se
                best_nbasis = nbasis
                best_k = k
                dmp.save(TRAINED_DMP_PATH)

    print(f" MSE: {mse}\n Best nbasis: {best_nbasis}\n Best k: {best_k}")


def main():
    env = make_env(n_substeps=1)
    dmp = DMP.load(TRAINED_DMP_PATH)

    policy = BallCatchDMPPolicy(dmp, dt=env.dt)
    test_policy(env, policy, eval_episodes=20, render_freq=1)


# q2_recons()
if __name__ == '__main__':
    if len(sys.argv) == 1:
        main()
    elif sys.argv[1] == 'recons':
        q2_recons()
    elif sys.argv[1] == 'tuning':
        q2_tuning()
