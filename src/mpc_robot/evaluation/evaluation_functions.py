import numpy as np
from jax import numpy as jnp

from scipy.stats import gaussian_kde
from matplotlib import pyplot as plt



def tracking_summary_MPC(state_direct, state_mpc, state_true):

    error_direct = state_direct[..., :4] - state_true[..., :4]
    error_mpc = state_mpc[..., :4] - state_true[..., :4]

    rmse_direct = jnp.sqrt(jnp.mean(error_direct**2, axis=1))
    rmse_mpc = jnp.sqrt(jnp.mean(error_mpc**2, axis=1))

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.ravel()

    for i, ax in enumerate(axes):

        direct = np.asarray(rmse_direct[:, i])
        mpc = np.asarray(rmse_mpc[:, i])

        bins = np.linspace(min(direct.min(), mpc.min()),max(direct.max(), mpc.max()),20)
        ax.hist(direct,bins=bins,alpha=0.5,color="tab:blue",label="Direct")
        ax.hist(mpc,bins=bins,alpha=0.5,color="tab:orange",label="MPC")

        ax.set_title(rf"$q_{i+1}$")
        ax.set_xlabel("RMSE [rad]")
        ax.set_ylabel("Count")
        ax.grid(alpha=0.3)

    axes[0].legend()

    plt.tight_layout()
    plt.show()

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.ravel()

    for i, ax in enumerate(axes):

        direct = jnp.asarray(rmse_direct[:, i])
        mpc = jnp.asarray(rmse_mpc[:, i])

        ax.boxplot([direct, mpc],positions=[1, 2],widths=0.5,showmeans=True)

        ax.scatter(jnp.full_like(direct, 1, dtype=float),direct,alpha=0.7,label="Direct")
        ax.scatter(jnp.full_like(mpc, 2, dtype=float),mpc,alpha=0.7,label="MPC")

        ax.set_xticks([1, 2])
        ax.set_xticklabels(["Direct", "MPC"])
        ax.set_title(rf"$q_{i+1}$")
        ax.set_ylabel("RMSE [rad]")
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.show()

    n_samples = rmse_direct.shape[0]
    if n_samples > 10:
        rng = np.random.default_rng(0)      # fixed seed
        sample_idx = np.sort(rng.choice(n_samples, 10, replace=False))
    else:
        sample_idx = np.arange(n_samples)

    rmse_direct_plot = rmse_direct[sample_idx]
    rmse_mpc_plot = rmse_mpc[sample_idx]

    x = np.arange(len(sample_idx))
    width = 0.4

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.ravel()

    for i, ax in enumerate(axes):

        ax.bar(x - width/2,rmse_direct_plot[:, i],width=width,label="Direct")
        ax.bar(x + width/2,rmse_mpc_plot[:, i],width=width,label="MPC")

        ax.set_xticks(x)
        ax.set_xticklabels(sample_idx)

        ax.set_title(rf"$q_{i+1}$")
        ax.set_xlabel("Sample index")
        ax.set_ylabel("RMSE [rad]")
        ax.grid(axis="y", alpha=0.3)

    axes[0].legend()

    plt.tight_layout()
    plt.show()

    def print_stats(title, rmse):

        print(f"\n{title}")
        print("-" * 75)
        print(
            f"{'Joint':<6}"
            f"{'Mean':>12}"
            f"{'25%':>12}"
            f"{'Median':>12}"
            f"{'75%':>12}"
            f"{'Min':>12}"
            f"{'Max':>12}"
        )

        for i in range(4):

            values = rmse[:, i]

            print(
            f"q{i+1:<5}"
            f"{jnp.mean(values):12.4e}"
            f"{jnp.percentile(values,25):12.4e}"
            f"{jnp.median(values):12.4e}"
            f"{jnp.percentile(values,75):12.4e}"
            f"{jnp.min(values):12.4e}"
            f"{jnp.max(values):12.4e}"
            )

    print_stats("DIRECT MODEL", rmse_direct)
    print_stats("MPC", rmse_mpc)

    return None




def input_summary_MPC(input_mpc , input_true):

    input_mpc = np.asarray(input_mpc)
    input_true = np.asarray(input_true)

    # Match lengths if MPC returns T-1 inputs
    n_steps = min(input_mpc.shape[1], input_true.shape[1])
    input_mpc = input_mpc[:, :n_steps]
    input_true = input_true[:, :n_steps]

    error = input_mpc - input_true
    delta_mpc = np.diff(input_mpc, axis=1)
    delta_true = np.diff(input_true, axis=1)

    # Shape of each metric: (N, 4)
    rmse = np.sqrt(np.mean(error**2, axis=1))
    effort_mpc = np.mean(input_mpc**2, axis=1)
    effort_true = np.mean(input_true**2, axis=1)
    smooth_mpc = np.sqrt(np.mean(delta_mpc**2, axis=1))
    smooth_true = np.sqrt(np.mean(delta_true**2, axis=1))

    metrics = [
        ("Input tracking RMSE", rmse, None, "RMSE [Nm]"),
        ("Control effort", effort_mpc, effort_true, r"Mean $u^2$ [Nm$^2$]"),
        ("Input smoothness", smooth_mpc, smooth_true, r"RMS $\Delta u$ [Nm]"),
    ]

    fig, axes = plt.subplots(3, 4, figsize=(16, 10))

    for row, (title, mpc_values, true_values, ylabel) in enumerate(metrics):

        for joint in range(4):
            ax = axes[row, joint]

            if true_values is None:
                ax.boxplot(mpc_values[:, joint],tick_labels=["MPC"],showmeans=True)

                ax.scatter(np.ones(mpc_values.shape[0]),mpc_values[:, joint],alpha=0.5)

            else:
                ax.boxplot([true_values[:, joint],mpc_values[:, joint],],
                    tick_labels=["True", "MPC"],
                    showmeans=True,
                )

                ax.scatter(np.ones(true_values.shape[0]),true_values[:, joint],alpha=0.5,)

                ax.scatter(np.full(mpc_values.shape[0], 2),mpc_values[:, joint],
                    alpha=0.5,
                )

            if row == 0:
                ax.set_title(rf"$u_{joint+1}$")

            if joint == 0:
                ax.set_ylabel(f"{title}\n{ylabel}")
            else:
                ax.set_ylabel(ylabel)

            ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.show()

    def print_stats(title, values):

        print(f"\n{title}")
        print("-" * 76)
        print(
            f"{'Input':<7}"
            f"{'Mean':>12}"
            f"{'25%':>12}"
            f"{'Median':>12}"
            f"{'75%':>12}"
            f"{'Min':>12}"
            f"{'Max':>12}"
        )

        stats = np.column_stack([
            np.mean(values, axis=0),
            np.percentile(values, 25, axis=0),
            np.median(values, axis=0),
            np.percentile(values, 75, axis=0),
            np.min(values, axis=0),
            np.max(values, axis=0),
        ])

        for i, row in enumerate(stats):
            print(
                f"u{i+1:<6}"
                + "".join(f"{value:12.4e}" for value in row)
            )

    print_stats("INPUT TRACKING RMSE — MPC vs TRUE", rmse)
    print_stats("CONTROL EFFORT — TRUE", effort_true)
    print_stats("CONTROL EFFORT — MPC", effort_mpc)
    print_stats("INPUT SMOOTHNESS — TRUE", smooth_true)
    print_stats("INPUT SMOOTHNESS — MPC", smooth_mpc)






def evaluate_ensemble_training(
    state_pred,
    q_true,
    bins="auto",
):
    """Evaluate ensemble predictions and select a representative model.

    Parameters
    ----------
    state_pred : array, shape (N, M, T, state_dim)  Predictions for N samples and M ensemble models.

    q_true : array, shape (N, T, Q)  True joint positions.
    bins : str or int
        Histogram bin strategy, e.g. "auto", "fd", or an integer.

    Returns
    -------
    rmse : jax.Array, shape (N, M, Q)
        Trajectory RMSE for every sample, model, and joint.

    selected_idx : int
        Index of the representative model.

    mean_ranks : ndarray, shape (M,)
        Mean rank of every model. Rank 1 is best.
    """

    state_pred = jnp.asarray(state_pred)
    q_true = jnp.asarray(q_true)

    n_joints = q_true.shape[-1]

    if state_pred.shape[0] != q_true.shape[0]:
        raise ValueError("Prediction and truth sample counts do not match.")

    if state_pred.shape[2] != q_true.shape[1]:
        raise ValueError("Prediction and truth time dimensions do not match.")

    q_pred = state_pred[..., :n_joints]

    # (samples, models, joints)
    rmse = jnp.sqrt(
        jnp.mean((q_pred - q_true[:, None, :, :]) ** 2,axis=2))

    rmse_np = np.asarray(rmse)
    n_samples, n_models, n_joints = rmse_np.shape

    model_names = [f"Model {i + 1}" for i in range(n_models)]

    # Rank models separately for every sample and joint.
    # argsort(argsort(...)) gives ranks 0 ... M-1.
    ranks = np.argsort(np.argsort(rmse_np, axis=1),axis=1) + 1

    mean_ranks = ranks.mean(axis=(0, 2))

    average_rank = np.mean(mean_ranks)

    selected_idx = int(
    np.argmin(np.abs(mean_ranks - average_rank)))

    # Dynamic plot layout
    n_cols = min(2, n_joints)
    n_rows = int(np.ceil(n_joints / n_cols))

    fig, axes = plt.subplots(n_rows,n_cols,figsize=(6.4 * n_cols, 3.8 * n_rows),squeeze=False,)

    axes = axes.ravel()

    for joint_idx in range(n_joints):

        ax = axes[joint_idx]
        joint_values = rmse_np[:, :, joint_idx]

        # Common bins across all models for fair comparison
        bin_edges = np.histogram_bin_edges(joint_values.ravel(),bins=bins)

        bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2

        for model_idx in range(n_models):

            counts, _ = np.histogram(joint_values[:, model_idx],bins=bin_edges)

            selected = model_idx == selected_idx

            ax.plot(bin_centres,counts,marker="o",
                markersize=4 if selected else 3,
                linewidth=2.5 if selected else 1.2,
                alpha=1.0 if selected else 0.6,
                label=(f"{model_names[model_idx]} — representative"
                    if selected
                    else model_names[model_idx]
                ),
            )

        ax.set_title(rf"$q_{{{joint_idx + 1}}}$")
        ax.set_xlabel("Trajectory RMSE")
        ax.set_ylabel("Number of samples")
        ax.grid(alpha=0.25)

    # Hide unused subplot axes
    for ax in axes[n_joints:]:
        ax.set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()

    fig.suptitle(
        "Test-sample RMSE distributions across model initialisations",fontsize=14,y=0.99)

    fig.legend(handles,labels,loc="upper center",ncol=min(n_models, 5),bbox_to_anchor=(0.5, 0.95))

    fig.tight_layout(rect=(0, 0, 1, 0.88))
    plt.show()

    print("Mean model ranks:")

    for model_idx, rank in enumerate(mean_ranks):
        marker = "  <-- representative" if model_idx == selected_idx else ""

        print(f"{model_names[model_idx]}: "f"{rank:.3f}{marker}")
    print(f"Selected model: {model_names[selected_idx]}")

    return rmse, selected_idx






def loss_history_mpc(loss_history):

    loss_history = np.asarray(loss_history)

    # ==========================================================
    # Global view: first vs last optimisation iteration
    # ==========================================================
    mpc_steps = np.arange(loss_history.shape[0])

    plt.figure(figsize=(10, 4))
    plt.plot(mpc_steps,loss_history[:, 0],label="Initial loss",linewidth=1.5)
    plt.plot(mpc_steps,loss_history[:, -1],label="Final loss",linewidth=1.5)
    plt.yscale("log")
    plt.xlabel("MPC step")
    plt.ylabel("Loss")
    plt.title("Initial and final MPC loss")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ==========================================================
    # Convergence for selected MPC steps
    # ==========================================================
    selected_steps = np.linspace(0,loss_history.shape[0] - 1,5,dtype=int)
    opt_iterations = np.arange(1,loss_history.shape[1] + 1)
    fig, axes = plt.subplots(1,len(selected_steps),figsize=(18, 4),sharey=False)
    for ax, step in zip(axes, selected_steps):

        ax.plot(opt_iterations,loss_history[step],marker="o")
        ax.set_title(f"time step {step}")
        ax.set_xlabel("Optimisation step")
        ax.set_ylabel("Loss")
        ax.ticklabel_format(
            axis="y",
            style="sci",
            scilimits=(0, 0),
        )
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
