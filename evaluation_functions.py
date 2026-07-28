import numpy as np
from jax import numpy as jnp

from scipy.stats import gaussian_kde
from matplotlib import pyplot as plt



def tracking_summary(state_direct, state_mpc, state_true):

    error_direct = state_direct[..., :4] - state_true[..., :4]
    error_mpc = state_mpc[..., :4] - state_true[..., :4]

    rmse_direct = jnp.sqrt(jnp.mean(error_direct**2, axis=1))
    rmse_mpc = jnp.sqrt(jnp.mean(error_mpc**2, axis=1))

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.ravel()

    for i, ax in enumerate(axes):

        direct = np.asarray(rmse_direct[:, i])
        mpc = np.asarray(rmse_mpc[:, i])

        x = np.linspace(
        min(direct.min(), mpc.min()),
        max(direct.max(), mpc.max()),
        300,
        )

        kde_direct = gaussian_kde(direct)
        kde_mpc = gaussian_kde(mpc)

        ax.plot(x,kde_direct(x),lw=2,color="tab:blue",label="Direct")
        ax.fill_between(x,kde_direct(x),alpha=0.25,color="tab:blue")
        ax.plot(x,kde_mpc(x),lw=2,color="tab:orange",label="MPC",)
        ax.fill_between(x,kde_mpc(x),alpha=0.25,color="tab:orange")

        ax.set_title(rf"$q_{i+1}$")
        ax.set_xlabel("RMSE [rad]")
        ax.set_ylabel("Density")
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




def input_summary(input_mpc , input_true):

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