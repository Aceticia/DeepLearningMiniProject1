#!/usr/bin/env python3

if __name__ == "__main__":
    import optuna
    from optuna.visualization import plot_contour
    from optuna.visualization import plot_slice
    from optuna.visualization import plot_edf

    # Load the study database
    study = optuna.load_study(
        study_name="DL2022_multi", storage="sqlite:///storage/multi.db"
    )

    # Plot the relationship wrt kernel and skip kernel
    for layer in range(4):
        fig = plot_contour(
            study,
            target=lambda t: t.values[1],
            target_name="validation acc",
            params=[f"layer{layer}_kernel_size", f"layer{layer}_skip_kernel_size"],
        )
        fig.write_image(f"./storage/layer{layer}_size_figure.png")

    # Plot the relationsihp between dropblock size and drop prob
    for layer in range(4):
        fig = plot_contour(
            study,
            target=lambda t: t.values[1],
            target_name="validation acc",
            params=[f"layer{layer}_drop_size", f"layer{layer}_drop_prob"],
        )
        fig.write_image(f"./storage/layer{layer}_dropblock_figure.png")

    # Plot the edf of validation acc
    fig = plot_edf(study, target=lambda t: t.values[1], target_name="Validation acc")
    fig.write_image("./storage/val_acc_edf.png")

    fig = plot_edf(study, target=lambda t: t.values[0], target_name="Size penalty")
    fig.write_image("./storage/size_penalty.png")

    # Plot the contour for sizes of each block and hidden units
    for layer in range(3):
        fig = plot_contour(
            study,
            target=lambda t: t.values[1],
            target_name="validation acc",
            params=[f"layer{layer_}_nblocks" for layer_ in [layer, layer + 1]],
        )
        fig.write_image(f"./storage/layer{layer}_and_layer{layer+1}.png")
