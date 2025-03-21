import argparse
import uuid
from collections import defaultdict
from typing import Dict
from skimage.io import imsave

import neptune as neptune
import numpy as np
import tensorflow as tf

from st_cvdm.configs.utils import (
    create_data_config,
    create_eval_config,
    create_model_config,
    create_neptune_config,
    load_config_from_yaml,
)
from st_cvdm.diffusion_models.joint_model import instantiate_st_cvdm
from st_cvdm.utils.inference_utils import (
    log_loss,
    log_metrics,
    obtain_output_montage_and_metrics,
    save_output_montage,
    ddpm_obtain_sr_img
)
from st_cvdm.utils.training_utils import prepare_dataset, prepare_model_input


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-path", help="Path to the configuration file", required=True
    )
    parser.add_argument("--neptune-token", help="API token for Neptune")

    args = parser.parse_args()

    print("Num CPUs Available: ", len(tf.config.list_physical_devices("CPU")))
    print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))

    config = load_config_from_yaml(args.config_path)
    model_config = create_model_config(config)
    data_config = create_data_config(config)
    eval_config = create_eval_config(config)
    neptune_config = create_neptune_config(config)
    print(model_config)
    task = config.get("task")
    assert task in [
        "loco",
        "biosr_sr",
        "imagenet_sr",
        "biosr_phase",
        "imagenet_phase",
        "hcoco_phase",
        "other",
    ], "Possible tasks are: biosr_sr, imagenet_sr, biosr_phase, imagenet_phase, hcoco_phase, other"

    print("Getting data...")
    batch_size = data_config.batch_size

    dataset, x_shape, y_shape = prepare_dataset(task, data_config, training=False)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    generation_timesteps = eval_config.generation_timesteps

    print("Creating model...")
    noise_model, joint_model, schedule_model, mu_model = instantiate_st_cvdm(
        lr=0.0,
        generation_timesteps=generation_timesteps,
        cond_shape=x_shape,
        out_shape=y_shape,
        model_config=model_config,
    )
    if model_config.load_weights is not None:
        joint_model.load_weights(model_config.load_weights)
    if model_config.load_mu_weights is not None and mu_model is not None:
        mu_model.load_weights(model_config.load_mu_weights)
    run = None
    if args.neptune_token is not None and neptune_config is not None:
        run = neptune.init_run(
            api_token=args.neptune_token,
            name=neptune_config.name,
            project=neptune_config.project,
        )
        run["config.yaml"].upload(args.config_path)

    output_path = eval_config.output_path
    diff_inp = model_config.diff_inp

    cumulative_loss = np.zeros(5)
    run_id = str(uuid.uuid4())
    step = 0
    cumulative_metrics: Dict[str, float] = defaultdict(float)
    total_samples = 0
    for batch in dataset:
        batch_x, batch_y = batch

        cmap = (
            "gray" if task in ["loco", "biosr_phase", "imagenet_phase", "hcoco_phase"] else None
        )
        model_input = prepare_model_input(batch_x, batch_y, diff_inp=diff_inp)
        cumulative_loss += joint_model.evaluate(
            model_input, np.zeros_like(batch_y), verbose=0
        )

        print('Saving at: ' + output_path)
        n_iters=1
        print(batch_x.shape)
        for sample in range(n_iters):
            pred_diff, gamma_vec, _ = ddpm_obtain_sr_img(
                batch_x,
                generation_timesteps,
                noise_model,
                schedule_model,
                mu_model,
                batch_y.shape,
            )
            pred_diff = np.clip(pred_diff, -1, 1)
            imsave(output_path+f'/z-{step}-{sample}.tif', np.squeeze(pred_diff))
            imsave(output_path+f'/x-{step}-{sample}.tif', np.squeeze(batch_x))
            imsave(output_path+f'/y-{step}-{sample}.tif', np.squeeze(batch_y))        
            #imsave(output_path+f'/z-{step}-{sample}.tif', pred_diff[:,:,:,0])
            #imsave(output_path+f'/x-{step}-{sample}.tif', batch_x[:,:,:,0])
            #imsave(output_path+f'/y-{step}-{sample}.tif', batch_y[:,:,:,0])  
        step += 1
        
        """
        output_montage, metrics = obtain_output_montage_and_metrics(
            batch_x,
            batch_y.numpy(),
            noise_model,
            schedule_model,
            mu_model,
            generation_timesteps,
            diff_inp,
            task,
        )

        for metric_name, metric_value in metrics.items():
            cumulative_metrics[metric_name] += metric_value * batch_size
        total_samples += batch_size
        step += 1

    average_metrics = {
        metric_name: total / total_samples
        for metric_name, total in cumulative_metrics.items()
    }

    log_metrics(run, average_metrics, prefix="val")
    print('Saving to ' + output_path)
    save_output_montage(
        run=run,
        output_montage=output_montage,
        step=step,
        output_path=output_path,
        run_id=run_id,
        prefix="val",
        cmap=cmap,
    )
    print("Loss: ", cumulative_loss)
    log_loss(run=run, avg_loss=cumulative_loss / (step + 1), prefix="val")
    """

    if run is not None:
        run.stop()

if __name__ == "__main__":
    main()
