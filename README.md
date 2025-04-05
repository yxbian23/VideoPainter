
# VideoPainter

This repository contains the implementation of the paper "VideoPainter: Any-length Video Inpainting and Editing with Plug-and-Play Context Control"

Keywords: Video Inpainting, Video Editing, Video Generation


**📖 Table of Contents**


- [VideoPainter](#videopainter)
  - [🛠️ Method Overview](#️-method-overview)
  - [🚀 Getting Started](#-getting-started)
    - [Environment Requirement 🌍](#environment-requirement-)
  - [🏃🏼 Running Scripts](#-running-scripts)
    - [Training 🤯](#training-)
    - [Inference 📜](#inference-)
    - [Evaluation 📏](#evaluation-)
  - [🤝🏼 Cite Us](#-cite-us)
  - [💖 Acknowledgement](#-acknowledgement)

## 🛠️ Method Overview

We propose a novel dual-stream paradigm VideoPainter that incorporates an efficient context encoder (comprising only 6\% of the backbone parameters) to process masked videos and inject backbone-aware background contextual cues to any pre-trained video DiT, producing semantically consistent content in a plug-and-play manner. This architectural separation significantly reduces the model's learning complexity while enabling nuanced integration of crucial background context. We also introduce a novel target region ID resampling technique that enables any-length video inpainting, greatly enhancing our practical applicability. Additionally, we establish a scalable dataset pipeline leveraging current vision understanding models, contributing VPData and VPBench to facilitate segmentation-based inpainting training and assessment, the largest video inpainting dataset and benchmark to date with over 390K diverse clips. Using inpainting as a pipeline basis, we also explore downstream applications including video editing and video editing pair data generation, demonstrating competitive performance and significant practical potential. 
![](assets/teaser.jpg)



## 🚀 Getting Started

<details>
<summary><b>Environment Requirement 🌍</b></summary>


Clone the repo:

```
git clone https://github.com/TencentARC/VideoPainter.git
```

We recommend you first use `conda` to create virtual environment, and install needed libraries. For example:


```
conda create -n videopainter python=3.10 -y
conda activate videopainter
pip install -r requirements.txt
```

Then, you can install diffusers (implemented in this repo) with:

```
cd ./diffusers
pip install -e .
```

After that, you can install required ffmpeg thourgh:

```
conda install -c conda-forge ffmpeg -y
```

Optional, you can install sam2 for gradio demo thourgh:

```
cd ./app
pip install -e .
```
</details>



## 🏃🏼 Running Scripts

<details>
<summary><b>Training 🤯</b></summary>

You can train the VideoPainter using the script:

```
# cd train
# bash VideoPainter.sh

export MODEL_PATH="../ckpt/CogVideoX-5b-I2V"
export CACHE_PATH="~/.cache"
export DATASET_PATH="../data/videovo/raw_video"
export PROJECT_NAME="pexels_videovo-inpainting"
export RUNS_NAME="VideoPainter"
export OUTPUT_PATH="./${PROJECT_NAME}/${RUNS_NAME}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

accelerate launch --config_file accelerate_config_machine_single_ds.yaml  --machine_rank 0 \
  train_cogvideox_inpainting_i2v_video.py \
  --pretrained_model_name_or_path $MODEL_PATH \
  --cache_dir $CACHE_PATH \
  --meta_file_path ../data/pexels_videovo_train_dataset.csv \
  --val_meta_file_path ../data/pexels_videovo_val_dataset.csv \
  --instance_data_root $DATASET_PATH \
  --dataloader_num_workers 1 \
  --num_validation_videos 1 \
  --validation_epochs 1 \
  --seed 42 \
  --mixed_precision bf16 \
  --output_dir $OUTPUT_PATH \
  --height 480 \
  --width 720 \
  --fps 8 \
  --max_num_frames 49 \
  --video_reshape_mode "resize" \
  --skip_frames_start 0 \
  --skip_frames_end 0 \
  --max_text_seq_length 226 \
  --branch_layer_num 2 \
  --train_batch_size 1 \
  --num_train_epochs 10 \
  --checkpointing_steps 1024 \
  --validating_steps 256 \
  --gradient_accumulation_steps 1 \
  --learning_rate 1e-5 \
  --lr_scheduler cosine_with_restarts \
  --lr_warmup_steps 1000 \
  --lr_num_cycles 1 \
  --enable_slicing \
  --enable_tiling \
  --noised_image_dropout 0.05 \
  --gradient_checkpointing \
  --optimizer AdamW \
  --adam_beta1 0.9 \
  --adam_beta2 0.95 \
  --max_grad_norm 1.0 \
  --allow_tf32 \
  --report_to wandb \
  --tracker_name $PROJECT_NAME \
  --runs_name $RUNS_NAME \
  --inpainting_loss_weight 1.0 \
  --mix_train_ratio 0 \
  --first_frame_gt \
  --mask_add \
  --mask_transform_prob 0.3 \
  --p_brush 0.4 \
  --p_rect 0.1 \
  --p_ellipse 0.1 \
  --p_circle 0.1 \
  --p_random_brush 0.3

# cd train
# bash VideoPainterID.sh
export MODEL_PATH="../ckpt/CogVideoX-5b-I2V"
export BRANCH_MODEL_PATH="../ckpt/VideoPainter/checkpoints/branch"
export CACHE_PATH="~/.cache"
export DATASET_PATH="../data/videovo/raw_video"
export PROJECT_NAME="pexels_videovo-inpainting"
export RUNS_NAME="VideoPainterID"
export OUTPUT_PATH="./${PROJECT_NAME}/${RUNS_NAME}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

accelerate launch --config_file accelerate_config_machine_single_ds_wo_cpu.yaml --machine_rank 0 \
  train_cogvideox_inpainting_i2v_video_resample.py \
  --pretrained_model_name_or_path $MODEL_PATH \
  --cogvideox_branch_name_or_path $BRANCH_MODEL_PATH \
  --cache_dir $CACHE_PATH \
  --meta_file_path ../data/pexels_videovo_train_dataset.csv \
  --val_meta_file_path ../data/pexels_videovo_val_dataset.csv \
  --instance_data_root $DATASET_PATH \
  --dataloader_num_workers 1 \
  --num_validation_videos 1 \
  --validation_epochs 1 \
  --seed 42 \
  --rank 256 \
  --lora_alpha 128 \
  --mixed_precision bf16 \
  --output_dir $OUTPUT_PATH \
  --height 480 \
  --width 720 \
  --fps 8 \
  --max_num_frames 49 \
  --video_reshape_mode "resize" \
  --skip_frames_start 0 \
  --skip_frames_end 0 \
  --max_text_seq_length 226 \
  --branch_layer_num 2 \
  --train_batch_size 1 \
  --num_train_epochs 10 \
  --checkpointing_steps 256 \
  --validating_steps 128 \
  --gradient_accumulation_steps 1 \
  --learning_rate 5e-5 \
  --lr_scheduler cosine_with_restarts \
  --lr_warmup_steps 200 \
  --lr_num_cycles 1 \
  --enable_slicing \
  --enable_tiling \
  --noised_image_dropout 0.05 \
  --gradient_checkpointing \
  --optimizer AdamW \
  --adam_beta1 0.9 \
  --adam_beta2 0.95 \
  --max_grad_norm 1.0 \
  --allow_tf32 \
  --report_to wandb \
  --tracker_name $PROJECT_NAME \
  --runs_name $RUNS_NAME \
  --inpainting_loss_weight 1.0 \
  --mix_train_ratio 0 \
  --first_frame_gt \
  --mask_add \
  --mask_transform_prob 0.3 \
  --p_brush 0.4 \
  --p_rect 0.1 \
  --p_ellipse 0.1 \
  --p_circle 0.1 \
  --p_random_brush 0.3 \
  --id_pool_resample_learnable
```
</details>


<details>
<summary><b>Inference 📜</b></summary>

You can inference for the video inpainting or editing with the script:

```
cd infer
# video inpainting
bash inpaint.sh
# video inpainting with ID resampling
bash inpaint_id_resample.sh
# video editing
bash edit.sh
```

Our VideoPainter can also function as a video editing pair data generator, you can inference with the script:
```
bash edit_bench.sh
```

Since VideoPainter is trained on public Internet videos, it primarily performs well on general scenarios. For high-quality industrial applications (e.g., product exhibitions, virtual try-on), we recommend training the model on your domain-specific data. We welcome and appreciate any contributions of trained models from the community!
</details>

<details>
<summary><b>Gradio Demo 🖌️</b></summary>

You can also inference through gradio demo:

```
# cd app
CUDA_VISIBLE_DEVICES=0 python app.py \
    --model_path ../ckpt/CogVideoX-5b-I2V \
    --inpainting_branch ../ckpt/VideoPainter/checkpoints/branch \
    --id_adapter ../ckpt/VideoPainterID/checkpoints \
    --img_inpainting_model ../ckpt/flux_inp
```
</details>


<details>
<summary><b>Evaluation 📏</b></summary>

You can evaluate using the script:

```
cd evaluate
# video inpainting
bash eval_inpainting.sh
# video inpainting with ID resampling
bash eval_inpainting_id_resample.sh
# video editing
bash eval_edit.sh
# video editing with ID resampling
bash eval_editing_id_resample.sh
```
</details>



## 💖 Acknowledgement
<span id="acknowledgement"></span>

Our code is modified based on [diffusers](https://github.com/huggingface/diffusers) and [CogVideoX](https://github.com/THUDM/CogVideo), thanks to all the contributors!
