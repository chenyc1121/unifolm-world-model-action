res_dir="unitree_z1_dual_arm_cleanup_pencils/case4"
dataset="unitree_z1_dual_arm_cleanup_pencils"

{
    time CUDA_VISIBLE_DEVICES=0 python3 scripts/evaluation/world_model_interaction.py \
        --seed 123 \
        --ckpt_path ckpts/unifolm_wma_dual.ckpt \
        --config configs/inference/world_model_interaction.yaml \
        --savedir "${res_dir}/output" \
        --bs 1 --height 320 --width 512 \
        --unconditional_guidance_scale 1.0 \
        --ddim_steps 50 \
        --ddim_eta 1.0 \
        --prompt_dir "unitree_z1_dual_arm_cleanup_pencils/case4/world_model_interaction_prompts" \
        --dataset ${dataset} \
        --video_length 16 \
        --frame_stride 4 \
        --n_action_steps 16 \
        --exe_steps 16 \
        --n_iter 8 \
        --timestep_spacing 'uniform_trailing' \
        --guidance_rescale 0.7 \
        --perframe_ae
} 2>&1 | tee "${res_dir}/output.log"
