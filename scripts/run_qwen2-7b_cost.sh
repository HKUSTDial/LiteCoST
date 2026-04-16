set -x
export CUDA_VISIBLE_DEVICES=6,7
export N_GPUS=2
export BASE_MODEL="./training/models/qwen2.5-7b-instruct"

# Use relative paths
# HOME now points to litecost (for data etc.)
export HOME="./"
# VERL_HOME points to the local verl repo to run GRPO training via python -m
export PYTHONPATH="$VERL_HOME:$PYTHONPATH"

# If you are using vllm<=0.6.3, you might need to set the following environment variable to avoid bugs:
export VLLM_ATTENTION_BACKEND=XFORMERS
export LOG_FILE_NAME="training/log/qwen2-7b_grpo_cost.log"

# Create log directory if not exists
mkdir -p "$(dirname "$LOG_FILE_NAME")"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$HOME/dataset/training_data/train.parquet \
    data.val_files=$HOME/dataset/training_data/test.parquet \
    data.train_batch_size=16 \
    data.max_prompt_length=4096 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='right' \
    reward_model.reward_manager=batch \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=2 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.project_name='verl_grpo_cost' \
    trainer.experiment_name='qwen2_7b_instruct' \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=-1 \
    trainer.total_epochs=5 | tee $LOG_FILE_NAME

    # trainer.resume_mode="resume_path" \
    # trainer.resume_from_path="./checkpoints/verl_grpo_cost_old/qwen2_7b_instruct/global_step_100" \
    # trainer.val_before_train=False \ 
    # reward_model.custom_reward_function.path=/path/to/your/custom_reward.py \

