export HF_ENDPOINT=https://hf-mirror.com

huggingface-cli download --resume-download lmms-lab/LLaVA-OneVision-1.5-4B-Instruct --local-dir llava_onevision --local-dir-use-symlinks False

huggingface-cli download --resume-download llava-hf/llava-onevision-qwen2-7b-ov-hf --local-dir llava_onevision_1 --local-dir-use-symlinks False

huggingface-cli download --repo-type dataset LiAuto-DriveAction/drive-action --local-dir drive-action --local-dir-use-symlinks False

python convert_driveaction_to_llava_onevision.py --dataset ./drive-action --out-dir ./drive_action_output --lang en --max-samples 5000