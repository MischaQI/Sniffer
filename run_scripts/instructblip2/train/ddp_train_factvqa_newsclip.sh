GPU=$1
PORT=$2
python -m torch.distributed.run --nproc_per_node=${GPU} --master_port=${PORT} train.py --cfg-path lavis/projects/instructblip2/train/factvqa_newsclip_ft.yaml ${@:3}
