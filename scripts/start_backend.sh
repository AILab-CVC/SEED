
python3 gradio_demo/seed_llama_flask.py \
    --image_transform configs/transform/clip_transform.yaml \
    --tokenizer configs/tokenizer/seed_llama_tokenizer.yaml \
    --model configs/llm/seed_llama_14b.yaml \
    --port 7890 \
    --llm_device cuda:0 \
    --tokenizer_device cuda:1