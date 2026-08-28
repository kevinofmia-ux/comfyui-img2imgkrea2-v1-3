# ============================================================
# KREA2 IMG2IMG - RUNPOD SERVERLESS
# ============================================================

FROM runpod/worker-comfyui:5.8.4-base

# TOKEN HF - TEST
ARG HF_TOKEN="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"

ENV HF_TOKEN=${HF_TOKEN}


# ============================================================
# OUTILS
# ============================================================

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git wget curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*


# ============================================================
# CUSTOM NODES
# ============================================================

# RGTHREE
RUN git clone https://github.com/rgthree/rgthree-comfy \
    /comfyui/custom_nodes/rgthree-comfy


# IMPACT PACK
# FaceDetailer + SAMLoader
RUN git clone https://github.com/ltdrdata/ComfyUI-Impact-Pack \
    /comfyui/custom_nodes/comfyui-impact-pack && \
    pip install --no-cache-dir \
    -r /comfyui/custom_nodes/comfyui-impact-pack/requirements.txt


# IMPACT SUBPACK
# UltralyticsDetectorProvider
RUN git clone https://github.com/ltdrdata/ComfyUI-Impact-Subpack \
    /comfyui/custom_nodes/comfyui-impact-subpack && \
    pip install --no-cache-dir \
    -r /comfyui/custom_nodes/comfyui-impact-subpack/requirements.txt


# CHROMAGRADE
RUN git clone https://github.com/MONKEYFOREVER2/ComfyUI-ChromaGrade \
    /comfyui/custom_nodes/ComfyUI-ChromaGrade && \
    cd /comfyui/custom_nodes/ComfyUI-ChromaGrade && \
    if [ -f requirements.txt ]; then \
        pip install --no-cache-dir -r requirements.txt; \
    fi


# ============================================================
# DOSSIERS MODELES
# ============================================================

RUN mkdir -p \
    /comfyui/models/diffusion_models \
    /comfyui/models/text_encoders \
    /comfyui/models/vae \
    /comfyui/models/upscale_models \
    /comfyui/models/loras \
    /comfyui/models/sams \
    /comfyui/models/ultralytics/bbox \
    /comfyui/input


# ============================================================
# KREA2 TURBO
# ============================================================

RUN HF_TOKEN=${HF_TOKEN} comfy model download \
    --url "https://huggingface.co/Comfy-Org/Krea-2/resolve/main/diffusion_models/krea2_turbo_nvfp4.safetensors" \
    --relative-path models/diffusion_models \
    --filename "krea2_turbo_nvfp4.safetensors"


# ============================================================
# TEXT ENCODER KREA2
# qwen3vl_4b_fp8_scaled.safetensors
# ============================================================

RUN HF_TOKEN=${HF_TOKEN} comfy model download \
    --url "https://huggingface.co/Comfy-Org/Krea-2/resolve/main/text_encoders/qwen3vl_4b_fp8_scaled.safetensors" \
    --relative-path models/text_encoders \
    --filename "qwen3vl_4b_fp8_scaled.safetensors"


# ============================================================
# VAE
# Le workflow API utilise wan21-vae.safetensors
# ============================================================

RUN HF_TOKEN=${HF_TOKEN} comfy model download \
    --url "https://huggingface.co/wangkanai/wan21-vae/resolve/main/vae/wan/wan21-vae.safetensors" \
    --relative-path models/vae \
    --filename "wan21-vae.safetensors"


# ============================================================
# UPSCALER
# ============================================================

RUN HF_TOKEN=${HF_TOKEN} comfy model download \
    --url "https://huggingface.co/ABDALLALSWAITI/Upscalers/resolve/main/photo/4xNomosWebPhoto_RealPLKSR.pth" \
    --relative-path models/upscale_models \
    --filename "4xNomosWebPhoto_RealPLKSR.pth"


# ============================================================
# SAM
# FaceDetailer / SAMLoader
# ============================================================

RUN wget --progress=dot:giga \
    -O /comfyui/models/sams/sam_vit_b_01ec64.pth \
    "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"


# ============================================================
# YOLO FACE DETECTOR
# UltralyticsDetectorProvider
# ============================================================

RUN wget --progress=dot:giga \
    -O /comfyui/models/ultralytics/bbox/face_yolov8m.pt \
    "https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8m.pt"


# ============================================================
# LORA SOFIA-KREA
# Repo Hugging Face privé
# ============================================================

RUN curl -L \
    -H "Authorization: Bearer ${HF_TOKEN}" \
    -o "/comfyui/models/loras/Sofia-KREA.safetensors" \
    "https://huggingface.co/datasets/Sofiavldzx/Sofia-KREA/resolve/main/Sofia-KREA.safetensors"


# ============================================================
# LORA REALISTIC SNAPSHOT KREA2
# ============================================================

RUN wget --progress=dot:giga \
    -O "/comfyui/models/loras/RealisticSnapshotKrea2.safetensors" \
    "https://civitai.com/api/download/models/2268008"


# ============================================================
# LORA LENOVO KREA2
# ============================================================

RUN wget --progress=dot:giga \
    -O "/comfyui/models/loras/lenovo_krea2.safetensors" \
    "https://civitai.com/api/download/models/3075606"


# ============================================================
# LORA HORNY AMATEUR
# ============================================================

RUN wget --progress=dot:giga \
    -O "/comfyui/models/loras/HORNY AMATEUR LORA .safetensors" \
    "https://drive.usercontent.google.com/download?id=1YBa6VCXJbeBzyoQskQ-lPl4HyRIIkiuN&export=download&confirm=t"


# ============================================================
# INPUT
# Les images img2img seront envoyées par l'API RunPod
# ============================================================

RUN mkdir -p /comfyui/input


# ============================================================
# VERIFICATION FINALE DU BUILD
# ============================================================

RUN echo "============================================" && \
    echo "CUSTOM NODES" && \
    echo "============================================" && \
    ls -lah /comfyui/custom_nodes && \
    echo "============================================" && \
    echo "DIFFUSION MODEL" && \
    echo "============================================" && \
    ls -lah /comfyui/models/diffusion_models && \
    echo "============================================" && \
    echo "TEXT ENCODER" && \
    echo "============================================" && \
    ls -lah /comfyui/models/text_encoders && \
    echo "============================================" && \
    echo "VAE" && \
    echo "============================================" && \
    ls -lah /comfyui/models/vae && \
    echo "============================================" && \
    echo "UPSCALER" && \
    echo "============================================" && \
    ls -lah /comfyui/models/upscale_models && \
    echo "============================================" && \
    echo "LORAS" && \
    echo "============================================" && \
    ls -lah /comfyui/models/loras && \
    echo "============================================" && \
    echo "SAM" && \
    echo "============================================" && \
    ls -lah /comfyui/models/sams && \
    echo "============================================" && \
    echo "YOLO" && \
    echo "============================================" && \
    ls -lah /comfyui/models/ultralytics/bbox && \
    echo "============================================" && \
    echo "BUILD READY" && \
    echo "============================================"
