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
    if [ -f /comfyui/custom_nodes/ComfyUI-ChromaGrade/requirements.txt ]; then \
        pip install --no-cache-dir \
        -r /comfyui/custom_nodes/ComfyUI-ChromaGrade/requirements.txt; \
    fi


# ============================================================
# DOSSIERS
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
# KREA2 MODEL
# ============================================================

RUN HF_TOKEN=${HF_TOKEN} comfy model download \
    --url "https://huggingface.co/Comfy-Org/Krea-2/resolve/main/diffusion_models/krea2_turbo_nvfp4.safetensors" \
    --relative-path models/diffusion_models \
    --filename "krea2_turbo_nvfp4.safetensors"


# ============================================================
# TEXT ENCODER KREA2 - LE BON
# ============================================================

RUN HF_TOKEN=${HF_TOKEN} comfy model download \
    --url "https://huggingface.co/Comfy-Org/Krea-2/resolve/main/text_encoders/qwen3vl_4b_fp8_scaled.safetensors" \
    --relative-path models/text_encoders \
    --filename "qwen3vl_4b_fp8_scaled.safetensors"


# ============================================================
# VAE
# Workflow API : wan21-vae.safetensors
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
# ============================================================

RUN wget --progress=dot:giga \
    -O /comfyui/models/sams/sam_vit_b_01ec64.pth \
    "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"


# ============================================================
# YOLO FACE
# ============================================================

RUN wget --progress=dot:giga \
    -O /comfyui/models/ultralytics/bbox/face_yolov8m.pt \
    "https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8m.pt"


# ============================================================
# LORA SOFIA-KREA - REPO HF PRIVÉ
# ============================================================

RUN HF_TOKEN=${HF_TOKEN} comfy model download \
    --url "https://huggingface.co/datasets/Sofiavldzx/Sofia-KREA/resolve/main/Sofia-KREA.safetensors" \
    --relative-path models/loras \
    --filename "Sofia-KREA.safetensors"


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
# Les images img2img seront envoyées via l'API RunPod.
# ============================================================

RUN mkdir -p /comfyui/input


# ============================================================
# VERIFICATION
# ============================================================

RUN echo "========== CUSTOM NODES ==========" && \
    ls -lah /comfyui/custom_nodes && \
    echo "========== DIFFUSION ==========" && \
    ls -lah /comfyui/models/diffusion_models && \
    echo "========== TEXT ENCODERS ==========" && \
    ls -lah /comfyui/models/text_encoders && \
    echo "========== VAE ==========" && \
    ls -lah /comfyui/models/vae && \
    echo "========== LORAS ==========" && \
    ls -lah /comfyui/models/loras && \
    echo "========== SAM ==========" && \
    ls -lah /comfyui/models/sams && \
    echo "========== YOLO ==========" && \
    ls -lah /comfyui/models/ultralytics/bbox && \
    echo "========== BUILD READY =========="
