import math
import numpy as np
import torch
from PIL import Image
import torchvision.transforms.functional as TF
from nanovllm.utils.video_processing import sample_video_frames

IMAGE_MEAN = (0.5, 0.5, 0.5)
IMAGE_STD = (0.5, 0.5, 0.5)

def smart_resize(height: int, width: int, factor: int = 32,
                 min_pixels: int = 3136, max_pixels: int = 1003520) -> tuple[int, int]:
    if height < factor or width < factor:
        raise ValueError(f"Image too small: {height}x{width}, min edge = {factor}")

    h_bar = max(factor, round(height / factor) * factor)
    w_bar = max(factor, round(width / factor) * factor)

    if h_bar * w_bar > max_pixels:
        scale = math.sqrt(max_pixels / (h_bar * w_bar))
        h_bar = max(factor, math.floor(h_bar * scale / factor) * factor)
        w_bar = max(factor, math.floor(w_bar * scale / factor) * factor)

    if h_bar * w_bar < min_pixels:
        scale = math.sqrt(min_pixels / (h_bar * w_bar))
        h_bar = max(factor, math.ceil(h_bar * scale / factor) * factor)
        w_bar = max(factor, math.ceil(w_bar * scale / factor) * factor)

    return h_bar, w_bar 

def _frame_to_tensor(image: Image.Image, new_h: int, new_w: int) -> torch.Tensor:
    image = image.convert("RGB").resize((new_w, new_h), Image.BICUBIC)
    pixels = TF.to_tensor(image)
    return TF.normalize(pixels, IMAGE_MEAN, IMAGE_STD)

def _pad_temporal(pixels: torch.Tensor, temporal_patch_size: int) -> torch.Tensor:
    T = pixels.shape[0]
    remainder = T % temporal_patch_size
    if remainder: 
        pixels = torch.cat([pixels, pixels[-1:].repeat(temporal_patch_size - remainder, 1, 1, 1)], dim=0)
    return pixels

def _patchify_frames(
    pixels: torch.Tensor,
    patch_size: int,
    temporal_patch_size: int,
    merge_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    T, C, H, W = pixels.shape
    grid_t = T // temporal_patch_size 
    grid_h = H // patch_size
    grid_w = W // patch_size

    patches = pixels.view(
        grid_t, temporal_patch_size,
        C,
        grid_h // merge_size, merge_size, patch_size,
        grid_w // merge_size, merge_size, patch_size,
    )
    patches = patches.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
    pixel_values = patches.reshape(-1, C * temporal_patch_size * patch_size * patch_size)

    grid_thw = torch.tensor([[grid_t, grid_h, grid_w]], dtype=torch.long)

    return pixel_values, grid_thw

def _as_pil_image(frame) -> Image.Image:
    if isinstance(frame, Image.Image):
        return frame
    if isinstance(frame, np.ndarray):
        return Image.fromarray(frame)
    raise TypeError(f"不支持的帧类型: {type(frame)}，只支持 PIL.Image 或 numpy 数组")

def process_image(
    image: Image.Image,
    patch_size: int = 16,
    temporal_patch_size: int = 2,
    merge_size: int = 2, 
    min_pixels: int = 3136,
    max_pixels: int = 1003520,
) -> tuple[torch.Tensor, torch.Tensor]:

    image = image.convert("RGB")
    w, h = image.size
    factor = patch_size * merge_size

    new_h, new_w = smart_resize(h, w, factor, min_pixels, max_pixels)

    pixels = _frame_to_tensor(image, new_h, new_w).unsqueeze(0)  
    pixels = _pad_temporal(pixels, temporal_patch_size)           

    return _patchify_frames(pixels, patch_size, temporal_patch_size, merge_size)

def process_video(
    video: str | list | tuple,
    patch_size: int = 16,
    temporal_patch_size: int = 2,
    merge_size: int = 2,
    min_pixels: int = 3136,
    max_pixels: int = 1003520,
    frame_interval: int = 12,
) -> tuple[torch.Tensor, torch.Tensor]:

    if isinstance(video, str):
        frames, _timesteps = sample_video_frames(video, frame_interval)
    else:
        frames = [_as_pil_image(frame) for frame in video]
    if len(frames) == 0:
        raise ValueError("视频帧为空，无法处理")

    frame = frames[0].convert("RGB")
    w, h = frame.size
    factor = patch_size * merge_size
    new_h, new_w = smart_resize(h, w, factor, min_pixels, max_pixels)

    pixels = torch.stack([_frame_to_tensor(f, new_h, new_w) for f in frames], dim=0) 
    pixels = _pad_temporal(pixels, temporal_patch_size)

    return _patchify_frames(pixels, patch_size, temporal_patch_size, merge_size)

def process_messages(
    msg: dict,
    tokenizer,
    patch_size: int = 16,
    temporal_patch_size: int = 2,
    merge_size: int = 2,
    image_token_id: int = 248056,
    vision_start_id: int = 248053,
    vision_end_id: int = 248054,
) -> tuple[list[int], torch.Tensor | None, torch.Tensor | None]:
    pixel_values_list = []
    grid_thw_list = []

    text_parts = []
        
    role = msg["role"]
    content = msg["content"]
    if isinstance(content, str):
        text_parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
    elif isinstance(content, list):
        parts_text = []
        for part in content:
            if part["type"] == "image":
                img = part["image"]
                img = Image.open(img)
                pv, gt = process_image(img, patch_size, temporal_patch_size, merge_size)
                pixel_values_list.append(pv)
                grid_thw_list.append(gt)
                t, h, w = gt[0].tolist() 
                num_tokens = t * (h // merge_size) * (w // merge_size) 
                placeholder = "<|vision_start|>" + "<|image_pad|>" * num_tokens + "<|vision_end|>"
                parts_text.append(placeholder)
            elif part["type"] == "video":
                vid = part["video"]
                frames, timesteps = sample_video_frames(vid, 12)   
                pv, gt = process_video(frames, patch_size, temporal_patch_size, merge_size)
                pixel_values_list.append(pv)
                grid_thw_list.append(gt)

                t, h, w = gt[0].tolist() 
                num_tokens = (h // merge_size) * (w // merge_size)
                group_times = [
                    (timesteps[i] + timesteps[i + temporal_patch_size - 1]) / 2
                    for i in range(0, t * temporal_patch_size, temporal_patch_size)
                ]
                placeholder = "<|vision_start|>" + "".join(
                    f"<{group_time:.1f} seconds>"
                    + "<|vision_start|>" + "<|video_pad|>" * num_tokens + "<|vision_end|>"
                    for group_time in group_times
                ) + "<|vision_end|>"
                parts_text.append(placeholder)
            elif part["type"] == "text":
                parts_text.append(part["text"])
        text_parts.append(f"<|im_start|>{role}\n{''.join(parts_text)}<|im_end|>\n")

    full_text = "".join(text_parts) + "<|im_start|>assistant\n"
    token_ids = tokenizer.encode(full_text, add_special_tokens=False)

    if pixel_values_list:
        pixel_values = torch.cat(pixel_values_list, dim=0)
        image_grid_thw = torch.cat(grid_thw_list, dim=0)
    else:
        pixel_values = None
        image_grid_thw = None

    return token_ids, pixel_values, image_grid_thw
