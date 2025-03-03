import argparse
from threading import Thread
import gradio as gr
from PIL import Image
from src.utils import load_pretrained_model, get_model_name_from_path, disable_torch_init
from transformers import TextIteratorStreamer
from functools import partial
import warnings
from qwen_vl_utils import process_vision_info
import re
import tempfile
from gradio import ChatMessage
import numpy as np
from special_list.segment_anything_2.load_models import load_sam2
import torch
import matplotlib.pyplot as plt
import tempfile
import os
warnings.filterwarnings("ignore")

def show_mask(mask, ax, random_color=False, borders = True):
    # 显示遮罩
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        import cv2
        contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        # Try to smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2) 
    ax.imshow(mask_image)
    
def show_box(box, ax):
    # 显示坐标框
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))    

def is_video_file(filename):
    video_extensions = ['.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.mpeg']
    return any(filename.lower().endswith(ext) for ext in video_extensions)

def bot_streaming(message, history, generation_args):
    # Initialize variables
    
    # 处理用户上传的文件路径并保存到对应变量
    images = []
    videos = []

    if message["files"]:
        for file_item in message["files"]:
            if isinstance(file_item, dict):
                file_path = file_item["path"]
            else:
                file_path = file_item
            if is_video_file(file_path):
                videos.append(file_path)
            else:
                images.append(file_path)

    # 构造对话历史
    conversation = []
    latest_image_path = None
    for user_turn, assistant_turn in history: # 遍历history
        user_content = []
        if isinstance(user_turn, tuple):
            file_paths = user_turn[0]
            # user_text = user_turn[1]
            user_text = user_turn[1] if len(user_turn) > 1 else ""
            if not isinstance(file_paths, list):
                file_paths = [file_paths]
            for file_path in file_paths:
                if is_video_file(file_path):
                    user_content.append({"type": "video", "video": file_path, "fps":1.0})
                else:
                    user_content.append({"type": "image", "image": file_path})
                    latest_image_path = file_path
            if user_text:
                user_content.append({"type": "text", "text": user_text})
        else:
            user_content.append({"type": "text", "text": user_turn})
        conversation.append({"role": "user", "content": user_content})

        # 如果助手有回复，则存储到conversation
        if assistant_turn is not None:
            assistant_content = [{"type": "text", "text": assistant_turn}]
            conversation.append({"role": "assistant", "content": assistant_content})

    # 处理当前轮次的用户输入
    user_content = []
    for image in images:
        user_content.append({"type": "image", "image": image})
        latest_image_path = image
    for video in videos:
        user_content.append({"type": "video", "video": video, "fps":1.0})
    user_text = message['text']
    if user_text:
        user_content.append({"type": "text", "text": user_text})
    conversation.append({"role": "user", "content": user_content})

    prompt = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(conversation)
    
    inputs = processor(text=[prompt], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(device) 

    streamer = TextIteratorStreamer(processor.tokenizer, **{"skip_special_tokens": False, "skip_prompt": True, 'clean_up_tokenization_spaces':False,}) 
    generation_kwargs = dict(inputs, streamer=streamer, eos_token_id=processor.tokenizer.eos_token_id, **generation_args)

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    buffer = ""
    for new_text in streamer:
        if "<|im_end|>" in new_text:
            continue  # 跳过 <|im_end|>
        buffer += new_text # 可能是‘’
        print(new_text)
    
    downstream_instructins = extract_tags(buffer)
    
    if not downstream_instructins:
        yield buffer
    else:
        segmented_result = None
        if 'bbox' in downstream_instructins and latest_image_path:
            bboxs = [downstream_instructins['bbox']]
            segment_image = Image.open(latest_image_path)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                SEG_model.set_image(segment_image)
                masks, scores, _ = SEG_model.predict(
                    point_coords=None,
                    point_labels=None,
                    box=np.array(bboxs),
                    multimask_output=False,
                )
            if len(bboxs) == 1:
                res = masks.squeeze(0)
            else:
                res = np.max(masks, axis=0).squeeze(0)
            plt.figure(figsize=(10, 10))
            plt.imshow(segment_image)
            show_mask(res, plt.gca(), random_color=True)
            for box in bboxs:
                show_box(box, plt.gca())
            plt.axis('off')
            
            # 创建临时文件
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_file:
                # 保存图像到临时文件
                plt.savefig(temp_file.name, bbox_inches='tight', pad_inches=0)
                temp_file_path = temp_file.name
            
            
            yield {
                    "files": [{"path": temp_file_path}],
                    "text": buffer
                }


def extract_tags(buffer):
    """
    解析文本中的标签，存入 downstream 字典。
    """
    downstream = {}
    
    # 解析 <module> 标签内容
    module_match = re.search(r'<module>(.*?)</module>', buffer)
    if module_match:
        downstream['module'] = module_match.group(1)
        
    # 解析 <instruction> 标签内容
    instruction_match = re.search(r'<instruction>(.*?)</instruction>', buffer)
    if instruction_match:
        downstream['instruction'] = instruction_match.group(1)
    
    # 解析 <category> 标签内容
    category_match = re.search(r'<category>(.*?)</category>', buffer)
    if category_match:
        downstream['category'] = category_match.group(1)
    
    # 解析 <bbox> 标签内容
    bbox_match = re.search(r'<bbox>\{<(\d+)><(\d+)><(\d+)><(\d+)>}</bbox>', buffer)
    if bbox_match:
        downstream['bbox'] = list(map(int, bbox_match.groups()))
    
    return downstream

def main(args):

    global processor, model, device, SEG_model

    device = args.device
    
    disable_torch_init()

    use_flash_attn = True
    
    model_name = get_model_name_from_path(args.model_path)
    
    if args.disable_flash_attention:
        use_flash_attn = False

    processor, model = load_pretrained_model(model_base = args.model_base, model_path = args.model_path, 
                                                device_map=args.device, model_name=model_name, 
                                                load_4bit=args.load_4bit, load_8bit=args.load_8bit,
                                                device=args.device, use_flash_attn=use_flash_attn
    )
    
    # load speciallist
    SEG_model = load_sam2(args.sam2_checkpoint)

    chatbot = gr.Chatbot(scale=2)
    chat_input = gr.MultimodalTextbox(interactive=True, file_types=["image", "video"], placeholder="Enter message or upload file...",
                                  show_label=False)
    
    generation_args = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "do_sample": True if args.temperature > 0 else False,
        "repetition_penalty": args.repetition_penalty,
    }
    
    bot_streaming_with_args = partial(bot_streaming, generation_args=generation_args) # 偏函数，参数预设为generation_args

    with gr.Blocks(fill_height=True) as demo:
        gr.ChatInterface(
            fn=bot_streaming_with_args,
            title="Concealed Visual Perception Agent",
            stop_btn="Stop Generation",
            multimodal=True,
            textbox=chat_input,
            chatbot=chatbot
        )


    demo.queue(api_open=False)
    demo.launch(show_api=False, share=False, server_name='0.0.0.0')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--model-path", type=str, default="/mnt/sdb/zpc/Code/Qwen2.5-VL-Finetune/output/lora_vision_2.5_3B_data_v2_4gpu/merge/")
    # parser.add_argument("--model-base", type=str, default="/mnt/sdb/zpc/Dataset/models/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--model-path", type=str, default="/mnt/sdb/zpc/Code/Qwen2-VL-Finetune/output/lora_vision_7B_data_v2_b7a2/merge/")
    parser.add_argument("--model-base", type=str, default="/mnt/sdb/zpc/Dataset/models/Qwen2-VL-7B-Instruct/")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--disable_flash_attention", action="store_true")
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--debug", action="store_true")
    
    parser.add_argument("--special_list", type=list, default=['SAM2'])
    parser.add_argument("--sam2_checkpoint", type=str, default="/mnt/sdb/zpc/Dataset/models/SAM2/sam2_hiera_large.pt")
    args = parser.parse_args()
    main(args)