import argparse
import torch
import requests
from PIL import Image
from io import BytesIO
import json
import os
from transformers import TextStreamer

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
    KeywordsStoppingCriteria,
)

def parse_jsonl(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def image_parser(args):
    if os.path.isdir(args.image_folder):
        image_files = [os.path.join(args.image_folder, f) for f in os.listdir(args.image_folder) if os.path.isfile(os.path.join(args.image_folder, f))]
        return image_files
    else:
        return []

def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image

def load_images(images):
    width = max(img.width for img in images)
    total_height = sum(img.height for img in images) + 25 * (len(images) - 1)

    new_img = Image.new('RGB', (width, total_height), (0, 0, 0))

    current_height = 0
    for img in images:
        new_img.paste(img, (0, current_height))
        current_height += img.height + 25

    return new_img

def setup_conv_mode(model_name):
    if 'llama-2' in model_name.lower():
        return "llava_llama_2"
    elif "v1" in model_name.lower():
        return "llava_v1"
    elif "mpt" in model_name.lower():
        return "mpt"
    else:
        return "llava_v0"
    
def initialize_conv(conv_mode, model_name, args):
    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
    else:
        args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    if "mpt" in model_name.lower():
        roles = ('user', 'assistant')
    else:
        roles = conv.roles

    return conv, roles

def process_input_images(args):
    images = [load_image(img_file) for img_file in args.images]
    image = load_images(images) if len(images) > 1 else images[0]

    if args.save_image:
        image.save("concat-image.jpg")

    return image

def process_image_tensor(image, image_processor, model):
    image_tensor = process_images([image], image_processor, model.config)
    if type(image_tensor) is list:
        image_tensor = [img.to(model.device, dtype=torch.float16) for img in image_tensor]
    else:
        image_tensor = image_tensor.to(model.device, dtype=torch.float16)

    return image_tensor


def run_inference(model, input_ids, image_tensor, args, tokenizer, caption):
    stop_str = ', please answer me yes or no'
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            streamer=streamer,
            use_cache=True,
            stopping_criteria=[stopping_criteria])

    outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
    
    # Check if 'yes' or 'no' in the generated output
    result = 1 if 'yes' in outputs.lower() else 0
    return result

def main(args):
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit, device=args.device)

    json_data = parse_jsonl(args.jsonl_file)
    
    correct_predictions = 0
    total_predictions = 0
    
    for item in json_data:
        image_file = item.get("image")
        caption = item.get("caption")
        label = item.get("label")
        
        if image_file is None or caption is None or label is None:
            print(f"Invalid data entry: {item}. Skipping...")
            continue
        
        image_path = os.path.join(args.image_folder, image_file)
        if not os.path.isfile(image_path):
            print(f"Image file not found: {image_path}. Skipping...")
            continue
        
        image = load_image(image_path)
        image_tensor = process_image_tensor(image, image_processor, model)

        try:
            inp = caption + ", please answer me yes or no"
        except EOFError:
            inp = ""
            
        prompt = inp
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)

        result = run_inference(model, input_ids, image_tensor, args, tokenizer, caption)
        
        total_predictions += 1
        if result == label:
            correct_predictions += 1
        
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    print(f"Accuracy: {accuracy * 100:.2f}% (Correct: {correct_predictions}, Total: {total_predictions})")

    return accuracy





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.5-13b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, required=True)
    parser.add_argument("--jsonl-file", type=str, required=True)
    parser.add_argument("--save-image", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--sep", type=str, default=",")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    main(args)

















