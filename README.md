# I Know About “Up”! Enhancing Spatial Reasoning in Visual Language Models Through 3D Reconstruction

This ZeroVLM model improves the visual-spatial reasoning capabilities of vision-language models. Details about ZeroVLM can be found in our paper. 

## Introduction 
Visual Language Models (VLMs) are essential for various tasks, particularly the visual reasoning tasks, due to their robust multi-modal information integration, visual reasoning capabilities, and contextual awareness. However, existing VLMs visual spatial reasoning capabilities are often inadequate, struggling even with basic tasks such as distinguishing left from right. To address this, we propose the ZeroVLM model, designed to enhance the visual spatial reasoning abilities of VLMs. ZeroVLM employs Zero-1-to-3, a 3D reconstruction model for obtaining different views of the input images and incorporates a view prompt to further improve visual spatial reasoning. Experimental results on four visual spatial reasoning datasets show that our ZeroVLM achieves up to 19.48\% accuracy improvement, which indicates the effectiveness of 3D reconstruction and view prompt of our ZeroVLM. 

![image](https://github.com/zhouhao028/Iknow_up/blob/main/Figures/model.png)

## Prepare Data 
The datasets used in our experiments can be downloaded from their official websites: [VSR](https://github.com/cambridgeltl/visual-spatial-reasoning), [Whatsup_vlm](https://github.com/amitakamath/whatsup_vlms).

## 3D Reconstruction 
[Zero-1-to-3](https://github.com/cvlab-columbia/zero123) is a model designed for visual reconstruction tasks. We use Zero-1-to-3 as our model component to perform 3D reconstruction on the dataset.

## Data Augmentation by 3D Reconstruction 
In our work, we use Zero-1-to-3 for 3D reconstruction on the dataset to generate single-view images from different viewpoints, e.g. the left, right and random views. Our investigation not only explores whether the creation of single-view images can enhance the visual spatial reasoning capabilities of VLMs but also examines whether multi-view images can help this improvement, where multi-view images are synthesized from different single-view images.

![image](https://github.com/zhouhao028/Iknow_up/blob/main/Figures/DataSet.png)

## View Prompt 
To further explore whether the context prompt can enhance VLMs visual spatial reasoning ability, we introduced a special prompt called view prompt in the experiment. This view prompt varies depending on the input image and its views. We designed a variety of view prompts based on the content of different view images to guide VLMs to better understand and reason about the view spatial relationship between the target object and other objects in the image. The following figure shows two view prompt examples over a single view and multiple views of the input.

![image](https://github.com/zhouhao028/Iknow_up/blob/main/Figures/View%20Prompt.png)

## Baselines And Backbones
In our study, we use [LLaVA](https://llava-vl.github.io/) and [MiniGpt-4](https://minigpt-4.github.io/) as baseline models. We use LLaVA and MiniGpt-4 as the backbone VLMs of our ZeroVLM in all the experiments. In particular, we denote ZeroVLM (LLaVA) as our model based on the LLaVA model, which can process image inputs and improve efficiency and performance through joint learning of image and text instructions, while  ZeroVLM (MiniGpt-4) as our model based on the MiniGpt-4 model, which combines the powerful generation ability of language models with visual information capabilities.

## Evaluation Metric 
For evaluation, we judge the accuracy of the visual spatial reasoning ability of ZeroVLM (LLaVA) and ZeroVLM (MiniGpt-4) based on the answers answered by ZeroVLM (LLaVA) and ZeroVLM (MiniGpt-4). 

### 1. ZeroVLM (LLaVA) 
Run the following command in [LLaVA](https://llava-vl.github.io/) (Our run_llava.py is inconsistent with the run_llava.py in [LLaVA](https://llava-vl.github.io/). Please replace [LLaVA](https://llava-vl.github.io/)/llava
/eval/run_llava with our run_llava.py): 
```
python run_llava.py \
    --model-path \
    --images \
    --query 
```
where `--model-path` specifies the folder containing the test model, `--images` indicates that multiple image inputs are supported, and `--query` indicates the question corresponding to the image. 

### 2. ZeroVLM (MiniGpt-4) 
Run the following command to construct reasoning chains:
```
python construct_reasoning_chains.py \
  --dataset hotpotqa \
  --input_data_file data/hotpotqa/dev_with_kgs.json \
  --output_data_file data/hotpotqa/dev_with_reasoning_chains.json \
  --calculate_ranked_prompt_indices \
  --max_chain_length 4 
```
The `calculate_ranked_prompt_indices` parameter denotes whether to use a retriever model to adaptively choose demonstrations for each question. The number of demonstrations can be set with the `num_examplars` parameter. Additionally, the number of candidate triples $K$ can be set with `num_choices` parameter. 


## Contact: 
If you have any questions about the code, feel free to contact me via hao.zhou28@outlook.com.
