<div align=center>  <img src="assets/logo.png" width=60%></div>

*Keyworks: Concealed Visual Perception, Concealment Counteraction, Multimodal Concealment Dataset, Multimodal Large Language Model, Camouflage Object Detection, Survey, Taxnomy*

---

[**Pancheng Zhao**](https://www.zhaopancheng.top)<sup>1</sup>· [**Deng-Ping Fan**](https://dengpingfan.github.io/)<sup>1</sup> · Shupeng Cheng<sup>2</sup>  · Salman Khan<sup>3</sup> · 

Fahad Shahbaz Khan<sup>3</sup> · David Clifton<sup>4</sup> · [**Peng Xu**](https://scholar.google.com/citations?user=9_v4tC0AAAAJ&hl=zh-CN&oi=sra)<sup>2+</sup> · [**Jufeng Yang**](https://cv.nankai.edu.cn/)<sup>1,5,6+</sup>

<sup>1</sup> VCIP & TMCC & DISSec, College of Computer Science, Nankai University

<sup>2</sup> Department of Electronic Engineering, Tsinghua University

<sup>3</sup> Mohammed Bin Zayed University of Artificial Intelligence 

<sup>4</sup> Department of Engineering Science, University of Oxford

<sup>5</sup> Nankai International Advanced Research Institute (SHENZHEN· FUTIAN)

<sup>6</sup> Pengcheng Laboratory, China

<sup>+</sup>corresponding authors

<a href="http://arxiv.org/abs/2504.10979"><img src='https://img.shields.io/badge/arXiv-CDP-red' alt='Paper PDF'></a>	<a href='https://zhaopancheng.top/publication'><img src='https://img.shields.io/badge/Project_Page-CDP-green' alt='Project Page'></a>

---

**Concealed Vision** consists of a series of challenging and complex tasks. The targets of these tasks cause confusion in low-level feature extraction and high-level semantic understanding by conveying limited visual information. We focus on those tasks which learns to map concealed targets in the input image to
complex output structures, called **Concealed Dense Prediction**.

- **Our Contribution.** 

  - Detailed survey for the CDP area, including analysis of 14 concealment mechanism.

    <div align=center>
    <img src="assets/concealment_mechanism.jpg" width=500/><img src="assets/concealment_mechanism_taxonomy.jpg" width=200/>
    </div>

  - A **taxonomy of CDP** deep learning techniques in the vein of concealment counteracting.

    <div align=center>
    <img src="assets/concealment counteracting.jpg" width=800/>
    </div>

  - A large-scale multimodal instruction fine-tuning dataset in Concealed Vision area, **CvpINST**

  - An attempt at a unified paradigm in Concealed Visual Perception community, **CvpAgent**

    <div align=center>
    <img src="assets/CVPAgent.jpg" width=800/>
    </div>

  

- **Quick View.** 

  - The following demonstrates some of the capabilities of our concealed visual perception-specific multimodal AI assistant, CvpAgent. Simply accepting pictures and user commands, the model sets multiple concealed vision tasks into one via interactive dialog.
  
    <div align=center>
    <img src="assets/demo.gif" width=600/>
    </div>



## 1. News

* **🔥2025-04-15🔥:** Code and dataset officially released as open source.
* **2025-02-25:** **Creating repository.** 



## 2. Get Start of CvpINST and CvpAgent 



#### 1. Requirements

Before running the demo, you need to install the dependencies of Qwen and SAM2.

You can install the official dependencies of them in one conda environment. 

Or install the compatible version we provide with the following steps:

```
conda create -n cvpagent python=3.10
conda activate cvpagent
# Qwen2.5 VL
pip install -r requirements.txt
pip install qwen-vl-utils
pip install git+https://github.com/huggingface/transformers.git@9d2056f12b66e64978f78a2dcb023f65b2be2108 accelerate
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.9.post1/flash_attn-2.5.9.post1+cu122torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install flash_attn-2.5.9.post1+cu122torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
# SAM2
cd special_list/segment_anything_2
pip install -e ".[demo]"
cd ../..
```

<details>
  <summary><b>Frequently asked questions about the environment installation</b></summary>
<ol>
  <li> <b>accelerate installation failed</b></li>
    Usually it's a network problem that can be solved with a few more tries 
  <li> <b>symbol __nvJitLinkComplete_12_4 version libnvJitLink.so.12 not defined in file libnvJitLink.so.12 with link time reference</b></li>
ln -s /path/to/envs/cvpagent/lib/python3.10/site-packages/nvidia/nvjitlink/lib/libnvJitLink.so.12 /path/to/envs/cvpagent/lib/python3.10/site-packages/nvidia/cusparse/lib/libnvJitLink.so.12
<br>
export LD_LIBRARY_PATH=/path/to/envs/cvpagent/lib/python3.10/site-packages/nvidia/cusparse/lib:$LD_LIBRARY_PATH
</ol>
</details>



#### 2. Download Datasets and Checkpoints.

##### Datasets:

We collected and organized the dataset **CvpINST** from existing datasets. 

Our dataset contains **three parts**:

- **Concealed Images**: 81k+ images collected from 20 Datasets. Category labels for some of the images were manually labeled, and the data were organized in the following form:

```shell
CvpINST                         % dataset root
├── JSON                        % annotation
├── train                       % trainset
│   ├── image                   % images
│   │   ├── Concealed           % hierarchical categorization structure  
│   │   │   ├── Artificial 
│   │   │   │   ├── Copymove  
│   │   │   │   ├── ...
│   │   │   ├── Biological  
│   │   │   │   ├── Aquatic                                       
│   │   │   │   │   ├── Batfish                                                
│   │   │   │   │   ├── ...
│   │   │   ├── Optical  
│   │   │   │   ├── Glass                                       
│   │   │   │   │   ├── Bottle                                                
│   │   │   │   │   ├── ...
│   │   ├── General
│   ├── mask                    % pixel-wised binary mask
│   ├── instance                % instance annotation （optional）
├── test                        % testset
│   ├── image
│   ├── mask
│   ├── instance
```

- **Human-machine Dialogues:**  one-round or multi-round conversational dialogue about 4 tasks

- **Intermediate results:** Some intermediate results, including Image Captions, Prompt templates and so on.

  **Specifically, the json folder contains the following details：**

```
JSON
├── CVP_test.json           % Original annotation of the dataset
├── CVP_train.json
├── CvpINST_test.json       % Human-machine Dialogues instruction dataset
├── CvpINST_train.json
├── image_caption.json      % the description about concealment
└── prompt_template.json    % Prompt templates used in constructing dialogues
```

​		**Examples of samples from CVP_train.json/CVP_test.json:**

```json
{
        "idx": 5283,
        "source_data": {
            "image_name": "PlantCAMO_PC_Aloe_striata_1.jpg",
            "anno": "The image depicts a succulent ...",
            "category": "Concealed/Biological/Aloe/Striata"
        },
        "image": "path to image",
        "mask": "path to mask",
        "instance": null,
        "width": 5472,
        "height": 3648,
        "bbox": [
            599.0,
            810.0,
            2283.0,
            2431.0
        ],
        "dataset_name": "PlantCAMO"
    }
```

​		**Examples of samples from CvpINST_train.json/CvpINST_test.json:**

```json
{
        "idx": 5283,
        "source_data": {
            "image_name": "PlantCAMO_PC_Aloe_striata_1.jpg",
            "anno": null,
            "category": "Concealed/Biological/Aloe/Striata"
        },
        "image": "path to image",
        "mask": "path to mask",
        "dataset_name": "PlantCAMO",
        "conversations": [
            {
                "from": "human",
                "value": "<image>\nCan any part of the image be considered concealed due to similar background patterns?"
            },
            {
                "from": "gpt",
                "value": "Yes, part of the image is concealed by similar background patterns."
            },
            {
                "from": "human",
                "value": "Please elaborate on how the target is camouflaged or obscured in the identified region."
            },
            {
                "from": "gpt",
                "value": "..."
            }
        ],
        "task": "CAP"
    }
```

**📥 How to Access the Dataset**

You can obtain access to the dataset by filling out the following form:

**Please note:**

- Make sure the information you provide (especially your affiliation and email address) is accurate. This information is used **solely for access authorization purposes**, and will **not be shared or used for any other purpose**.
- The download link currently uses **BaiduNetdisk**. Access may be restricted or unstable in certain regions — please verify availability in advance.

|                        Datasets                         |                     Request Form                      |
| :-----------------------------------------------------: | :---------------------------------------------------: |
|   Full Dataset (Images + Annotations + Instructions)    | 👉 [Request Form](https://forms.gle/WpB9eSieEDfz7bQX9) |
|                        Full JSON                        | 👉 [Request Form](https://forms.gle/EWHtVY35iwsUgEv28) |
|        Original Annotation (Cvp_train/test.json)        | 👉 [Request Form](https://forms.gle/7cPjDQ59AnHKSBvh6) |
|    Human-machine Dialogues (CvpINST_train/test.json)    | 👉 [Request Form](https://forms.gle/NB77j9saF5WsEMg56) |
| Intermediate Results (Image Captions, Prompt templates) | 👉 [Request Form](https://forms.gle/PH8TuzQ4BeDzJFRn6) |



##### Checkpoint:

The Official Released Models:

|  Model  |                       LINK                       |
| :-----: | :----------------------------------------------: |
|  SAM2   | [Link](https://github.com/facebookresearch/sam2) |
| Qwen-VL |   [Link](https://github.com/QwenLM/Qwen2.5-VL)   |

Our fine-tuned version

|         Model          |                     Request Form                     |
| :--------------------: | :--------------------------------------------------: |
|  CvpAgent-Qwen2-VL-7B  | 👉[Request Form](https://forms.gle/6KgWAmrSh4nFLjqV8) |
| CvpAgent-Qwen2.5-VL-3B | 👉[Request Form](https://forms.gle/6KgWAmrSh4nFLjqV8) |
| CvpAgent-Qwen2.5-VL-7B | 👉[Request Form](https://forms.gle/6KgWAmrSh4nFLjqV8) |



#### 3. Quick Demo:

You can quickly experience the model with the following commands:

~~~ 
pip install gradio
python -m src.serve.app_cvpagent \
    --model-path /path/to/finetuned/weight \
    --sam2_checkpoint /path/to/sam2/weight
~~~



#### 4. Train

##### 4.1 Finetune with LoRA

~~~
bash scripts/finetune_lora_vision.sh
~~~

##### 4.2 Merge LoRA Weights

~~~
bash scripts/merge_lora.sh
~~~



## Contact

If you have any questions, please feel free to contact me:

zhaopancheng@mail.nankai.edu.cn

pc.zhao99@gmail.com



## Citation

If you find this project useful, please consider citing:

```
@article{zhao2025deep,
  title={Deep Learning in Concealed Dense Prediction},
  author={Zhao, Pancheng and Fan, Deng-Ping and Cheng, Shupeng and Khan, Salman and Khan, Fahad Shahbaz and Clifton, David and Xu, Peng and Yang, Jufeng},
  journal={arXiv preprint arXiv:2504.10979},
  year={2025}
}
```



## Acknowledgements

We gratefully acknowledge the contributions of the following projects, which served as the foundation and inspiration for our work:

- [2U1/Fine-tuning Qwen2-VL Series](https://github.com/2U1/Qwen2-VL-Finetune) 
- [facebookresearch/sam2](https://github.com/facebookresearch/sam2)
