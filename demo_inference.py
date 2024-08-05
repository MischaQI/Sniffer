import cv2
import gradio as gr
import torch
from PIL import Image
import os 
import json
from datetime import datetime

from lavis.models import load_model_and_preprocess

from my_utils import detect_web, find_new_file
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

print ("loading mllm")
mllm, vis_processors, _ = load_model_and_preprocess(name="blip2_vicuna_instruct", model_type="vicuna13b", is_eval=True, device=torch.device("cuda:0"))
mllm.load_checkpoint('llm-ckpt/checkpoint_best.pth')

print ("loading llm")
model_id = "llm-ckpt/Llama-2-13b-chat-hf"
llm = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16).to('cuda:0').eval()
tokenizer = AutoTokenizer.from_pretrained(model_id)

path_log = "demo/logs/"

enable_btn = gr.Button.update(interactive=True)
disable_btn = gr.Button.update(interactive=False)


def get_instruction_mllm(text, visent):
    prompt = f"Some rumormongers use images from other events as illustrations of current news event to make multimodal misinformation. Given a news caption and a news image, judge whether the given image is wrongly used in a different news context. Let's analyze their inconsistency from perspectives of main news elements, including time, place, person, event, artwork, etc. You should answer in the following forms: 'No, the image is rightly used.' or 'Yes, the image is wrongly used in a different news context. The given news caption and image are inconsistent in <element>. The <element> in caption is <entity_1>, and the <element> in image is <entity_2>. ' The news caption is '{text}'. The possible visual entities is {','.join(visent)}. The answer is "
    return prompt

def revise_output(output):
    import re
    modified_text = re.sub(r'^(Yes|No),\s+', '', output)
    modified_text = re.sub(r'\s+element:.*', '', modified_text)
    modified_text = modified_text[0].upper() + modified_text[1:]
    return (modified_text)

def get_prompt_evidence():
    system_prompt = "You will be provided with a claim and some retrieved evidence. You need to determine whether the given claim is supported by these evidence, meaning whether they describe the same news event. It is sufficient if any evidence supports the claim. Please provide your judgment followed by your reasoning."
    input_1 = "Claim: Nicky Henderson with Sprinter Sacre during an open day at his Seven Barrows stables in February 2013. \n Evidence: Cheltenham Gold Cup Winner Bob's Worth Nicky Henderson. || Nicky Henderson already planning fresh triumphs after Cheltenham | Cheltenham Festival 2013 | The Guardian. \n Your answer is:"
    output_1 = "The claim is not supported by the provided evidence. Reasoning: The claim specifically mentions 'Nicky Henderson with Sprinter Sacre during an open day at his Seven Barrows stables in February 2013.' The first piece of evidence talks about 'Cheltenham Gold Cup Winner Bob's Worth Nicky Henderson,' which refers to a different horse (Bob's Worth) and does not mention the event described in the claim (open day at Seven Barrows stables). The second piece of evidence mentions 'Nicky Henderson already planning fresh triumphs after Cheltenham | Cheltenham Festival 2013,' which is related to Nicky Henderson and the Cheltenham Festival in 2013, but does not specifically mention Sprinter Sacre or the open day at Seven Barrows stables in February 2013. Thus, while both pieces of evidence are related to Nicky Henderson and his activities in 2013, neither directly supports the specific scenario described in the claim."
 
    conversation_evidence = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": input_1}, 
        {"role": "assistant", "content": output_1},
        # {"role": "user", "content": input_2}, 
        # {"role": "assistant", "content": output_2},
    ]
    return conversation_evidence


def get_prompt_composed():
    system_prompt = "Given a news claim, a news image and several image-retrieved webpages, we aim to judge whether the given image is rightly used in the given claim. We analyze it from two perspectives: the consistency between the given claim and image, the relationship between the given claim and image-retrieved webpages. Based on the provided two-perspective analysis, please provide your judgment (i.e., The image is wrongly/rightly used in the given news claim.) followed by your reasoning. "
    input_1 = "News claim: Dead Israeli children were buried in mud. \n Analysis from the text-image consistency: The image is wrongly used in a differnet news context. The given news caption and image are inconsistent in event. The event in caption is Dead Israeli children being buried in mud, and the event in image is Children playing in a mud pit during the annual Holi festival. \n Analysis from the relationship between the given claim and image-retrieved webpage (evidence): The claim is not supported by the provided evidence. The evidence consists of various news articles and images related to an annual event called \"Mud Day\" in Michigan, where people engage in various mud-related activities such as playing in mud, getting covered in mud, and throwing mud at each other. The event is depicted as a fun and messy celebration, but there is no connection to dead Israeli children being buried in mud. Therefore, the claim is not supported by the provided evidence. \n Your answer is:"
    output_1 = "The image is wrongly used in the given claim. \n The reasoning is as follows:\n From the text-image consistency perspective, while the image and the text may appear to be consistent, a closer look reveals that the image does not accurately depict the content of the claim. The claim is about Israeli children being buried in mud, but the image shows people playing and having fun in mud, with no mention of Israeli children or any negative context. This inconsistency suggests that the image is not rightly used in the claim.\n From the perspective of the relationship between the given claim and image-retrieved webpage (evidence), the evidence does not support the claim. The image is from an annual mud day in Michigan, where people play and have fun in mud, but there is no mention of Israeli children being buried in mud. This lack of support from the evidence undermines the validity of the claim. \n Therefore, based on the provided analysis, the image is wrongly used in the given claim."
    conversation_composed = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": input_1}, 
        {"role": "assistant", "content": output_1},
        # {"role": "user", "content": input_2}, 
        # {"role": "assistant", "content": output_2},
    ]
    return conversation_composed


def sniffer_detect(image_path, text, progress=gr.Progress()):
    progress(0, desc="Receiving request...")
    searchid = datetime.now().strftime("%D-%H:%M:%S").replace('/','-')

    one_dict = {}
    one_dict['text'] = text

    raw_image = Image.open(image_path).convert("RGB")
    raw_image.save(f"{path_log}{searchid}.jpg")

    image = vis_processors["eval"](raw_image).unsqueeze(0).to('cuda:0')

    progress(0.2, desc="Retrieving image...")
    try:
        annotations, inverse_search_results, visent, evidence = detect_web(image_path)
    except:
        annotations, inverse_search_results, visent, evidence = "error", "error", [], []
    
    one_dict['annotations'] = str(annotations)
    one_dict['inverse_search_results'] = inverse_search_results
    one_dict['visent'] = visent
    one_dict['evidence'] = list(evidence)

    # mllm detect
    with torch.no_grad():
        if len(one_dict['evidence'])==0:
            progress(0.5, desc="Analyzing text-image content consistency...")
        else:
            progress(0.4, desc="Analyzing text-image content consistency...")
        answer_mllm  = mllm.generate({"image": image, "prompt": get_instruction_mllm(text, visent)})[0]
        
    answer_mllm = answer_mllm.replace("<s>","").strip()
    answer_mllm = revise_output(answer_mllm)
    one_dict['answer_mllm'] = answer_mllm

    if len(one_dict['evidence'])==0:
        progress(1, desc="Finished")
        with open(f"{path_log}{searchid}.json",'w') as fw:
            json.dump(one_dict, fw, indent=4)
        return (answer_mllm,) + (enable_btn,) * 3
    else:
        progress(0.6, desc="Analyzing relationship between given claim and image-retrieved webpage...")


    # llm evidence detect
    input_evidence = f"Claim: {text} \n Evidence: {' || '.join('%s' %a for a in evidence)} \n Your answer is:"
    conversation_evidence = get_prompt_evidence()
    conversation_evidence.append({"role": "user", "content": input_evidence})
    input_ids_evidence = tokenizer.apply_chat_template(conversation_evidence, return_tensors="pt").to('cuda:0') 
    with torch.no_grad():
        output_llm = llm.generate(input_ids_evidence, max_new_tokens=512, do_sample=False)
        answer_evidence = tokenizer.decode(output_llm[0][len(input_ids_evidence[0]):], skip_special_tokens=True).replace('[/INST]','').strip()
        one_dict['answer_evidence'] = answer_evidence
        progress(0.8, desc="Summarizing...")


    # llm composed reasoning
    input_composed = f"News claim: {text} \n Analysis from the text-image consistency: {answer_mllm} \n Analysis from the relationship between the given claim and image-retrieved webpage (evidence): {answer_evidence} \n Your answer is:"
    conversation_composed = get_prompt_composed()
    conversation_composed.append({"role": "user", "content": input_composed})
    input_ids_composed = tokenizer.apply_chat_template(conversation_composed, return_tensors="pt").to('cuda:1') 
    with torch.no_grad():
        output_llm_2 = llm.generate(input_ids_composed, max_new_tokens=512, do_sample=False)
        answer_composed = tokenizer.decode(output_llm_2[0][len(input_ids_composed[0]):], skip_special_tokens=True).replace('[/INST]','').strip()
        one_dict['answer_composed'] = answer_composed


    progress(1, desc="Finished")
    with open(f"{path_log}{searchid}.json",'w') as fw:
        json.dump(one_dict, fw, indent=4)

    return (answer_composed,) + (enable_btn,) * 3


title_markdown = ("""
# SNIFFER: Multimodal Large Language Model for Explainable Out-of-Context Misinformation Detection
[[Project Page](https://pengqi.site/Sniffer/)] [[Model](https://huggingface.co/MischaQI/SNIFFER)] [[Paper](https://openaccess.thecvf.com/content/CVPR2024/html/Qi_SNIFFER_Multimodal_Large_Language_Model_for_Explainable_Out-of-Context_Misinformation_Detection_CVPR_2024_paper.html)] 
""")

def create_multi_modal_demo():
    with gr.Blocks() as instruct_demo:
        with gr.Row():
            with gr.Column():
                img = gr.Image(label='Image', type='filepath', sources='upload')
                question = gr.Textbox(lines=2, label="Text")

                run_botton = gr.Button("Detect Out-of-Context Using")
            with gr.Column():
                result = gr.Textbox(lines=15, label="Output")
                with gr.Row():
                    upvote_btn = gr.Button(value="👍  Upvote", interactive=False)
                    downvote_btn = gr.Button(value="👎  Downvote", interactive=False)
                    flag_btn = gr.Button(value="⚠️  Flag", interactive=False)

        inputs = [img, question]
        outputs = [result, upvote_btn, downvote_btn, flag_btn]

        examples = [
        ["imgs/22-bbc-images-0440-388.jpg", "Thousands of people march in Madrid against the Israel Hamas war.", 256, 0.1, 0.75], 
        ["imgs/mud.jpg", "Dead Israeli children were buried in mud.", 256, 0.1, 0.75]
        ]
 
        gr.Examples(
            examples=examples,
            inputs=inputs,
            outputs=outputs,
            fn=sniffer_detect,
            cache_examples=False, 
        )
        run_botton.click(fn=sniffer_detect,
                         inputs=inputs, outputs=outputs)

        upvote_btn.click(
            fn=upvote_last_response,
            outputs=[upvote_btn, downvote_btn, flag_btn],
            queue=False
        )
        downvote_btn.click(
            fn=downvote_last_response,
            outputs=[upvote_btn, downvote_btn, flag_btn],
            queue=False
        )
        flag_btn.click(
            fn=flag_last_response,
            outputs=[upvote_btn, downvote_btn, flag_btn],
            queue=False
        )
    return instruct_demo


def upvote_last_response():
    newest_path = find_new_file(path_log)
    
    one_dict = json.load(open(newest_path, 'r'))
    one_dict['review'] = "upvote"

    with open(newest_path, 'w') as fw:
        json.dump(one_dict, fw, indent=4)

    return (disable_btn,) * 3

def downvote_last_response():
    newest_path = find_new_file(path_log)
    
    one_dict = json.load(open(newest_path, 'r'))
    one_dict['review'] = "downvote"

    with open(newest_path, 'w') as fw:
        json.dump(one_dict, fw, indent=4)

    return (disable_btn,) * 3

def flag_last_response():
    newest_path = find_new_file(path_log)
    
    one_dict = json.load(open(newest_path, 'r'))
    one_dict['review'] = "flag"

    with open(newest_path, 'w') as fw:
        json.dump(one_dict, fw, indent=4)

    return (disable_btn,) * 3



with gr.Blocks(css="h1,p {text-align: center;}") as demo:
    with gr.Row():
        with gr.Column(scale=1, min_width=70):
            gr.Image("logo.png",height=70,show_download_button=False,container=False,interactive=False)
        with gr.Column(scale=30, min_width=70):
            gr.Markdown(title_markdown)
    with gr.TabItem(""):
        # "Out-of-Context multimodal misinformation detection"
        create_multi_modal_demo()

demo.queue().launch(share=True)