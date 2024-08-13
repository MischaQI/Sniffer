def detect_web_v1(path):
    """Detects web annotations given an image."""
    from google.cloud import vision

    client = vision.ImageAnnotatorClient()

    with open(path, "rb") as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response = client.web_detection(image=image)
    annotations = response.web_detection

    print (annotations)


    # visual entities 
    vis_ents = []

    if annotations.best_guess_labels:
        for label in annotations.best_guess_labels:
            vis_ents.append(label.label)

    if annotations.web_entities:
        for entity in annotations.web_entities:
            if entity.description:
                vis_ents.append(entity.description)

    # matched_pages_title and urls
    titles = []
    urls = []

    if annotations.pages_with_matching_images:
        for page in annotations.pages_with_matching_images:
            urls.append(page.url)
            titles.append(page.page_title)

    if response.error.message:
        raise Exception(
            "{}\nFor more info on error messages, check: "
            "https://cloud.google.com/apis/design/errors".format(response.error.message)
        )
    
    return vis_ents, titles, urls


from utils import get_captions_from_page,save_html
import fasttext
import os

PRETRAINED_MODEL_PATH = 'llm-ckpt/lid.176.bin'
model = fasttext.load_model(PRETRAINED_MODEL_PATH)


def get_captions_and_process(img_url,page_url,page, save_folder_path,file_save_counter):
    caption,title,code,req = get_captions_from_page(img_url, page_url)
                
    if title is None: title = ''   
    page_title = page.page_title if page.page_title else ''
    if len(title) < len(page_title.lstrip().rstrip()):
        title = page_title
    if title: 
        lang_pred = model.predict(title.replace("\n"," "))
        if lang_pred[0][0] != '__label__en':
            return{}
    saved_html_flag = save_html(req, os.path.join(save_folder_path,str(file_save_counter)+'.txt'))     
    if saved_html_flag:            
        html_path = os.path.join(save_folder_path,str(file_save_counter)+'.txt')
    else:
        html_path = ''
        
    if caption:
        new_entry = {'page_link':page_url,'image_link':img_url,'title': title, 'caption':caption, 'html_path': html_path}  
    else:
        caption,title,code,req = get_captions_from_page(img_url,page_url,req,15) # hashing_cutoff
        if caption:
            new_entry = {'page_link':page_url,'image_link':img_url, 'caption':caption, 'html_path': html_path, 'matched_image': 1}
        else:            
            new_entry = {'page_link':page_url,'image_link':img_url, 'html_path': html_path}  
    if title: 
        new_entry['title'] = title    
    return new_entry


def get_inverse_search_annotation(annotations,id_in_clip,save_folder_path): 
    file_save_counter = -1

    vis_ents = []
    titles = []
    caption_extracted = []

    all_fully_matched_captions_webs = []
    all_fully_matched_no_caption_webs = []

    if annotations.best_guess_labels:
        for label in annotations.best_guess_labels:
            vis_ents.append(label.label)
    if annotations.web_entities:
        for entity in annotations.web_entities:
            if entity.description:
                vis_ents.append(entity.description)


    if annotations.pages_with_matching_images:
        for page in annotations.pages_with_matching_images:
            # urls.append(page.url)
            if page.full_matching_images:
                titles.append(page.page_title)
        
            file_save_counter = file_save_counter + 1
            new_entry = {}
            if page.full_matching_images:
                for image_url in page.full_matching_images:
                    try:
                        new_entry = get_captions_and_process(image_url.url, page.url,page,save_folder_path,file_save_counter)
                    except: 
                        print("Error in getting captions - id in clip:%5d"%id_in_clip)
                        continue
                    if not 'caption' in new_entry.keys(): 
                        continue 
                    else:
                        break 
                if 'caption' in new_entry.keys():
                    all_fully_matched_captions_webs.append(new_entry)   
                    caption_extracted.append(new_entry['caption'])   
                elif new_entry:
                    all_fully_matched_no_caption_webs.append(new_entry)
                        
    annotations_revised = {'entities': vis_ents,'titles': titles, 'caption_extracted': caption_extracted, 'all_fully_matched_captions_webs': all_fully_matched_captions_webs, 'all_fully_matched_no_caption_webs': all_fully_matched_no_caption_webs} 
    return annotations_revised 



def detect_web(path):
    """Detects web annotations given an image."""
    from google.cloud import vision

    client = vision.ImageAnnotatorClient()

    with open(path, "rb") as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response = client.web_detection(image=image)
    annotations = response.web_detection

    inverse_search_results = get_inverse_search_annotation(annotations,path,"demo/saved/")

    visent = inverse_search_results['entities']
    evidence = set(inverse_search_results['titles'])
    for item in inverse_search_results['caption_extracted']:
        for key, value in item.items():
            evidence.add(value)

    for tlist in [inverse_search_results['all_fully_matched_captions_webs'], inverse_search_results['all_fully_matched_no_caption_webs']]:
        for one_dict in tlist:
            if 'title' in one_dict.keys():
                if one_dict['title']!='Cloudflare capcha page':
                    evidence.add(one_dict['title'])
            if 'caption' in one_dict.keys():
                for key,value in one_dict['caption'].items():
                    evidence.add(value)

    print (visent)
    print (evidence)

    return annotations, inverse_search_results, visent, evidence



def find_new_file(dir):
    file_lists = os.listdir(dir)
    file_lists.sort(key=lambda fn: os.path.getmtime(dir + fn)
                    if not os.path.isdir(dir + fn) else 0)
    file = os.path.join(dir, file_lists[-1])
    return file