#!/usr/bin/env python
# coding: utf-8


from bs4 import NavigableString
import requests
import PIL
import shutil
from PIL import Image
import imghdr
from bs4 import BeautifulSoup
import bs4
import time
import io
import os
from bs4 import NavigableString
import numpy as np
import imagehash
import ast

def get_img_content_req(img_url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'}
        r = requests.get(img_url, headers=headers, stream=True,timeout=(60,60))
        #im_pil = Image.open(io.BytesIO(r.content))
        im_pil = Image.open(r.raw)
        im = np.asarray(im_pil)
        return im, im_pil 
    except:
        if not ('http' in img_url or 'https' in img_url):
            try:
                img_url2 = 'http:' + img_url
                im_pil = Image.open(requests.get(img_url2, headers=headers, stream=True,timeout=(60,60)).raw)
                im = np.asarray(im_pil)
                return im, im_pil
            except:
                try:
                    img_url3 = 'https:' + img_url
                    im_pil = Image.open(requests.get(img_url3, headers=headers, stream=True,timeout=(60,60)).raw)
                    im = np.asarray(im_pil)
                    return im, im_pil
                except:
                    return 0,0           
    return 0, 0

def compare_images(img1,img1_pil, img2, img2_pil, cutoff):
    try:
        if isinstance(img1,(np.ndarray)) and isinstance(img2,(np.ndarray)):
            if img1.shape == img2.shape:
                comp = img1 == img2
                return comp.all()
            else:
                hash0 = imagehash.average_hash(img1_pil)
                hash1 = imagehash.average_hash(img2_pil)
                diff = hash0 - hash1
                #print(diff)
                if diff < cutoff:
                    return True
                else:
                    #print('false1')
                    return False
        else:
            #print('false2')
            return False
    except:
        #print('false3')
        return False 
    
def get_html(url,method,browser=None):
    if method=='request':
        headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'}
        r = requests.get(url,timeout=(60,60))
        html = r.text
        code = str(r.status_code)
        if len(code)>0 and (code[0] == '4' or code[0] == '5'):
            #headers = {"User-Agent": "Mozilla/5.0 (X11; CrOS x86_64 13816.55.0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.86 Safari/537.36"}
            #headers = {"User-Agent":"Mozilla/5.0"}
            r = requests.get(url, headers=headers,timeout=(60,60))
            html = r.text 
            code = str(r.status_code)
    return r,html,code

def find_image_name(img_url):
    extensions=['.jpg','.jpeg','.JPG','.pjpeg','.pjp','.jfif','.png','.gif','.tiff','.svg','.webp','.avif','.apng']
    found_ext = ''
    image_file_name = ''
    for ext in extensions:
        if ext in img_url:
            found_ext = ext
            break 
    if found_ext != '':
        image_file_name = img_url.split('/')[-1].split(ext)[0]
    return image_file_name


def save_html(r,file_name):
    try:
        with open(file_name, 'w',encoding="utf8",newline='') as file:
            #file.write(r.text)
            file.write(r.content.decode('utf-8'))
        return 1
    except:
        return 0

def find_tags_by_matching(img_url, page_url, soup, cutoff):
    first_img, first_img_pil = get_img_content_req(img_url)
    images = soup.findAll('img')
    for image in images:
        if image.has_attr('src') and len(image['src']) > 0:
            img_url2 = image['src']
            second_img, second_img_pil = get_img_content_req(img_url2)
            #print(img_url2)
            comparison = compare_images(first_img,first_img_pil, second_img, second_img_pil, cutoff)
            if comparison:
                return [image]
        if image.has_attr('data-widths') and len(image['data-widths']) > 0 and 'https://www.nytimes.com/' in page_url:
            data_width_str = image['data-widths']
            try:
                data_width_dict = ast.literal_eval(data_width_str)
                for group in dict_data_width['MASTER']:
                    img_url2 = group['url']
                    second_img, second_img_pil = get_img_content_req(img_url2)
                    comparison = compare_images(first_img,first_img_pil, second_img, second_img_pil, cutoff)
                    if comparison:
                        return [image]
            except:
                return []  
        if image.has_attr('srcset') and len(image['srcset']) > 0:
            urls = image['srcset'].split(',')
            for url in urls:
                img_url2 = url.split(' ')[0]
                #print(img_url2)
                try:
                    second_img, second_img_pil = get_img_content_req(img_url2)
                    comparison = compare_images(first_img,first_img_pil, second_img, second_img_pil, cutoff)
                    if comparison:
                        return [image]
                except:
                    return []
    return []
    
def process_dict(input_dict):
    old_kepys = list(input_dict.keys())
    for key in old_kepys:
        if type(input_dict[key]) == str:
            if len(input_dict[key].strip()) == 0:
                del input_dict[key]
        elif input_dict[key] == None:
            del input_dict[key]
    return input_dict
    
def find_img_tag(soup,src_img_link):
    #exact matches 
    old_img_link = src_img_link
    special_cases_domains = ['https://www.irishtimes.com', 'https://www.stripes.com']
    for domain in special_cases_domains:
        if domain in src_img_link:
            #Example: https://www.stripes.com/news/middle-east/pentagon-more-us-fire-bases-could-open-in-iraq-1.403096
            #replace the domain name from the link bec. img urls in irishtimes is different than what is returned by the search engine
            src_img_link = src_img_link.replace(domain,'')
            break 
    images = soup.findAll('img',{'src': src_img_link})
    if len(images) == 0:
        images = soup.findAll('img',{'data-low-res-src': src_img_link})
    if len(images) == 0:
        images = soup.findAll('img',{'data-hi-res-src': src_img_link})
    if len(images) == 0:
        images = soup.findAll('img',{'data-raw-src': src_img_link})
    #Example: https://www.hollywoodreporter.com/news/general-news/sochi-paralympic-winter-games-begin-686837/
    if len(images) == 0:
        images = soup.findAll('img',{'data-lazy-src': src_img_link})
    #cover if the url is part of a set 
    attrb_names = ['srcset','data-low-res-src','data-hi-res-src','data-src','data-src-full16x9','data-src-mini1x1','data-src-mini','data-src-medium','data-src-large','data-src-xsmall','data-srcset','data-lazy-src','data-lazy','data-lazy-srcset','data-gl-src','data-gl-srcset','data-raw-src','data-widths']
    for attr in attrb_names:
        if len(images) == 0:
            images = soup.findAll(lambda tag:tag.name=="img" and tag.has_attr(attr) and src_img_link in tag[attr])
        else:
            break
    #media image tag instead of img tag 
    if len(images) == 0:
        images = soup.findAll(lambda tag:tag.name=="media-image" and tag.has_attr('image-set') and src_img_link in tag['image-set'])

    #if this was irishtimes returns
    #this is because all images in irishtimes have the same image name
    #so don't further check with the image name    
    for domain in special_cases_domains:
        if domain in src_img_link:
            #replace the domain name from the link bec. img urls in irishtimes is different than what is returned by the search engine
            return images 

    image_file_name = find_image_name(src_img_link)
    if image_file_name != '' and image_file_name != 'image':
        attrb_names = ['src','srcset','data-low-res-src','data-original','data-src-full16x9','data-src-mini1x1','data-hi-res-src','data-src-mini','data-src-medium','data-src-large','data-src','data-srcset','data-lazy','data-lazy-src','data-lazy-srcset','data-gl-src','data-gl-srcset','data-raw-src','data-medium-file','data-large-file','data-widths']
        for attr in attrb_names:
            if len(images) == 0:
                images = soup.findAll(lambda tag:tag.name=="img" and tag.has_attr(attr) and image_file_name in tag[attr])
            else:
                break
        if len(images) == 0:
            images = soup.findAll(lambda tag:tag.name=="media-image" and tag.has_attr('image-set') and image_file_name in tag['image-set'])     
    return images 

def process_titles_or_captions(input_text):
    output_text = input_text.rstrip().lstrip()
    return output_text


def get_captions_from_page(src_img_link, url, req_res=None, cutoff=20):
    if req_res == None:
        try:
            r,html,code =  get_html(url,'request')
        except:
            print('Error in request')
            return {},'','5',None
        soup = BeautifulSoup(r.content.decode('utf-8'), "html.parser")
        images = find_img_tag(soup,src_img_link)
    else:
        r = req_res 
        code = r.status_code
        soup = BeautifulSoup(r.content.decode('utf-8'), "html.parser")
        images = find_tags_by_matching(src_img_link, url, soup, cutoff)
        
    caption = {}
    title = ''
    if soup.title:
        title = soup.title.string
    
    for img in images:  
        next_imgs = img.find_next('figcaption')
        if next_imgs:
            for node in list(next_imgs):
                #https://news.sky.com/video/fall-out-over-scottish-referendum-10386472
                #sky news                     
                if '<span class="sdc-site-video__caption-text"' in str(node):
                    caption['caption_node'] = str(node.string)
                    break 
                #https://www.justapinch.com/recipes/dessert/cake/inside-out-german-chocolate-bundt-cake.html?utm_source=CNHI&utm_medium=curatorcrowd&utm_campaign=TRX&utm_content=www.gloucestertimes.com
                if '<span class="caption-text"' in str(node):
                    if '<p>' in str(node):
                        caption['caption_node'] = str(node).split('<p>')[-1].split('</p>')[0]
                        break
                #https://www.telegraph.co.uk/news/worldnews/middleeast/syria/11150024/Air-strikes-might-not-be-enough-to-save-Kobane-from-Isil-US-warns.html
                #telegraph
                if '<span' in str(node) and 'data-test="caption"' in str(node):
                    caption['caption_node'] = str(node.string)
                    break                   
                #Example: https://www.independent.ie/style/celebrity/celebrity-news/meet-the-new-prince-kate-middleton-and-prince-william-introduce-baby-son-to-the-world-36834554.html
                if len(node)>0 and not '<span' in str(node) and node.string:
                    if node.string:
                        caption['caption_node'] = str(node.string.strip()).strip()
                        break
                #NYtimes
                #Example: https://www.nytimes.com/2020/04/10/world/americas/venezuela-pregnancy-birth-death.html
                if '<span aria-hidden' in str(node):
                    if node.string:  
                        caption['caption_node'] = str(node.string.strip())
                        break 
                #Example: https://www.nytimes.com/2014/11/09/world/europe/where-berlin-wall-once-stood-lights-now-illuminate.html
                if '<span class="css-16f3y1r e13ogyst0"' in str(node):
                    if node.string:  
                        caption['caption_node'] = str(node.string.strip())
                        break 
                #Example: https://www.mirror.co.uk/money/breaking-ikea-set-re-open-21981857     
                #Example: https://nationalpost.com/news/the-berlin-wall-has-almost-been-down-as-long-as-it-was-up
                if 'span class="caption"' in str(node):  
                    if node.string: 
                        caption['caption_node'] = str(node.string.strip())
                        break                     
                #statnews 
                #Example: https://www.statnews.com/2017/02/08/cuba-doctors-meager-pay/
                if '<span class="media-caption"' in str(node):
                    if node.string:
                        caption['caption_node'] = str(node.string)
                        break 
                #Example: https://www.hollywoodreporter.com/lifestyle/shopping/scott-foley-ellens-next-great-designer-best-furniture-1234955069/
                if 'span class="a-font-secondary-s lrv-u-margin-r-025"' in str(node):
                    if node.string:
                        caption['caption_node'] = str(node.string) 
                        break 
                    #Example: https://www.hollywoodreporter.com/news/general-news/sochi-paralympic-winter-games-begin-686837/
                    elif '<p>' in str(node):
                        caption['caption_node'] = str(node).split('<p>')[-1].split('</p>')[0]
                        break
                if '<span class' in str(node):
                    #Example: https://www.newyorker.com/magazine/2016/01/18/the-front-lines
                    if 'sc-pNWdM sc-jrsJWt sc-ezzafa lfZoIg fPQsnI eidwJs caption__text' in str(node):
                        if node.string:
                            caption['caption_node'] = str(node.string).strip()
                            break 
                    #NBC: https://www.nbcnews.com/news/world/kids-revel-large-mud-pit-during-michigan-ritual-flna874580
                    if 'caption__container' in str(node):
                        caption['caption_node'] = str(node.string.strip())
                        break
                    #Gaurdian
                    #Example: https://www.theguardian.com/world/2021/may/10/dozens-injured-in-clashes-over-israeli-settlements-ahead-of-jerusalem-day-march
                    if node.nextSibling and '<span class' in str(node.nextSibling):
                        if node.nextSibling.string:
                            caption['caption_node'] = str(node.nextSibling.string.strip())
                            break
                    #BBC
                    #Example: https://www.bbc.com/news/world-africa-57055014
                    elif node.nextSibling and not '<span class' in str(node.nextSibling):
                        if str(node.nextSibling).strip():
                            caption['caption_node'] = str(node.nextSibling.string)
                            break 
            
        #for usa today 
        if img.has_attr('image-alt') and len(img['image-alt'])>0:
            caption['media_image_image_alt'] = str(img['image-alt'])
            
        if img.has_attr('caption') and len(img['caption'])>0:
            caption['media_image_caption'] = str(img['caption'])
        
        if img.has_attr('alt') and len( img['alt'])>0 :
            caption['alt_node'] = str(img['alt'])   

        if img.has_attr('title') and len( img['title'])>0 :
            caption['title_node'] = str(img['title'])   
            
        if img.has_attr('data-caption') and len(img['data-caption'])>0:
            caption['img_data_caption_attr'] = str(img['data-caption']) 
            
        #Example: http://edition.cnn.com/2009/TECH/space/10/28/nasa.ares.rocket/index.html 
        next_nodes = img.find_next('div',{'class':'cnn_strycaptiontxt'})
        if next_nodes:
            for node in list(next_nodes):
                if node.string:
                    caption['div_cnn_strycaptiontxt'] = str(node.string)
                    break
            
        #Example:https://www.reuters.com/business/green-thumb-revenue-zooms-surging-demand-weed-based-products-2021-05-12/  
        next_nodes = img.find_next('p',{'id':'primary-image-caption'})
        if next_nodes:
            for node in list(next_nodes):
                if node.string:
                    caption['p_primary-image-caption'] = str(node.string)
                    break
            
        #Example: url = 'https://www.washingtonpost.com/politics/flow-of-illegal-immigration-slows-as-us-mexico-border-dynamics-evolve/2015/05/27/c5caf02c-006b-11e5-833c-a2de05b6b2a4_story.html'
        next_nodes = img.find_next('span',{'class':'pb-caption'})
        if next_nodes:
            for node in list(next_nodes):
                if node.string:
                    caption['pb-caption'] = str(node.string)
                    break 
        #Example: https://www.ibtimes.co.in/al-shabaab-militants-briefly-take-over-garissa-mosques-deliver-sermons-worshippers-633320
        next_nodes = img.find_next('span',{'class':'cap'})
        if next_nodes:
            for node in list(next_nodes):
                if node.string:
                    caption['class_cap'] = str(node.string)
                    break 
        #Example: https://www.foxnews.com/science/nasas-ares-1-x-vs-the-worlds-tallest-rockets
        next_nodes = img.find_next('p',{'class':'speakable'})
        if next_nodes:
            for node in list(next_nodes):
                if node.string:
                    caption['p_class_speakable'] = str(node.string).strip()
                    
        #Example: https://www.timesofisrael.com/idf-strikes-gaza-targets-in-response-to-earlier-rocket-fire/
        next_nodes = img.find_next('div',{'class':'caption'})
        if next_nodes:
            for node in list(next_nodes):
                if node.string:
                    caption['div_caption'] = str(node.string).strip()
                    #Example: https://www.staradvertiser.com/2012/10/15/breaking-news/pakistani-girl-shot-by-taliban-now-in-uk-for-care/
                    if 'staradvertiser' in url and node.nextSibling.nextSibling.string:
                        caption['div_caption'] = caption['div_caption'] + '. ' + str(node.nextSibling.nextSibling.string).strip()
                    break 
        #Example: https://www.wfdd.org/story/paris-security-checks-shoppers-%E2%80%94-and-children
        next_nodes = img.find_next('div',{'class':'field-caption'})
        if next_nodes:
            for node in list(next_nodes):
                if node.string:
                    caption['div_field_caption'] = str(node.string).strip()
                    break
                    
        next_nodes = img.find_next('span',{'class':'article-image-credit'})
        if next_nodes:
            for node in list(next_nodes):
                if node.string:
                    caption['span_img_credit'] = str(node.string)
                    break 
                
        #EXAMPLE: https://www.thenation.com/article/world/cuba-doctors-covid-19/
        next_nodes = img.find_next('p',{'class':'caption'})
        if next_nodes:
            for node in list(next_nodes):
                if node.string:
                    caption['p_class_caption'] = str(node.string)
                    break 
        #Example: https://somaliagenda.com/somalias-al-shabab-enters-kenyan-village/ 
        next_nodes = img.find_next('p',{'class':'wp-caption-text'})
        if next_nodes:
            for node in list(next_nodes):
                if node.string:
                    caption['p_wp_caption_text'] = str(node.string)
                    break 
                    
        #Example: https://www.csmonitor.com/World/Asia-South-Central/2012/1015/The-Malala-moment-6-Pakistani-views-on-the-girl-shot-by-the-Taliban/I-want-my-daughter-to-love-my-faith-so-she-will-not-visit-Pakistan
        next_nodes = img.find_next('div',{'class':'eza-caption'})
        if next_nodes:
            for node in list(next_nodes):
                if node.string:
                    caption['div_eza_caption'] = str(node.string)
                    break
        #Example: https://www.tennessean.com/picture-gallery/news/2017/05/24/how-ikea-has-grown-over-the-years/102118034/
        next_nodes = img.find_next('div',{'id':'imageCaption'})
        if next_nodes:
            for node in list(next_nodes):
                if node.string:
                    caption['div_id_img_caption'] = str(node.string)
                    break
        #Example: https://www.dailymail.co.uk/news/article-3340215/Mexican-authorities-confirm-burned-van-two-charred-bodies-inside-belonged-missing-Australian-surfers.html
        next_nodes = img.find_next('p',{'class':'imageCaption'})
        if next_nodes:
            for node in list(next_nodes):
                if node.string:
                    caption['pclass_img_caption'] = str(node.string)
                    break
                    
        #Example: https://www.news18.com/news/sports/speculation-over-tokyo-olympics-2021-2032-or-not-at-all-3328064.html 
        next_nodes = img.find_next('p',{'class':'jsx-58535537 imageCaption'}) 
        if next_nodes:
            for node in list(next_nodes):
                if node.string:
                    caption['p_class_img_caption'] = str(node.string)
                    break
        #Example: https://www.stripes.com/news/middle-east/pentagon-more-us-fire-bases-could-open-in-iraq-1.403096 
        next_nodes = img.find_next('div',{'class':'caption_for_main'}) 
        if next_nodes:
            for node in list(next_nodes):
                if node.string:
                    caption['div_class_caption_for_main'] = str(node.string)
                    break 
        #Example: https://www.nytimes.com/interactive/2020/12/23/magazine/breonna-taylor-meme.html 
        next_nodes = img.find_next('span',{'class':'rad-caption-text'}) 
        if next_nodes:
            for node in list(next_nodes):
                if node.string:
                    caption['rad_caption_text'] = str(node.string)
                    break   
        #Example: https://www.latimes.com/nation/ct-mosul-iraq-islamic-state-20161018-story.html
        next_nodes = img.find_next('div',{'class':'figure-caption'}) 
        if next_nodes:
            for node in list(next_nodes):
                if node.string:
                    if '<p>' in node.string:
                        caption['div_class_fig-caption'] = str(node.string).split('<p>')[-1].split('</p>')[0]
                        break
                    else:
                        caption['div_class_fig-caption'] = str(node.string)
                        break
                    
        process_dict(caption)
        if caption:
            break
    title = process_titles_or_captions(title)
    return caption, title, code, r