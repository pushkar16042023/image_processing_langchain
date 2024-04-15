from transformers import BlipProcessor, BlipForConditionalGeneration, AutoImageProcessor, DetrForObjectDetection
from PIL import Image
import torch
def get_image(image_path):
    raw_image = Image.open(image_path).convert('RGB')
    device = torch.device('cpu')
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(device)
    input = processor(raw_image,return_tensors = 'pt').to(device)
    output = model.generate(**input)
    
    caption = processor.decode(output[0],skip_special_tokens=True)
    return caption
    
def get_obj(image_path):
    
    image = Image.open(image_path).convert('RGB')
    image_processor = AutoImageProcessor.from_pretrained("facebook/detr-resnet-50")
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
    inputs = image_processor(images=image,return_tensors = "pt")
    outputs = model(**inputs)
    
    target_sizes = torch.tensor([image.size[::-1]])
    results = image_processor.post_process_object_detection(outputs,threshold=0.9,target_sizes = target_sizes)[0]
    
    
    detections = ""
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        detection_str = (
            f"Detected {model.config.id2label[label.item()]} with confidence "
            f"{round(score.item(), 3)} at location {box}\n"
        )
        detections += detection_str
    return detections
if __name__ == '__main__':
    image_path = "C:\Users\LENOVO\Downloads\pexels-helena-lopes-1996333.jpg"
    detections = get_obj(image_path)
    print(detections)
    
