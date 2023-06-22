import gradio as gr
import cv2
from model import FasterRCNN
import torch
import numpy as np

classes=["Background","Messi","Ronaldo","Neymar"]
num_classes=len(classes)
detection_threshold=0.85


model=FasterRCNN(num_classes=num_classes)
model.load_state_dict(torch.load('faster_rcnn.pth'))
model.eval()


def show_preds_image(image_path):
    image=cv2.imread(image_path)
    image_copy=image.copy()
    image = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB).astype(np.float32)

    image /= 255.0
    image = np.transpose(image, (2, 0, 1)).astype(np.float64)
    image = torch.tensor(image, dtype=torch.float)
    image = torch.unsqueeze(image, 0)
    with torch.no_grad():
        outputs=model(image)
    outputs=[{k:v.to('cpu') for k,v in t.items()} for t in outputs]
    if len(outputs[0]['boxes']) !=0:
        boxes=outputs[0]['boxes'].data.numpy()
        scores=outputs[0]['scores'].data.numpy()
        boxes=boxes[scores>=detection_threshold].astype(np.int32)
        make_boxes=boxes.copy()
        pred=[classes[i] for i in outputs[0]['labels'].cpu().numpy()]
        
        for i,box in enumerate(make_boxes):
            cv2.rectangle(image_copy,
                        (int(box[0]),int(box[1])),
                        (int(box[2]),int(box[3])),
                        (0,0,255),2)
            cv2.putText(image_copy,pred[i],
                        (int(box[0]),int(box[1]-5)),
                        cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),
                        2,lineType=cv2.LINE_AA)
            
    return cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)

inputs_image = [
    gr.components.Image(type="filepath", label="Input Image"),
]
outputs_image = [
    gr.components.Image(type="numpy", label="Output Image"),
]
interface_image = gr.Interface(
    fn=show_preds_image,
    inputs=inputs_image,
    outputs=outputs_image,
    title="Facial Recogntition(Messi,Ronaldo,Neymar)",
    cache_examples=False,
)

gr.TabbedInterface(
    [interface_image],
    tab_names=['Image inference']
).queue().launch()