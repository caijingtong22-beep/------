# %%
from ultralytics import SAM

# Load a model
model = SAM("/home/youlika/sam_b.pt")

# Display model information (optional)
model.info()

# %%

# Run inference with point prompt
results = model("/home/youlika/code/gitee/yolo/Screenshot.png", points=[185,130])
results[0].save(filename='/home/youlika/output.jpg')  # 可以保存带掩膜的图片

# %%

# Run inference with bboxes prompt
results = model("/home/youlika/code/gitee/yolo/Screenshot.png", bboxes=[100,45,295,256])
results[0].save(filename='/home/youlika/output2.jpg')  # 可以保存带掩膜的图片