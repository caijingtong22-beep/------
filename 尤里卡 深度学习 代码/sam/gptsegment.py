from ultralytics import SAM

# 1. 加载 SAM / SAM2
# model = SAM("sam_b.pt")      # 原版 SAM
model = SAM("sam2_b.pt")       # SAM2，也可以用 sam2_t.pt / sam2_s.pt / sam2_l.pt

img = "test.jpg"


# -----------------------------
# Prompt 1：框选一个物体 bbox
# 格式：[x1, y1, x2, y2]
# -----------------------------
results_box = model.predict(
    source=img,
    bboxes=[100, 120, 400, 500],
    save=True,
    project="sam_outputs",
    name="bbox_prompt"
)


# -----------------------------
# Prompt 2：用一个前景点分割
# points: [x, y]
# labels: 1 表示这个点属于目标物体
# -----------------------------
results_point = model.predict(
    source=img,
    points=[250, 300],
    labels=[1],
    save=True,
    project="sam_outputs",
    name="point_prompt"
)


# -----------------------------
# Prompt 3：前景点 + 背景点
# 第一个点告诉 SAM：这里是目标
# 第二个点告诉 SAM：这里不是目标
# -----------------------------
results_pos_neg = model.predict(
    source=img,
    points=[[250, 300], [420, 280]],
    labels=[1, 0],
    save=True,
    project="sam_outputs",
    name="positive_negative_points"
)


# -----------------------------
# Prompt 4：多个 bbox，一次分割多个物体
# -----------------------------
results_multi_box = model.predict(
    source=img,
    bboxes=[
        [100, 120, 400, 500],
        [450, 150, 700, 520]
    ],
    save=True,
    project="sam_outputs",
    name="multi_bbox_prompt"
)


# -----------------------------
# 查看某次结果
# -----------------------------
for r in results_box:
    print(r.masks)     # 分割 mask
    print(r.boxes)     # mask 对应的 bbox
    r.show()           # 弹窗显示