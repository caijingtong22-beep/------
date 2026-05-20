#!/usr/bin/env python
# encoding: utf-8
'''
@author: youzeshun
@file: a.py
@time: 2021/9/7 12:05 AM
'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

def show_cluster_image(vectors, labels, excepted_labels=None):
    """
    根据数据绘制出聚类散点图，目前最多8个类别
    :param vectors: 向量
    :param labels: 该点属于哪个簇
    :param excepted_labels: 排除的标签，该标签不绘制
    :return:
    """
    # 降维
    estimator = PCA(n_components=2)
    data_set = estimator.fit_transform(vectors)
    # 分成若干个簇
    clusters = {}
    for index in range(len(data_set)):
        datum = data_set[index]
        # 标签所代表的簇
        label = labels[index]
        # 异常值目前不显示
        if excepted_labels and label in excepted_labels:
            continue
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(datum)
    # 遍历簇
    for label, array in clusters.items():
        matrix = np.array(array)
        plt.scatter(matrix[:, 0], matrix[:, 1], label='cluter%d' % label)

    plt.legend(loc='upper right')
    plt.show()

# Corpus with example sentences
corpus = [
    "直肠癌术后,腰痛,失眠,高血压,冠状动脉性心脏病   高血压性脑病",
    "胰头穿刺物坏死组织中及散在少数异型腺体    多腺体功能亢进",
    "右环指软组织坏死缺损骨外露  手坏死",
    "子宫平滑肌瘤，伴腺瘤样瘤性平滑肌瘤和腺肌症；增生期样子宫内膜；慢性宫颈炎。  血管平滑肌瘤",
    "左侧气胸41双侧胸部颈部皮下气肿   操作后皮下气肿",
    "右腿截肢右腿功能障碍 睾丸功能障碍",
    "左上肺周围型低分化腺癌纵隔淋巴结转移 上呼吸道感染  呼吸道感染",
    "食道高-中分化鳞癌粒子植入术后    自主性高反射",
    "多脏器功能不全累及呼吸心肌肝肾凝血代谢等   手术后慢性肺功能不全",
    "2左侧11肋后肋骨折 肋骨多处骨折",
    "鼻咽癌根治性放射治疗 鼻咽恶性肿瘤",
    "双侧基底节区小腔隙灶脑白质脱髓鞘改变 基底节脑梗死",
    "直肠肿物性质待查   直肠脓肿",
    "慢性肺源性心脏心功能失代偿期 失认",
    "双侧甲状腺乳头状微小癌    微乳头状浆液性癌",
    "骨盆尤文氏肉瘤    骨盆部肿瘤",
    "右肾盂输尿管占位   囊性肾盂输尿管炎",
    "上呼吸道感染慢性咽炎感冒,口腔黏膜溃疡,胃镜检查,病毒性感冒 慢性咽炎",
    "急性脑梗死；高血压病；心肌缺血    无症状心肌缺血",
    "低血糖病胰腺内分泌病 内分泌失调",
    "特指手术后状态(右髋骨、手掌骨折术后)    手术后食管瘘",
    "T911椎体压缩性改变部分椎体退变  腺样体肥大",
    "右髋关节骨折脱位   踝关节脱位",
    "卵巢癌化疗后 卵巢损伤",
    "中枢神经系统EB病毒感染   原发中枢神经系统弥漫大B细胞淋巴瘤",
    "妊娠期糖尿病G4P1孕39+4周待产LOA  未特指的妊娠期糖尿病",
    "传染性单核细胞综合征 慢性单核细胞白血病",
    "支气管哮喘继发感染糖尿病高血压病;冠心病   支气管哮喘",
    "上颌前牙区骨内埋伏多生牙   牙折断",
    "狼疮足细胞病 狼疮性脑病"
]
# model = "C:\\Users\\Alice\\.cache\\huggingface\\hub\\models--google-bert--bert-base-uncased"
# model = SentenceTransformer(model)
model = SentenceTransformer(
    r"C:\Users\Alice\.cache\huggingface\hub\models--google-bert--bert-base-uncased\snapshots\86b5e0934494bd15c9632b12f734a8a67f723594"
)

corpus_embeddings = model.encode(corpus)
# Perform kmean clustering
num_clusters = 5
clustering_model = KMeans(n_clusters=num_clusters)
clustering_model.fit(corpus_embeddings)

clustered_sentences = [[] for i in range(num_clusters)]
for sentence_id, cluster_id in enumerate(clustering_model.labels_):
    clustered_sentences[cluster_id].append(corpus[sentence_id])

for i, cluster in enumerate(clustered_sentences):
    print("Cluster ", i)
    print(cluster)
