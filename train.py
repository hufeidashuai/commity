import os.path as osp
import pandas as pd
from modelscope.trainers import build_trainer
from modelscope.msdatasets import MsDataset
from modelscope.utils.hub import read_config
from modelscope.metainfo import Metrics


model_id = 'damo/nlp_structbert_sentiment-classification_chinese-base'
dataset_id = 'jd'
train_path='./concat_traindataset.csv'
eval_path='./concat_evaldataset.csv'

WORK_DIR = 'workspace'

max_epochs = 2
def cfg_modify_fn(cfg):
    cfg.train.max_epochs = max_epochs
    cfg.train.hooks = [{
            'type': 'TextLoggerHook',
            'interval': 100
        }, {
            "type": "CheckpointHook",
            "interval": 1
        }]
    cfg.evaluation.metrics = [Metrics.seq_cls_metric]
    cfg['dataset'] = {
        'train': {
            'labels': ['负面', '正面'],
            'first_sequence': 'sentence',
            'label': 'label',
        }
    }
    cfg.train.optimizer.lr = 3e-5
    return cfg


# train_dataset = MsDataset.load(dataset_id, namespace='DAMO_NLP', split='train').to_hf_dataset()
# print("train_dataset:",train_dataset)
# train_dataset.to_csv("./jd.csv")

# eval_dataset = MsDataset.load(dataset_id, namespace='DAMO_NLP', split='validation').to_hf_dataset()
#转换为huggingface dataset对象
train_dataset = MsDataset.load(train_path).to_hf_dataset()

eval_dataset = MsDataset.load(eval_path).to_hf_dataset()

train_dataset=train_dataset.remove_columns(["Unnamed: 0"])
eval_dataset=eval_dataset.remove_columns(["Unnamed: 0"])
# eval_dataset = MsDataset.load(dataset_id, namespace='DAMO_NLP', split='validation')
print('1')
print("train_dataset",train_dataset)
print('2')
print("eval_dataset",eval_dataset)

# eval_dataset.to_csv("./jd_eval.csv")
# print("eval_dataset:",eval_dataset)

# train_dataset=pd.read_csv(train_path)
# eval_dataset=pd.read_csv(eval_path)

# features = ['sentence', 'label']

# # 创建一个新的 DataFrame 对象
# dataset = pd.DataFrame({
#    'features': features,
#    'num_rows': len(df)
# })

# train_dataset=
# remove useless case
# train_dataset = train_dataset.filter(lambda x: x["label"] != None and x["sentence"] != None)
# eval_dataset = eval_dataset.filter(lambda x: x["label"] != None and x["sentence"] != None)

# map float to index
def map_labels(examples):
    map_dict = {0: "负面", 1: "正面"}
    examples['label'] = map_dict[int(examples['label'])]
    # examples['label'] = map_dict[examples['label']]
    return examples

train_dataset = train_dataset.map(map_labels)
eval_dataset = eval_dataset.map(map_labels)
kwargs = dict(
    model=model_id,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    work_dir=WORK_DIR,
    cfg_modify_fn=cfg_modify_fn
)


trainer = build_trainer(name='nlp-base-trainer', default_args=kwargs)

print('===============================================================')
print('pre-trained model loaded, training started:')
print('===============================================================')

trainer.train()

print('===============================================================')
print('train success.')
print('===============================================================')

for i in range(max_epochs):
    eval_results = trainer.evaluate(f'{WORK_DIR}/epoch_{i+1}.pth')
    print(f'epoch {i} evaluation result:')
    print(eval_results)


print('===============================================================')
print('evaluate success')
print('===============================================================')
