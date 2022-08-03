from pathlib import Path
import krippendorff
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from pandas import DataFrame as df

base_dir = Path('../robot-or-brain-data/robot-or-brain-data/images_by_class')


def load_dataset(split, current_base_dir=base_dir):
    validation_dir = current_base_dir / split
    class_list = [p.name for p in validation_dir.iterdir()]
    data_lists = [[f, cls] for cls in class_list for f in (validation_dir / cls).iterdir()]
    dataset = df({'paths': [path for path, _ in data_lists], 'y': [cls for _, cls in data_lists]})
    return dataset, class_list


def print_performance_metrics(trues, predicted, class_list):
    print('accuracy_score', accuracy_score(trues, predicted))
    print('recall_score', recall_score(trues, predicted, average=None))
    print('precision_score', precision_score(trues, predicted, average=None))
    print('f1_score', f1_score(trues, predicted, average=None))
    print('krippendorff.alpha', krippendorff.alpha(reliability_data=[[class_list.index(label) for label in trues],
                                                                     [class_list.index(label) for label in predicted]]))
