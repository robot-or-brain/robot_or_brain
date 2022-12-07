from pathlib import Path
import krippendorff
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from pandas import DataFrame as df
from IPython.display import display

public_base_dir = Path('../robot_or_brain_public_data/images_by_class')
private_base_dir = Path('../robot_or_brain_private_data/images_by_class')
combined_base_dir = Path('../robot_or_brain_combined_data/images_by_class')


def load_dataset(split, base_dir=combined_base_dir):
    validation_dir = base_dir / split
    class_list = [p.name for p in validation_dir.iterdir()]
    data_lists = [[f, cls] for cls in class_list for f in (validation_dir / cls).iterdir()]
    dataset = df({'paths': [path for path, _ in data_lists], 'y': [cls for _, cls in data_lists]})
    print(f'Loaded {split} set with {len(dataset)} instances.')
    return dataset, class_list


def display_performance_metrics(trues, predicted, class_list):
    class_metrics, general_metrics = calculate_performance_metrics(trues, predicted, class_list)
    display(class_metrics.round(2))
    display(general_metrics.round(2))


def print_performance_metrics(trues, predicted, class_list):
    class_metrics, general_metrics = calculate_performance_metrics(trues, predicted, class_list)
    print(class_metrics.round(2))
    print(general_metrics.round(2))


def calculate_performance_metrics(trues, predicted, class_list):
    class_metrics_data = {'recall': recall_score(trues, predicted, average=None),
                          'precision': precision_score(trues, predicted, average=None),
                          'f1': f1_score(trues, predicted, average=None)}
    class_metrics = df(class_metrics_data, index=class_list)
    general_metrics_data = [accuracy_score(trues, predicted),
                            krippendorff.alpha(
                                reliability_data=[[list(class_list).index(label) for label in trues],
                                                  [list(class_list).index(label) for label in predicted]])]
    general_metrics = df(general_metrics_data, index=['accuracy', 'krippendorff alpha'], columns=['score'])
    return class_metrics, general_metrics
