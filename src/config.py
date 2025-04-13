import os
from torchvision import transforms
from typing import Optional


class DatasetConfig:
    def __init__(self, dataset_name: str, attack_name: Optional[str] = None, target_class: Optional[str] = None):
        assert dataset_name in ['deepfake', 'dogs-vs-cats']
        assert not attack_name or attack_name in ['FGSM', 'PGD']
        assert not target_class or target_class in ['real', 'fake', 'dog', 'cat']

        self.dataset_name = dataset_name
        self.target_class = target_class.lower() if target_class else target_class
        self.attack_name = attack_name.lower() if attack_name else attack_name
        self._dataset_save_name = dataset_name.replace('-', '_')
        self._dataset_base_path = 'dataset'
        self._heatmap_dataset_base_path = os.path.join(self._dataset_base_path, 'heatmap')
        self._model_base_path = 'model'
        self._results_base_path = 'results'

        self._create_required_dirs()

    def _create_required_dirs(self):
        required_dirs = [
            self._dataset_base_path,
            self._heatmap_dataset_base_path,
            self._model_base_path,
            self._results_base_path,
            os.path.join(self._results_base_path, 'figs'),
        ]

        for path in required_dirs:
            os.makedirs(path, exist_ok=True)

    def _check_attack_set(self):
        if not self.attack_name:
            raise ValueError('No attack set')

    def get_heatmap_dataset_path(self):
        if self.target_class:
            return os.path.join(
                self._heatmap_dataset_base_path,
                f'{self._dataset_save_name}_{self.target_class}_hm_dataset.pt',
            )
        return os.path.join(self._heatmap_dataset_base_path, f'{self._dataset_save_name}_hm_dataset.pt')

    def get_vae_save_path(self):
        if self.target_class:
            return os.path.join(self._model_base_path, f'{self._dataset_save_name}_{self.target_class}_hm_vae.pt')
        return os.path.join(self._model_base_path, f'{self._dataset_save_name}_hm_vae.pt')

    def get_classifier_save_path(self):
        return os.path.join(self._model_base_path, f'{self._dataset_save_name}_classifier.pt')

    def get_classifier_dataset_split(self, split: str):
        assert split in ['train', 'validation']
        return os.path.join(self._dataset_base_path, self.dataset_name, split)

    def get_classifier_transform(self):
        if self.dataset_name == 'deepfake':
            return transforms.Compose(
                [
                    transforms.Resize(224),
                    transforms.ToTensor(),
                ]
            )
        if self.dataset_name == 'dogs-vs-cats':
            return transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                ]
            )
        raise ValueError(f'No transform for {self.dataset_name}')

    def get_vae_results_save_path(self):
        self._check_attack_set()
        if self.target_class:
            return os.path.join(self._results_base_path, f'{self._dataset_save_name}_{self.attack_name}_{self.target_class}_scores.pt')
        return os.path.join(self._results_base_path, f'{self._dataset_save_name}_{self.attack_name}_scores.pt')

    def get_vae_figs_save_path(self):
        self._check_attack_set()
        if self.target_class:
            return os.path.join(self._results_base_path, 'figs', f'{self._dataset_save_name}_{self.attack_name}_{self.target_class}.png')
        return os.path.join(self._results_base_path, 'figs', f'{self._dataset_save_name}_{self.attack_name}.png')

    def get_classes(self):
        if self.dataset_name == 'deepfake':
            return ["Fake", "Real"]
        else:
            return ["Cat", "Dog"]
