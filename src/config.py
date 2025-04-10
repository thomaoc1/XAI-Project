import os

from torchvision import transforms


class DatasetConfig:
    def __init__(self, dataset_name: str, attack_name: str | None = None, target_class: str | None = None):
        assert dataset_name in ['deepfake', 'dogs-vs-cats']
        assert not attack_name or attack_name in ['FGSM', 'PGD']

        self.dataset_name = dataset_name
        self._target_class = target_class.lower() if target_class else target_class
        self.attack_name = attack_name.lower() if attack_name else attack_name
        self._dataset_save_name = dataset_name.replace('-', '_')
        self._dataset_base_path = 'dataset'
        self._heatmap_dataset_base_path = os.path.join(self._dataset_base_path, 'heatmap')
        self._model_base_path = 'model'
        self._results_base_path = 'results'

    def _check_attack_set(self):
        if not self.attack_name:
            raise ValueError('No attack set')

    def get_heatmap_dataset_path(self):
        return os.path.join(self._heatmap_dataset_base_path, f'{self._dataset_save_name}_hm_dataset.pt')

    def get_vae_save_path(self):
        return os.path.join(self._model_base_path, f'{self._dataset_save_name}_hm_vae.pt')

    def get_classifier_save_path(self):
        return os.path.join(self._model_base_path, f'{self._dataset_save_name}_classifier.pt')

    def get_classifier_dataset_split(self, split: str):
        assert split in ['train', 'validation']
        return os.path.join(self._dataset_base_path, self.dataset_name, split)

    def get_classifier_transform(self):
        if self.dataset_name == 'deepfake':
            return transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
            ])
        if self.dataset_name == 'dogs-vs-cats':
            return transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ])
        else:
            raise ValueError(f'No transform for {self.dataset_name}')

    def get_vae_results_save_path(self):
        self._check_attack_set()
        return os.path.join(self._results_base_path, f'{self._dataset_save_name}_{self.attack_name}_scores.pt')

    def get_vae_figs_save_path(self):
        self._check_attack_set()
        return os.path.join(self._results_base_path, 'figs', f'{self._dataset_save_name}_{self.attack_name}.png')

    def get_vae_transform(self):
        if self.dataset_name == 'deepfake':
            return transforms.Compose([])
        if self.dataset_name == 'dogs-vs-cats':
            return transforms.Compose([])
        else:
            raise ValueError(f'No transform for {self.dataset_name}')