# coding=utf-8
"""Snacks Data Set"""

import os

from sklearn import datasets


_CITATION = """
@article{OpenImages2,
  title={OpenImages: A public dataset for large-scale multi-label and multi-class image classification.},
  author={Krasin, Ivan and Duerig, Tom and Alldrin, Neil and Ferrari, Vittorio and Abu-El-Haija, Sami and Kuznetsova, Alina and Rom, Hassan and Uijlings, Jasper and Popov, Stefan and Kamali, Shahab and Malloci, Matteo and Pont-Tuset, Jordi and Veit, Andreas and Belongie, Serge and Gomes, Victor and Gupta, Abhinav and Sun, Chen and Chechik, Gal and Cai, David and Feng, Zheyun and Narayanan, Dhyanesh and Murphy, Kevin},
  journal={Dataset available from https://storage.googleapis.com/openimages/web/index.html},
  year={2017}
}
"""

_DESCRIPTION = "This is a dataset of 20 different types of snack foods that accompanies the book Machine Learning by Tutorials, https://www.raywenderlich.com/books/machine-learning-by-tutorials/v2.0 — Based on images from Google Open Images dataset."

_HOMEPAGE = "https://huggingface.co/datasets/Matthijs/snacks/"

_LICENSE = "cc-by-4.0"

_IMAGES_URL = "https://huggingface.co/datasets/Matthijs/snacks/resolve/main/images.zip"

_NAMES = ["apple", "banana", "cake", "candy", "carrot", "cookie", 
          "doughnut", "grape", "hot dog", "ice cream", "juice", 
          "muffin", "orange", "pineapple", "popcorn", "pretzel",
          "salad", "strawberry", "waffle", "watermelon"]


class Snacks(datasets.GeneratorBasedBuilder):
    """Snacks Data Set"""

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="default",
            version=datasets.Version("0.0.1", ""),
            description="",
        )
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    #"image_file_path": datasets.Value("string"),
                    "image": datasets.Image(),
                    "label": datasets.features.ClassLabel(names=_NAMES),
                }
            ),
            supervised_keys=("image", "label"),
            homepage=_HOMEPAGE,
            citation=_CITATION,
            license=_LICENSE,
        )

    def _split_generators(self, dl_manager):
        archive = os.path.join(dl_manager.download_and_extract(_IMAGES_URL), "data")
        train_path = os.path.join(archive, "train")
        test_path = os.path.join(archive, "test")
        val_path = os.path.join(archive, "val")

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN, gen_kwargs={"files": dl_manager.iter_files(train_path)}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST, gen_kwargs={"files": dl_manager.iter_files(test_path)}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION, gen_kwargs={"files": dl_manager.iter_files(val_path)}
            ),
        ]

    def _generate_examples(self, files):
        for i, file in enumerate(files):
            if os.path.basename(file).endswith(".jpg"):
                yield str(i), {
                    #"image_file_path": file,                    
                    "image": file,
                    "label": os.path.basename(os.path.dirname(file)).lower(),
                }                
