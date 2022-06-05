from os.path import join

FULL_DATASET_PATH : str = "./full_dataset"

LABELS_XML_FILEPATH : str = join(FULL_DATASET_PATH, "labels.xml")

DATASET_PATH : str = "./dataset"

NB_IMAGES : int = 2200

NB_TRAIN_IMAGES : int = 2000

NB_TEST_IMAGES : int = 200

TRAIN_XML_FILEPATH : str = join(DATASET_PATH, "train.xml")

TEST_XML_FILEPATH : str = join(DATASET_PATH, "test.xml")

OUTPUT_MODEL_PATH: str = "./output_model.dat"