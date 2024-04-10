import yaml


def f(a, **kwargs):
    print(a)
    print(kwargs)


with open(
    "/home/davidwong/documents/FruitDetector/configs/fruit_dataset.yaml", "r"
) as file:
    data = yaml.safe_load(file)
    print(data["augmentations"])
    print(type(data["augmentations"]))

f(1, **data)
