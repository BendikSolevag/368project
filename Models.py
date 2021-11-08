from simple_elmo import ElmoModel


def get_nor_elmo(path="Vectors/218.zip"):
    nor_elmo = ElmoModel()
    nor_elmo.load(path)
    return nor_elmo

def get_nor_bert():
    pass