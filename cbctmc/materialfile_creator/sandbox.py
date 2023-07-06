from  material_data import MaterialData


def run():
    data_example = MaterialData("SalzWasser_nachher_2", "0.9:H2O_0.1:NaCl", 1.0, 125000, "/home/crohling/Documents/test_mat")
    data_example.createMaterialDataFile()



if __name__ == "__main__":
    run()
