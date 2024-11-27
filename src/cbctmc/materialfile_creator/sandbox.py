from material_data import MaterialData


def run():
    data_example = MaterialData(
        "muscle",
        "0.102:H_0.142:C_0.034:N_0.711:O_0.001:Na_0.002:P_0.003:S_0.001:Cl_0.004:K",
        1.05,
        125000,
        "/home/crohling/Documents/test_mat",
    )
    data_example.createMaterialDataFile()


if __name__ == "__main__":
    run()
