import os


def create_exp_folder():
    if not os.path.exists("run"):
        os.mkdir("run")

    train_folder = os.path.join("run", "train")
    if not os.path.exists(train_folder):
        os.mkdir(train_folder)

    exp_folder = os.path.join(train_folder, "exp")
    if not os.path.exists(exp_folder):
        os.mkdir(exp_folder)
        os.mkdir(os.path.join(exp_folder, "weights"))
        return exp_folder, os.path.join(exp_folder, "weights")

    exp_num = 1
    while True:
        exp_folder_name = f"exp{exp_num}"
        exp_folder = os.path.join(train_folder, exp_folder_name)
        if not os.path.exists(exp_folder):
            os.mkdir(exp_folder)
            os.mkdir(os.path.join(exp_folder, "weights"))
            return exp_folder, os.path.join(exp_folder, "weights")
        exp_num += 1


def create_val_exp_folder():
    if not os.path.exists("run"):
        os.mkdir("run")

    train_folder = os.path.join("run", "predict")
    if not os.path.exists(train_folder):
        os.mkdir(train_folder)

    exp_folder = os.path.join(train_folder, "exp")
    if not os.path.exists(exp_folder):
        os.mkdir(exp_folder)

    exp_num = 1
    while True:
        exp_folder_name = f"exp{exp_num}"
        exp_folder = os.path.join(train_folder, exp_folder_name)
        if not os.path.exists(exp_folder):
            os.mkdir(exp_folder)
            return exp_folder
        exp_num += 1
