def create_material_library(material_path, type="n"):

    f = open(material_path + type + ".mtl", "w")
    f.write("newmtl {}\n".format(type))
    f.write("Ka 0.0000 0.0000 0.0000\n")
    if type=="n":
        f.write("Kd 0.4000 0.4000 0.4000\n")
    elif type=="":
        f.write("Kd 1.0000 1.0000 1.0000\n")
    elif type=="s":
        f.write("Kd 0.2000 1.0000 0.2000\n")
    elif type=="p":
        f.write("Kd 0.0000 0.2000 1.0000\n")
    elif type=="o":
        f.write("Kd 0.0000 0.0000 0.0000\n")
    else:
        print("No material created as type does not exist")
    f.write("Ks 1.0000 1.0000 1.0000\n")
    f.write("Tf 0.0000 0.0000 0.0000\n")
    f.write("d 1.0000\n")
    f.write("Ns 0.0000")
    f.close()