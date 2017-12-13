from importlib.util import find_spec


def check_modules_installed(module_name_list):
    for module_name in module_name_list:
        found = find_spec(module_name)
        if found is None:
            print("modules not found")
            return False
    return True