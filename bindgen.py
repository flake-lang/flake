import sys
import re

types = {
    "int": "i32",
    "unsigned_int": "u32",
    "double": "f64",
    "float": "f32",
    "FILE": "u64",
    "_Decimal32": "u32",
    "_Decimal64": "u64",
    "_Decimal128": "u128",
    "wchar_t": "u32",
    "long_int": "u64",
    "long": "i64",
    "wint_t": "wchar_t",
    "va_list": "*void",
    "var_name": "str",
    "__struct_ptr_t": "*void"
}

FUNCTIONS = list(sys.argv)

FUNCTIONS.pop()

print(f"Generating bindings for function(`libc-modified.h`): {', '.join(FUNCTIONS)}")

with open("libc.h") as header:
    with open("libc.fl", 'w') as output:
        for prototype in header.readlines():
            return_ty = '_'.join(prototype.split("(")[0].split(" ")[0:-1])

            func_name = prototype.split("(")[0].split(" ")[-1]

            pointer_specs = len(func_name)
            func_name = func_name.lstrip('*')
            pointer_specs -= len(func_name)

 

            args_str = '('.join(prototype.split("(")[1:]).rstrip("\);\n").replace("const", "")

            if return_ty == "struct":
                if pointer_specs > 0:
                    return_ty = "__struct_pointer_t"
                    pointer_specs -= 1
                else:
                    print(f"Cannot return structs, skipping... {func_name}!")
                    continue

            args = []

            if not func_name.strip() in FUNCTIONS: 
                continue
            else:
                print("[Started] {}", func_name)

            for arg_str in args_str.split(', '):
                arg_sep = arg_str.strip().split(' ')
                arg_ty = '_'.join(arg_sep[0:-1])
                if len(arg_sep) <= 1:
                    arg_ty = arg_sep[0]
                if len(arg_sep) == 2:
                    arg_name = arg_sep[-1]
                else:
                    arg_name = f"arg{len(args)}"

                arg_ptr_specs = len(arg_name)
                arg_name = arg_name.lstrip('*')
                arg_ptr_specs -= len(arg_name)

                if arg_name == "struct":
                    if arg_ptr_specs > 0:
                        arg_name = "__struct_type"
                        arg_ptr_specs -= 1
                    else:
                        print(f"Cannot use structs, skipping... {func_name}::{arg_name}!")
                        continue

                if arg_ty == "void":
                    continue
                if arg_ty == "...":
                    arg_ty = "*void"

                if arg_ty in types.keys():
                    arg_ty = types[arg_ty]
                else:
                    types.update({arg_ty: arg_ty})

                args.append(f"{arg_name}: {arg_ptr_specs*'*'}{arg_ty}")

            if len(func_name) == 0:
                continue

            
            if return_ty in types.keys():
                return_ty = types[return_ty]
            else:
                types.update({return_ty: return_ty})

            # ... (write)

            print("[Done] {}", func_name)

            output.write(f"// {prototype}fn {func_name}[{', '.join(args)}]: {pointer_specs*'*'}{return_ty};\n");

        for (alias, ty) in types.items():
            output.write(f"type {alias} = {ty};\n")


